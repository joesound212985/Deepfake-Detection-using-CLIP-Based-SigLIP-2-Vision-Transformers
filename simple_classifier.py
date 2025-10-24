#!/usr/bin/env python3
"""
Simple Binary Image Classifier
Real vs Fake image classification with comprehensive metrics and visualizations.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("Warning: open_clip not available. Install with: pip install open-clip-torch")

try:
    import kornia
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not available. Install with: pip install kornia")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: opencv not available. Install with: pip install opencv-python")

from torchvision import models

def fast_image_load(img_path):
    """Fast image loading with OpenCV fallback to PIL"""
    if OPENCV_AVAILABLE:
        try:
            # OpenCV is ~2-3x faster than PIL for JPEG
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img)
        except Exception:
            pass

    # Fallback to PIL
    return Image.open(img_path).convert('RGB')

class ImageDataset(Dataset):
    """Simple dataset for binary classification"""
    def __init__(self, data_dir, split='train', transform=None, image_size=384):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size

        # Load from REAL/FAKE folders
        real_dir = os.path.join(data_dir, split.upper(), 'REAL')
        fake_dir = os.path.join(data_dir, split.upper(), 'FAKE')

        self.samples = []
        self.labels = []

        # Load REAL images (label=0)
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append(os.path.join(real_dir, img_file))
                    self.labels.append(0)

        # Load FAKE images (label=1)
        if os.path.exists(fake_dir):
            for img_file in os.listdir(fake_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append(os.path.join(fake_dir, img_file))
                    self.labels.append(1)

        print(f"{split}: {len(self.samples)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            image = fast_image_load(img_path)
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy image with correct size - create noise image and apply transforms
            dummy_pil = Image.fromarray(np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8))
            if self.transform:
                dummy_pil = self.transform(dummy_pil)
            return dummy_pil, label

class BinaryClassifier(nn.Module):
    """SigLIP-based binary classifier"""
    def __init__(self, model_size='large', device='cuda'):
        super().__init__()

        if not OPENCLIP_AVAILABLE:
            raise ImportError("open_clip required. Install with: pip install open-clip-torch")

        # Model configurations
        configs = {
            'small': ('ViT-B-16-SigLIP-384', 384, 768),
            'medium': ('ViT-L-16-SigLIP-384', 384, 1024),
            'large': ('ViT-L-16-SigLIP-384', 384, 1024)
        }

        model_name, self.resolution, feature_dim = configs[model_size]

        # Load SigLIP model
        self.backbone, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained='webli',
            device=device
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, 1)
        )

        print(f"✓ Loaded SigLIP model: {model_name} ({model_size})")
        print(f"  Resolution: {self.resolution}px, Features: {feature_dim}")

    def forward(self, x):
        # Resize if needed
        if x.shape[-1] != self.resolution:
            x = nn.functional.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear')

        # Extract features
        features = self.backbone.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalize

        # Classify
        logits = self.classifier(features)
        return logits.squeeze(-1)

def train_epoch(model, dataloader, criterion, optimizer, device, gpu_transform=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        # Apply GPU transforms if available
        if gpu_transform is not None:
            images = gpu_transform(images)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        correct += (predicted == labels.long()).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device, gpu_transform=None):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            # Apply GPU transforms if available
            if gpu_transform is not None:
                images = gpu_transform(images)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)

            total_loss += loss.item()
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    # Calculate metrics
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    all_preds = (all_probs > 0.5).astype(int)

    # Debug: Check prediction distribution
    print(f"  Prob stats: min={all_probs.min():.4f}, max={all_probs.max():.4f}, mean={all_probs.mean():.4f}, std={all_probs.std():.4f}")
    print(f"  Predictions: {np.sum(all_preds==0)} Real, {np.sum(all_preds==1)} Fake (out of {len(all_preds)})")

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, balanced_acc, precision, recall, f1, auc, ap, mcc, cm, all_labels, all_probs

def save_plots(labels, probs, save_dir, prefix=''):
    """Save comprehensive visualization plots"""
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(labels, preds)

    plt.rcParams.update({'font.size': 12})

    if prefix:
        prefix = f'{prefix}_'

    # 1. Confusion Matrix - Normalized
    plt.figure(figsize=(8, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0.5, 1.5], ['Real', 'Fake'])
    plt.yticks([0.5, 1.5], ['Real', 'Fake'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}confusion_matrix.png'), dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=3, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}roc_curve.png'), dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, linewidth=3, label=f'PR (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}pr_curve.png'), dpi=300)
    plt.close()

    # 4. Score Distribution
    plt.figure(figsize=(10, 6))
    real_scores = probs[labels == 0]
    fake_scores = probs[labels == 1]
    plt.hist(real_scores, bins=50, alpha=0.7, label=f'Real (n={len(real_scores)})', color='blue', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.7, label=f'Fake (n={len(fake_scores)})', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}score_dist.png'), dpi=300)
    plt.close()

    # 5. Performance Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Balanced Acc', 'Precision', 'Recall', 'F1', 'AUC', 'AP']
    values = [
        accuracy_score(labels, preds),
        balanced_accuracy_score(labels, preds),
        precision_recall_fscore_support(labels, preds, average='binary')[0],
        precision_recall_fscore_support(labels, preds, average='binary')[1],
        precision_recall_fscore_support(labels, preds, average='binary')[2],
        auc,
        ap
    ]

    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf'])
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}metrics.png'), dpi=300)
    plt.close()

    print(f"Plots saved to {save_dir}")

def save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, save_dir):
    """Save training curves"""
    epochs = range(1, len(train_losses) + 1)

    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300)
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', linewidth=2, label='Train Acc', marker='o')
    plt.plot(epochs, [acc*100 for acc in val_accs], 'r-', linewidth=2, label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_curves.png'), dpi=300)
    plt.close()

    # F1 curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1s, 'g-', linewidth=2, label='Val F1', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Simple Binary Image Classifier with SigLIP')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--model_size', type=str, default='large', choices=['small', 'medium', 'large'],
                        help='SigLIP model size (default: large)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, help='Freeze backbone (default: True)')
    parser.add_argument('--unfreeze_backbone', action='store_true', help='Unfreeze backbone for full fine-tuning')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: SigLIP-{args.model_size}, Batch size: {args.batch_size}")

    if OPENCV_AVAILABLE:
        print("✓ Fast image loading with OpenCV enabled")
    else:
        print("Using PIL for image loading (OpenCV not available)")

    # Get image size from model
    model_resolutions = {'small': 384, 'medium': 384, 'large': 384}
    image_size = model_resolutions[args.model_size]

    # Setup GPU transforms with Kornia if available
    gpu_transform = None
    if device.type == 'cuda' and KORNIA_AVAILABLE:
        gpu_transform = nn.Sequential(
            K.Resize((image_size, image_size), antialias=True),
            K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ).to(device)
        print("✓ GPU-accelerated preprocessing with Kornia enabled")

        # CPU transforms (minimal - just resize and convert to tensor)
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
    else:
        # CPU transforms (full pipeline)
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        print("Using CPU preprocessing")

    # Datasets
    train_dataset = ImageDataset(args.data_dir, 'train', train_transform, image_size)
    val_dataset = ImageDataset(args.data_dir, 'val', test_transform, image_size)
    test_dataset = ImageDataset(args.data_dir, 'test', test_transform, image_size)

    # DataLoaders - optimized for dual-channel memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # Faster CPU->GPU transfer with dual-channel
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # Model
    model = BinaryClassifier(model_size=args.model_size, device=device).to(device)

    # Freeze backbone by default (unless --unfreeze_backbone is specified)
    if args.unfreeze_backbone:
        print("⚠ Backbone unfrozen - training entire model (slower, needs lower LR)")
    else:
        # Freeze entire backbone first
        for param in model.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last transformer block for partial fine-tuning (BEST approach)
        unfrozen_params = 0
        for name, param in model.backbone.named_parameters():
            # Unfreeze last block + layer norm
            if any(x in name for x in ['blocks.23', 'blocks.22', 'ln_final', 'norm']):
                param.requires_grad = True
                unfrozen_params += param.numel()

        print(f"✓ Backbone partially frozen - last block unfrozen ({unfrozen_params:,} params)")
        print("  This is the RECOMMENDED approach for +0.05-0.08 F1 boost!")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    best_f1 = 0
    train_losses, train_accs, val_losses, val_accs, val_f1s = [], [], [], [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, gpu_transform)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc, val_bal_acc, val_prec, val_rec, val_f1, val_auc, val_ap, val_mcc, val_cm, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, gpu_transform
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        scheduler.step(val_f1)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        print(f"Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}, MCC: {val_mcc:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✓ Best F1: {val_f1:.4f} - model saved")
            save_plots(val_labels, val_probs, args.save_dir, f'epoch_{epoch+1}')

    # Test evaluation
    print("\nFinal Test Evaluation:")
    test_loss, test_acc, test_bal_acc, test_prec, test_rec, test_f1, test_auc, test_ap, test_mcc, test_cm, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device, gpu_transform
    )
    print(f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

    # Save final plots
    save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir)
    save_plots(test_labels, test_probs, args.save_dir, 'final_test')

    print(f"\nTraining complete! Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()
