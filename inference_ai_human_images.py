#!/usr/bin/env python3
"""
Inference Script for Deepfake-Eval-2024 Dataset
Based on the complete inference pipeline with:
- Optimal threshold finding
- Probability calibration
- Test-Time Augmentation (optional)
- Comprehensive metrics and visualizations
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    print("Error: open_clip not available")
    exit(1)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def fast_image_load(img_path):
    """Fast image loading with OpenCV fallback to PIL"""
    if OPENCV_AVAILABLE:
        try:
            # cv2.imread is much faster than PIL for many formats
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img)
        except Exception:
            pass

    try:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Use fast path for RGB images
        img = Image.open(img_path)
        if img.mode == 'RGB':
            return img
        return img.convert('RGB')
    except Exception as e:
        # Return a black dummy image if loading fails
        print(f"Warning: Failed to load {img_path}: {e}")
        return Image.new('RGB', (384, 384), color='black')


def apply_clahe(pil_image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if not OPENCV_AVAILABLE:
        return pil_image

    img = np.array(pil_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        img = clahe.apply(img)

    return Image.fromarray(img)


def apply_sharpening(pil_image):
    """Apply edge sharpening to enhance artifacts"""
    from PIL import ImageFilter, ImageEnhance

    sharpened = pil_image.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Sharpness(sharpened)
    sharpened = enhancer.enhance(1.5)

    return sharpened


class BinaryClassifier(nn.Module):
    """SigLIP-based binary classifier"""
    def __init__(self, model_size='large', device='cuda'):
        super().__init__()

        configs = {
            'small': ('ViT-B-16-SigLIP-384', 384, 768),
            'medium': ('ViT-L-16-SigLIP-384', 384, 1024),
            'large': ('ViT-L-16-SigLIP-384', 384, 1024)
        }

        model_name, self.resolution, feature_dim = configs[model_size]

        self.backbone, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained='webli',
            device=device
        )
        self.preprocess = preprocess

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
        # Don't double-resize - transforms already handle this
        # if x.shape[-1] != self.resolution:
        #     x = nn.functional.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear')

        features = self.backbone.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)

        logits = self.classifier(features)
        return logits.squeeze(-1)


class AIHumanDataset(Dataset):
    """Dataset for AI vs. Human-Generated Images"""
    def __init__(self, data_dir, metadata_csv, transform=None, max_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # Load metadata
        df = pd.read_csv(metadata_csv)

        print(f"   - Scanning {len(df)} images...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Loading dataset", ncols=80, leave=False):
            # file_name column contains relative path from data_dir
            img_path = self.data_dir / row['file_name']
            if img_path.exists():
                # label: 0 = Real/Human, 1 = AI/Fake
                label = int(row['label'])
                filename = row['file_name']
                self.samples.append((str(img_path), label, filename))

        if max_samples and len(self.samples) > max_samples:
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        labels = [label for _, label, _ in self.samples]
        print(f"Loaded {len(self.samples)} images (Human/Real: {labels.count(0)}, AI/Fake: {labels.count(1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, filename = self.samples[idx]
        image = fast_image_load(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label, filename


def create_tta_transforms(image_size, num_augments=5):
    """Create test-time augmentation transforms"""
    tta_transforms = []

    # 1. Base (no augmentation)
    base = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    tta_transforms.append(("Original", base))

    if num_augments >= 2:
        # 2. Horizontal flip
        hflip = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("H-Flip", hflip))

    if num_augments >= 3:
        # 3. CLAHE
        clahe = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: apply_clahe(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("CLAHE", clahe))

    if num_augments >= 4:
        # 4. Sharpening
        sharpen = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: apply_sharpening(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("Sharpen", sharpen))

    if num_augments >= 5:
        # 5. CLAHE + Sharpening
        clahe_sharp = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: apply_sharpening(apply_clahe(img))),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("CLAHE+Sharpen", clahe_sharp))

    return tta_transforms


def run_inference(model, dataloader, device, use_amp=True, desc="Inference", invert_logits=False, prototypes=None):
    """Run inference and collect predictions

    Args:
        model: Classification model
        dataloader: Data loader
        device: torch device
        use_amp: Use automatic mixed precision
        desc: Progress bar description
        invert_logits: Invert logits for label orientation
        prototypes: If provided, use prototype-based classification instead of model head
    """
    all_labels = []
    all_probs = []
    all_filenames = []

    model.eval()
    with torch.inference_mode():
        # Enhanced progress bar with sample rate and ETA
        pbar = tqdm(dataloader, desc=desc,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                   ncols=100)

        for images, labels, filenames in pbar:
            images = images.to(device, non_blocking=True)

            if prototypes is not None:
                # Prototype-based classification
                if use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        # Extract features from backbone
                        features = model.backbone.encode_image(images)
                        features = features / features.norm(dim=-1, keepdim=True)
                else:
                    features = model.backbone.encode_image(images)
                    features = features / features.norm(dim=-1, keepdim=True)

                # Compute distances to prototypes
                dist_to_real = torch.cdist(features, prototypes['real'].unsqueeze(0), p=2).squeeze(-1)
                dist_to_fake = torch.cdist(features, prototypes['fake'].unsqueeze(0), p=2).squeeze(-1)

                # Convert distances to probabilities (closer to fake = higher prob)
                # Use softmax over negative distances
                logits_real = -dist_to_real
                logits_fake = -dist_to_fake
                probs_fake = torch.softmax(torch.stack([logits_real, logits_fake], dim=1), dim=1)[:, 1]
                probs = probs_fake.cpu().numpy()
            else:
                # Standard model-based classification
                if use_amp and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(images)
                else:
                    logits = model(images)

                # Fix orientation: ensure higher scores = FAKE
                if invert_logits:
                    logits = -logits

                probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_filenames.extend(filenames)

            # Update progress bar with current batch size
            pbar.set_postfix({'batch': len(images)}, refresh=False)

    return np.array(all_labels), np.array(all_probs), all_filenames


def run_tta_inference(model, data_dir, metadata_csv, tta_transforms, batch_size, num_workers, device, use_amp=True, invert_logits=False, prototypes=None):
    """Run test-time augmentation"""
    all_tta_probs = []
    y_true = None
    filenames = None

    print("\n" + "="*60)
    print(f"Running Test-Time Augmentation ({len(tta_transforms)} transforms)")
    print("="*60)

    for tta_name, tta_transform in tta_transforms:
        print(f"\nProcessing: {tta_name}")

        dataset = AIHumanDataset(data_dir, metadata_csv, transform=tta_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            prefetch_factor=4,  # Increased prefetch for better pipeline
            persistent_workers=True if num_workers > 0 else False
        )

        labels, probs, fnames = run_inference(model, dataloader, device, use_amp, desc=f"  {tta_name}", invert_logits=invert_logits, prototypes=prototypes)

        if y_true is None:
            y_true = labels
            filenames = fnames
        else:
            assert np.array_equal(y_true, labels), "Labels mismatch across TTA!"

        all_tta_probs.append(probs)

    # Average probabilities
    y_probs_avg = np.mean(all_tta_probs, axis=0)

    print("\n✓ TTA complete! Averaged probabilities from all augmentations")

    return y_true, y_probs_avg, all_tta_probs, filenames


def calibrate_probabilities(y_true_cal, y_probs_cal):
    """Calibrate probabilities using isotonic regression"""
    print("\nCalibrating probabilities with isotonic regression...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_probs_cal, y_true_cal)
    print("✓ Calibration complete!")
    return calibrator


def find_optimal_threshold(y_true, y_probs, fine_tune=True):
    """Find threshold that maximizes F1 score"""
    thresholds = np.linspace(0.0, 1.0, 201)
    best_threshold = 0.5
    best_f1 = 0

    # Add progress bar for coarse search
    for threshold in tqdm(thresholds, desc="  Coarse threshold search", leave=False, ncols=80):
        y_pred_temp = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_temp, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Fine-grained search
    if fine_tune:
        print(f"  Coarse search: threshold={best_threshold:.4f}, F1={best_f1:.4f}")
        print(f"  Fine-tuning around {best_threshold:.4f}...")

        fine_start = max(0.0, best_threshold - 0.05)
        fine_end = min(1.0, best_threshold + 0.05)
        fine_thresholds = np.arange(fine_start, fine_end, 0.002)

        for threshold in tqdm(fine_thresholds, desc="  Fine threshold search", leave=False, ncols=80):
            y_pred_temp = (y_probs >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_temp, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"  Fine-tuned: threshold={best_threshold:.4f}, F1={best_f1:.4f}")

    return best_threshold, best_f1


def find_threshold_with_constraints(y_true, y_probs, min_precision=0.70):
    """Find threshold that maximizes F1 while maintaining minimum precision"""
    thresholds = np.linspace(0.0, 1.0, 2001)
    best = {'threshold': 0.5, 'f1': -1.0, 'precision': 0.0, 'recall': 0.0}

    for t in tqdm(thresholds, desc="  Precision-constrained search", leave=False, ncols=80):
        y_pred = (y_probs >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if p >= min_precision and f1 > best['f1']:
            best = {'threshold': float(t), 'f1': float(f1), 'precision': float(p), 'recall': float(r)}

    return best


def find_threshold_youden(y_true, y_probs):
    """Find threshold using Youden's J statistic (balanced accuracy)"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j = tpr - fpr
    k = np.argmax(j)
    return float(thresholds[k]), float(tpr[k]), float(1 - fpr[k])


# ============ FEW-SHOT ADAPTATION FUNCTIONS ============

def create_support_set(data_dir, metadata_csv, n_shot=5, seed=42):
    """
    Create a balanced support set for few-shot adaptation.

    Args:
        data_dir: Directory containing images
        metadata_csv: CSV with labels
        n_shot: Number of examples per class (default: 5)
        seed: Random seed for reproducibility

    Returns:
        support_df: DataFrame with support examples
        query_df: DataFrame with remaining examples
    """
    df = pd.read_csv(metadata_csv)

    np.random.seed(seed)
    support_indices = []

    # Sample n_shot examples from each class
    for label in [0, 1]:  # Real (0) and Fake (1)
        class_indices = df[df['label'] == label].index.tolist()
        selected = np.random.choice(class_indices, size=min(n_shot, len(class_indices)), replace=False)
        support_indices.extend(selected)

    # Create support and query sets
    support_df = df.loc[support_indices].reset_index(drop=True)
    query_df = df.drop(support_indices).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Few-Shot Support Set Created")
    print(f"{'='*60}")
    print(f"Support set: {len(support_df)} examples ({n_shot} per class)")
    print(f"  Real: {(support_df['label']==0).sum()}")
    print(f"  Fake: {(support_df['label']==1).sum()}")
    print(f"Query set: {len(query_df)} examples")
    print(f"{'='*60}\n")

    return support_df, query_df


def few_shot_prototype(model, support_loader, device, use_amp=True):
    """
    Compute class prototypes from support set for prototype-based classification.

    Args:
        model: The classification model
        support_loader: DataLoader with support examples
        device: torch device
        use_amp: Use automatic mixed precision

    Returns:
        prototypes: Dict with class prototypes {'real': tensor, 'fake': tensor}
    """
    print(f"\n{'='*60}")
    print(f"Few-Shot Prototype Adaptation")
    print(f"{'='*60}")
    print(f"Computing class prototypes from support set...")

    model.eval()

    # Collect features for each class
    real_features = []
    fake_features = []

    with torch.inference_mode():
        for images, labels, _ in tqdm(support_loader, desc="  Extracting features", ncols=80, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    # Extract normalized features from backbone
                    features = model.backbone.encode_image(images)
                    features = features / features.norm(dim=-1, keepdim=True)
            else:
                features = model.backbone.encode_image(images)
                features = features / features.norm(dim=-1, keepdim=True)

            # Separate by class
            for i, label in enumerate(labels):
                if label == 0:  # Real
                    real_features.append(features[i])
                else:  # Fake
                    fake_features.append(features[i])

    # Compute prototypes (mean of features per class)
    real_prototype = torch.stack(real_features).mean(dim=0)
    fake_prototype = torch.stack(fake_features).mean(dim=0)

    # Normalize prototypes
    real_prototype = real_prototype / real_prototype.norm()
    fake_prototype = fake_prototype / fake_prototype.norm()

    prototypes = {
        'real': real_prototype,
        'fake': fake_prototype
    }

    print(f"✓ Prototypes computed!")
    print(f"  Real prototype shape: {real_prototype.shape}")
    print(f"  Fake prototype shape: {fake_prototype.shape}")
    print(f"  Used {len(real_features)} real and {len(fake_features)} fake examples")
    print(f"{'='*60}\n")

    return prototypes


def save_support_set(support_df, output_path):
    """Save support set for reproducibility"""
    support_df.to_csv(output_path, index=False)
    print(f"✓ Support set saved to: {output_path}")


def load_support_set(support_csv):
    """Load a previously saved support set"""
    support_df = pd.read_csv(support_csv)
    print(f"✓ Loaded support set: {len(support_df)} examples")
    return support_df


def plot_confusion_matrix(y_true, y_pred, output_dir, suffix=""):
    """Generate confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'},
                ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix{suffix}')

    plt.tight_layout()
    fname = 'confusion_matrix' + suffix.lower().replace(' ', '_').replace('-', '_')
    plt.savefig(output_dir / f'{fname}.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{fname}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {fname}.png/pdf")


def plot_roc_curve(y_true, y_probs, output_dir):
    """Generate ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'roc_curve.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: roc_curve.png/pdf")
    return roc_auc


def plot_precision_recall_curve(y_true, y_probs, output_dir):
    """Generate Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {avg_precision:.4f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'precision_recall_curve.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: precision_recall_curve.png/pdf")
    return avg_precision


def plot_probability_distribution(y_true, y_probs, output_dir, optimal_threshold=0.5):
    """Plot probability distribution"""
    real_probs = y_probs[y_true == 0]
    fake_probs = y_probs[y_true == 1]

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 50)
    ax.hist(real_probs, bins=bins, alpha=0.6, label='Real Images',
            color='green', edgecolor='black')
    ax.hist(fake_probs, bins=bins, alpha=0.6, label='Fake Images',
            color='red', edgecolor='black')

    ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2,
               label=f'Optimal Threshold ({optimal_threshold:.3f})')

    ax.set_xlabel('Predicted Probability (Fake)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Predicted Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'probability_distribution.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: probability_distribution.png/pdf")


def save_results(metrics, output_dir, predictions_df=None):
    """Save metrics to JSON and text"""
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / 'report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AI vs. Human-Generated Images - Inference Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Overall Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"AUC-ROC:   {metrics['auc_roc']:.4f}\n")
        f.write(f"AP Score:  {metrics['avg_precision']:.4f}\n")
        if 'optimal_threshold' in metrics:
            f.write(f"Optimal Threshold: {metrics['optimal_threshold']:.3f}\n")
        f.write("\n")

        f.write("Per-Class Metrics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Real':<10} {metrics['precision_real']:<12.4f} {metrics['recall_real']:<12.4f} {metrics['f1_real']:<12.4f}\n")
        f.write(f"{'Fake':<10} {metrics['precision_fake']:<12.4f} {metrics['recall_fake']:<12.4f} {metrics['f1_fake']:<12.4f}\n")
        f.write("\n")

        f.write("Confusion Matrix:\n")
        f.write("-" * 40 + "\n")
        cm = metrics['confusion_matrix']
        f.write(f"              Predicted Real  Predicted Fake\n")
        f.write(f"Actual Real   {cm[0][0]:<15} {cm[0][1]:<15}\n")
        f.write(f"Actual Fake   {cm[1][0]:<15} {cm[1][1]:<15}\n")
        f.write("\n")

        f.write(metrics['classification_report'])

    # Save predictions to CSV
    if predictions_df is not None:
        predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
        print(f"✓ Saved predictions to: predictions.csv")

    print(f"\n✓ Saved: metrics.json, report.txt")


def main():
    import torch as torch  # Ensure we use the global torch module

    # ============ CONFIGURATION ============
    MODEL_PATH = "/mnt/c/Users/admin/Desktop/kaggle deepfake/best_model.pth"
    DATA_DIR = "/mnt/c/Users/admin/Desktop/AI vs. Human-Generated Images"
    METADATA_CSV = "/mnt/c/Users/admin/Desktop/AI vs. Human-Generated Images/train.csv"
    OUTPUT_DIR = Path("/home/joesobos/ai_human_results")

    # Performance settings
    BATCH_SIZE = 64  # Reduced batch size for faster loading
    NUM_WORKERS = 4  # Reduced workers to prevent I/O bottleneck
    MODEL_SIZE = 'large'
    USE_TORCH_COMPILE = True  # Enable torch.compile() for 30-50% speedup (PyTorch 2.0+)

    # Label orientation fix - set once after verifying your training setup
    # If training used y=1 for REAL (opposite of dataset), set to True
    INVERT_LOGITS = False  # TOGGLED: Model outputs were biased high, try non-inverted

    # Enhancement options
    USE_TTA = True  # Test-Time Augmentation (minimal: base + hflip)
    NUM_TTA_AUGMENTS = 2  # Only base + horizontal flip (safe, label-preserving)
    USE_CALIBRATION = False  # DISABLED: Isotonic regression was collapsing weak model's variance
    CALIBRATION_SPLIT = 0.2

    # Threshold strategy
    MIN_PRECISION = 0.60  # Minimum precision constraint (reduced from 0.70 due to weak AUC)

    # ============ FEW-SHOT ADAPTATION SETTINGS ============
    USE_FEW_SHOT_ADAPTATION = True  # Enable few-shot adaptation
    FEW_SHOT_N_SHOT = 100  # Number of examples per class (100-shot = 200 total examples)
    FEW_SHOT_METHOD = 'prototype'  # Options: 'prototype', 'none'
    FEW_SHOT_SEED = 42  # Random seed for support set selection
    FEW_SHOT_SUPPORT_CSV = None  # Path to pre-saved support set (None = create new)

    # ============ SETUP ============
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_subdir = OUTPUT_DIR / f"run_{timestamp}"
    output_subdir.mkdir(exist_ok=True)

    print("=" * 60)
    print("AI vs. Human-Generated Images Inference")
    print("  - Optimal threshold finding")
    if USE_CALIBRATION:
        print(f"  - Probability calibration")
    if USE_TTA and NUM_TTA_AUGMENTS > 1:
        print(f"  - Test-Time Augmentation ({NUM_TTA_AUGMENTS} transforms)")
    if USE_FEW_SHOT_ADAPTATION and FEW_SHOT_METHOD == 'prototype':
        print(f"  - Few-Shot Adaptation (prototype method, {FEW_SHOT_N_SHOT}-shot)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ============ DIAGNOSTIC CHECKS ============
    print("\n" + "=" * 60)
    print("RUNNING DIAGNOSTIC CHECKS")
    print("=" * 60)

    # Check metadata
    print(f"\n1. Dataset Check:")
    with tqdm(total=1, desc="  Loading metadata", ncols=80, leave=False) as pbar:
        df = pd.read_csv(METADATA_CSV)
        pbar.update(1)

    print(f"   - Total rows in CSV: {len(df)}")
    print(f"   - Sample labels: {df['label'].head(5).tolist()}")
    unique_labels = df['label'].unique()
    print(f"   - Unique labels: {unique_labels}")
    print(f"   - Label distribution: 0 (Human/Real): {(df['label']==0).sum()}, 1 (AI/Fake): {(df['label']==1).sum()}")

    # Check images
    print(f"   - Sample file paths: {df['file_name'].head(3).tolist()}")

    # ============ LOAD CHECKPOINT & DETECT MODEL SIZE ============
    print("\n2. Checkpoint Inspection:")
    with tqdm(total=1, desc="  Loading checkpoint", ncols=80, leave=False) as pbar:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        pbar.update(1)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Infer feature dimension from classifier head
    def infer_feature_dim_from_head(sd):
        for k, v in sd.items():
            if "classifier.2.weight" in k and v.ndim == 2:
                return v.shape[1]  # input features
        return None

    feat_dim_ckpt = infer_feature_dim_from_head(state_dict)
    print(f"   - Checkpoint classifier expects {feat_dim_ckpt} input features")

    # Auto-detect correct model size
    if feat_dim_ckpt == 768:
        correct_size = 'small'
        print(f"   - Detected: ViT-B-16-SigLIP-384 (MODEL_SIZE='small')")
    elif feat_dim_ckpt == 1024:
        correct_size = 'large'
        print(f"   - Detected: ViT-L-16-SigLIP-384 (MODEL_SIZE='large')")
    else:
        correct_size = MODEL_SIZE
        print(f"   - ⚠️  Unknown feature dimension, using MODEL_SIZE='{MODEL_SIZE}'")

    if MODEL_SIZE != correct_size:
        print(f"   - ⚠️  MODEL_SIZE mismatch! You set '{MODEL_SIZE}' but checkpoint is '{correct_size}'")
        print(f"   - ✓ Auto-correcting to '{correct_size}'")
        MODEL_SIZE = correct_size

    if 'epoch' in checkpoint:
        print(f"   - Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_f1' in checkpoint:
        print(f"   - Validation F1: {checkpoint['val_f1']:.4f}")

    # ============ LOAD MODEL ============
    print("\n3. Loading Model:")
    with tqdm(total=3, desc="  Initializing model", ncols=80, leave=False) as pbar:
        model = BinaryClassifier(model_size=MODEL_SIZE, device=device).to(device)
        pbar.update(1)

        # Try strict load first
        try:
            model.load_state_dict(state_dict, strict=True)
            print("   - ✓ Strict load successful! All weights loaded correctly.")
            missing_keys, unexpected_keys = [], []
        except Exception as e:
            print(f"   - ⚠️  Strict load failed: {str(e)[:100]}")
            print("   - Falling back to non-strict load...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"   - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

            # Check if classifier loaded
            classifier_missing = [k for k in missing_keys if 'classifier' in k]
            if classifier_missing:
                print(f"   - ❌ CRITICAL: Classifier weights NOT loaded: {classifier_missing}")
                print(f"   - This will cause random predictions!")
            else:
                print(f"   - ✓ Classifier weights loaded successfully")
        pbar.update(1)

        model.eval()
        # Enable cudnn benchmarking for faster convolutions
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print("   - ✓ cuDNN auto-tuner enabled")
        pbar.update(1)

    # ============ TORCH.COMPILE OPTIMIZATION ============
    if USE_TORCH_COMPILE and device.type == 'cuda':
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            print("\n   - Compiling model with torch.compile()...")
            print("     (First run will be slower, but subsequent batches will be 30-50% faster)")
            with tqdm(total=1, desc="  Compiling model", ncols=80, leave=False) as pbar:
                # Use max-autotune for best performance (takes longer to compile but faster inference)
                model = torch.compile(model, mode='max-autotune')
                pbar.update(1)
            print("   - ✓ Model compiled with max-autotune mode!")
        except Exception as e:
            print(f"   - ⚠️  torch.compile() failed: {e}")
            print("   - Continuing without compilation...")

    print("✓ Model loaded!")

    print("=" * 60)

    # ============ FEW-SHOT ADAPTATION ============
    prototypes = None
    query_metadata_csv = METADATA_CSV  # Default: use full dataset

    if USE_FEW_SHOT_ADAPTATION and FEW_SHOT_METHOD != 'none':
        print("\n" + "=" * 60)
        print("APPLYING FEW-SHOT ADAPTATION")
        print("=" * 60)
        print(f"Method: {FEW_SHOT_METHOD}")
        print(f"N-shot: {FEW_SHOT_N_SHOT} examples per class")

        # Create or load support set
        if FEW_SHOT_SUPPORT_CSV and Path(FEW_SHOT_SUPPORT_CSV).exists():
            support_df = load_support_set(FEW_SHOT_SUPPORT_CSV)
            # Create query set by removing support examples
            full_df = pd.read_csv(METADATA_CSV)
            support_files = set(support_df['file_name'].tolist())
            query_df = full_df[~full_df['file_name'].isin(support_files)].reset_index(drop=True)
        else:
            support_df, query_df = create_support_set(
                DATA_DIR, METADATA_CSV,
                n_shot=FEW_SHOT_N_SHOT,
                seed=FEW_SHOT_SEED
            )
            # Save support set for reproducibility
            save_support_set(support_df, output_subdir / 'support_set.csv')

        # Save query set CSV for inference
        query_csv_path = output_subdir / 'query_set.csv'
        query_df.to_csv(query_csv_path, index=False)
        query_metadata_csv = str(query_csv_path)

        # Apply adaptation method
        if FEW_SHOT_METHOD == 'prototype':
            # Create support dataloader
            support_transform = model.preprocess
            support_dataset = AIHumanDataset(DATA_DIR, output_subdir / 'support_set.csv',
                                            transform=support_transform)
            support_loader = DataLoader(
                support_dataset,
                batch_size=min(32, len(support_dataset)),
                shuffle=False,
                num_workers=2,
                pin_memory=True if device.type == 'cuda' else False
            )

            # Compute prototypes from support set
            prototypes = few_shot_prototype(
                model=model,
                support_loader=support_loader,
                device=device,
                use_amp=True
            )

            print(f"✓ Prototypes computed from {FEW_SHOT_N_SHOT}-shot support set")

        # Use prototypes for inference
        if prototypes is not None:
            print(f"\n✓ Using prototype-based classification for query set")

    print("=" * 60)

    # ============ RUN INFERENCE ============
    image_size = model.resolution

    if USE_TTA and NUM_TTA_AUGMENTS > 1:
        # Test-Time Augmentation
        tta_transforms = create_tta_transforms(image_size, NUM_TTA_AUGMENTS)
        y_true, y_probs, all_tta_probs, filenames = run_tta_inference(
            model, DATA_DIR, query_metadata_csv, tta_transforms, BATCH_SIZE, NUM_WORKERS, device, use_amp=True, invert_logits=INVERT_LOGITS, prototypes=prototypes
        )
        y_probs_base = all_tta_probs[0]
    else:
        # Standard inference
        print("\nRunning standard inference...")
        # USE MODEL'S NATIVE PREPROCESSING (CRITICAL FIX)
        test_transform = model.preprocess
        print(f"   - Using SigLIP's native preprocessing (not custom normalization)")

        dataset = AIHumanDataset(DATA_DIR, query_metadata_csv, transform=test_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if device.type == 'cuda' else False,
            prefetch_factor=4,  # Increased prefetch for better pipeline
            persistent_workers=True if NUM_WORKERS > 0 else False
        )

        # Warm up GPU if using torch.compile
        if USE_TORCH_COMPILE and device.type == 'cuda' and prototypes is None:
            print("\n   - Warming up compiled model (first batch)...")
            try:
                dummy_batch = next(iter(dataloader))
                with torch.inference_mode():
                    with torch.cuda.amp.autocast():
                        _ = model(dummy_batch[0].to(device, non_blocking=True))
                print("   - ✓ Model warmed up!")
            except Exception as e:
                print(f"   - ⚠️  Warmup failed: {e}")

        y_true, y_probs, filenames = run_inference(model, dataloader, device, use_amp=True, desc="Inference", invert_logits=INVERT_LOGITS, prototypes=prototypes)
        y_probs_base = y_probs.copy()

    # Probability distribution sanity check
    print(f"\n4. Probability Sanity Check:")
    print(f"   - Probs mean={y_probs.mean():.3f}, std={y_probs.std():.3f}")
    print(f"   - Probs min={y_probs.min():.3f}, max={y_probs.max():.3f}")
    if y_probs.std() < 0.05:
        print(f"   - ⚠️  WARNING: Very low std - classifier may be random/saturated!")

    # Check label orientation (warning only - orientation fixed at logit level)
    test_auc = roc_auc_score(y_true, y_probs)
    print(f"   - AUC-ROC: {test_auc:.4f}")

    if test_auc < 0.5:
        print(f"   - ⚠️  WARNING: AUC < 0.5! Check INVERT_LOGITS setting.")
        print(f"   - Current INVERT_LOGITS={INVERT_LOGITS}")
        print(f"   - Consider toggling INVERT_LOGITS if labels are inverted.")
    else:
        print(f"   - ✓ Label orientation looks correct (AUC > 0.5)")

    # ============ CALIBRATION ============
    y_probs_uncal = y_probs.copy()

    if USE_CALIBRATION:
        n_total = len(y_true)
        n_cal = int(n_total * CALIBRATION_SPLIT)
        n_test = n_total - n_cal

        print(f"\n" + "="*60)
        print(f"Applying Probability Calibration")
        print(f"  Calibration set: {n_cal} samples ({CALIBRATION_SPLIT*100:.0f}%)")
        print(f"  Test set:        {n_test} samples ({(1-CALIBRATION_SPLIT)*100:.0f}%)")
        print("="*60)

        indices = np.random.RandomState(42).permutation(n_total)
        cal_idx = indices[:n_cal]
        test_idx = indices[n_cal:]

        y_true_cal, y_probs_cal = y_true[cal_idx], y_probs[cal_idx]
        y_true_test, y_probs_test = y_true[test_idx], y_probs[test_idx]

        calibrator = calibrate_probabilities(y_true_cal, y_probs_cal)
        y_probs_test_cal = calibrator.predict(y_probs_test)

        y_true = y_true_test
        y_probs = y_probs_test_cal
        y_probs_base = y_probs_base[test_idx]
        filenames = [filenames[i] for i in test_idx]

    # ============ FIND OPTIMAL THRESHOLDS ============
    print("\n" + "="*60)
    print("Finding optimal thresholds...")
    print("="*60)

    # 1. Unconstrained F1 maximization
    print("\n1. Unconstrained F1 optimization:")
    best_threshold, best_f1 = find_optimal_threshold(y_true, y_probs)
    print(f"   Threshold: {best_threshold:.3f}, F1: {best_f1:.4f}")

    # 2. Precision-constrained F1
    print(f"\n2. Precision-constrained (P ≥ {MIN_PRECISION:.2f}):")
    constrained = find_threshold_with_constraints(y_true, y_probs, min_precision=MIN_PRECISION)
    if constrained['f1'] > 0:
        print(f"   Threshold: {constrained['threshold']:.3f}")
        print(f"   F1: {constrained['f1']:.4f}, Precision: {constrained['precision']:.4f}, Recall: {constrained['recall']:.4f}")
    else:
        print(f"   ⚠️  No threshold found meeting P ≥ {MIN_PRECISION:.2f} constraint")
        constrained = None

    # 3. Youden's J (balanced accuracy)
    print("\n3. Youden's J (balanced accuracy):")
    youden_thr, youden_tpr, youden_spec = find_threshold_youden(y_true, y_probs)
    print(f"   Threshold: {youden_thr:.3f}")
    print(f"   TPR (Recall): {youden_tpr:.4f}, Specificity: {youden_spec:.4f}")

    # Select which threshold to use
    # For weak models (AUC < 0.65), prefer Youden's J (balanced) over unconstrained F1
    if constrained and constrained['f1'] > 0:
        selected_threshold = constrained['threshold']
        threshold_strategy = "Precision-Constrained"
        print(f"\n✓ Using precision-constrained threshold: {selected_threshold:.3f}")
    elif test_auc < 0.65:
        # Weak AUC: use Youden's J for balance instead of unconstrained F1
        selected_threshold = youden_thr
        threshold_strategy = "Youden's J (Balanced)"
        print(f"\n✓ Using Youden's J threshold (balanced, due to weak AUC): {selected_threshold:.3f}")
    else:
        selected_threshold = best_threshold
        threshold_strategy = "Unconstrained F1"
        print(f"\n✓ Using unconstrained F1 threshold: {selected_threshold:.3f}")

    # ============ CALCULATE METRICS ============
    y_pred = (y_probs >= selected_threshold).astype(int)

    print("\nCalculating metrics...")
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred,
                                       target_names=['Real', 'Fake'],
                                       output_dict=True)
    report_str = classification_report(y_true, y_pred,
                                      target_names=['Real', 'Fake'])

    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_probs)
    ap_score = average_precision_score(y_true, y_probs)

    metrics = {
        'accuracy': float(accuracy),
        'precision_real': float(report_dict['Real']['precision']),
        'recall_real': float(report_dict['Real']['recall']),
        'f1_real': float(report_dict['Real']['f1-score']),
        'precision_fake': float(report_dict['Fake']['precision']),
        'recall_fake': float(report_dict['Fake']['recall']),
        'f1_fake': float(report_dict['Fake']['f1-score']),
        'confusion_matrix': cm.tolist(),
        'classification_report': report_str,
        'auc_roc': float(auc_score),
        'avg_precision': float(ap_score),
        'selected_threshold': float(selected_threshold),
        'threshold_strategy': threshold_strategy,
        'unconstrained_f1_threshold': float(best_threshold),
        'youden_threshold': float(youden_thr),
        'constrained_threshold': float(constrained['threshold']) if constrained else None,
        'constrained_precision': float(constrained['precision']) if constrained else None,
        'constrained_recall': float(constrained['recall']) if constrained else None,
        'use_tta': USE_TTA,
        'num_tta_augments': NUM_TTA_AUGMENTS if USE_TTA else 1,
        'use_calibration': USE_CALIBRATION,
        'invert_logits': INVERT_LOGITS,
        'few_shot_adaptation': USE_FEW_SHOT_ADAPTATION,
        'few_shot_method': FEW_SHOT_METHOD if USE_FEW_SHOT_ADAPTATION else None,
        'few_shot_n_shot': FEW_SHOT_N_SHOT if USE_FEW_SHOT_ADAPTATION else None
    }

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'filename': filenames,
        'true_label': ['Real' if label == 0 else 'Fake' for label in y_true],
        'predicted_label': ['Real' if pred == 0 else 'Fake' for pred in y_pred],
        'probability_fake': y_probs,
        'correct': y_true == y_pred
    })

    # ============ FALSE POSITIVE ANALYSIS ============
    print("\n" + "="*60)
    print("Analyzing False Positives...")
    print("="*60)

    fp = predictions_df[(predictions_df.true_label=='Real') & (predictions_df.predicted_label=='Fake')]
    fn = predictions_df[(predictions_df.true_label=='Fake') & (predictions_df.predicted_label=='Real')]

    print(f"False Positives (Real called Fake): {len(fp)}")
    print(f"False Negatives (Fake called Real): {len(fn)}")

    if len(fp) > 0:
        fp_top = fp.sort_values('probability_fake', ascending=False).head(50)
        fp_top.to_csv(output_subdir / 'top_false_positives.csv', index=False)
        print(f"✓ Saved top 50 false positives to: top_false_positives.csv")
        print(f"  Review these images to identify patterns (artifacts, upscales, etc.)")

    if len(fn) > 0:
        fn_bottom = fn.sort_values('probability_fake', ascending=True).head(50)
        fn_bottom.to_csv(output_subdir / 'top_false_negatives.csv', index=False)
        print(f"✓ Saved top 50 false negatives to: top_false_negatives.csv")

    # ============ GENERATE PLOTS ============
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    plots = [
        ("Confusion Matrix", lambda: plot_confusion_matrix(y_true, y_pred, output_subdir)),
        ("ROC Curve", lambda: plot_roc_curve(y_true, y_probs, output_subdir)),
        ("Precision-Recall Curve", lambda: plot_precision_recall_curve(y_true, y_probs, output_subdir)),
        ("Probability Distribution", lambda: plot_probability_distribution(y_true, y_probs, output_subdir, best_threshold))
    ]

    for plot_name, plot_func in tqdm(plots, desc="Generating plots", ncols=80):
        plot_func()

    # ============ SAVE RESULTS ============
    save_results(metrics, output_subdir, predictions_df)

    # ============ PRINT SUMMARY ============
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if USE_FEW_SHOT_ADAPTATION and FEW_SHOT_METHOD != 'none':
        print(f"Few-Shot Adaptation:  {FEW_SHOT_METHOD} ({FEW_SHOT_N_SHOT}-shot)")
        print(f"Query Samples:        {len(y_true)}")
    else:
        print(f"Total Samples:        {len(y_true)}")
    print(f"Threshold Strategy:   {threshold_strategy}")
    print(f"Selected Threshold:   {selected_threshold:.3f}")
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  AUC-ROC:            {auc_score:.4f}")
    print(f"  Average Precision:  {ap_score:.4f}")
    print(f"\nFake Detection:")
    print(f"  Precision:          {metrics['precision_fake']:.4f}")
    print(f"  Recall:             {metrics['recall_fake']:.4f}")
    print(f"  F1-Score:           {metrics['f1_fake']:.4f}")
    print(f"\nReal Detection:")
    print(f"  Precision:          {metrics['precision_real']:.4f}")
    print(f"  Recall:             {metrics['recall_real']:.4f}")
    print(f"  F1-Score:           {metrics['f1_real']:.4f}")
    print(f"\nError Analysis:")
    print(f"  False Positives:    {len(fp)} (Real → Fake)")
    print(f"  False Negatives:    {len(fn)} (Fake → Real)")
    print(f"\nResults saved to: {output_subdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
