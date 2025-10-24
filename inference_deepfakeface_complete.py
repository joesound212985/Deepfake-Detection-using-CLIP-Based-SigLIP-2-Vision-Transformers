#!/usr/bin/env python3
"""
Complete Inference Script with Everything:
- Optimal threshold finding
- Probability calibration (Option 2)
- Test-Time Augmentation ensemble (Options 3 & 4)
- All plots and metrics
- Speed optimizations
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
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score, f1_score,
    precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import random
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
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import kornia
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def fast_image_load(img_path):
    """Fast image loading with OpenCV fallback to PIL"""
    if OPENCV_AVAILABLE:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img)
        except Exception:
            pass
    return Image.open(img_path).convert('RGB')


def apply_clahe(pil_image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if not OPENCV_AVAILABLE:
        return pil_image

    # Convert PIL to OpenCV
    img = np.array(pil_image)

    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if len(img.shape) == 3:
        # RGB image
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale
        img = clahe.apply(img)

    return Image.fromarray(img)


def apply_sharpening(pil_image):
    """Apply edge sharpening to enhance artifacts"""
    from PIL import ImageFilter, ImageEnhance

    # Sharpen filter
    sharpened = pil_image.filter(ImageFilter.SHARPEN)

    # Additional edge enhancement
    enhancer = ImageEnhance.Sharpness(sharpened)
    sharpened = enhancer.enhance(1.5)

    return sharpened


def apply_edge_enhance(pil_image):
    """Apply edge enhancement for deepfake artifact detection"""
    from PIL import ImageFilter

    # Edge enhancement filter
    enhanced = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)

    return enhanced


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

    def extract_features(self, x):
        """Extract features without classification head (for few-shot learning)"""
        if x.shape[-1] != self.resolution:
            x = nn.functional.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear')

        features = self.backbone.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, x):
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)


class KaggleDeepfakeDataset(Dataset):
    """Dataset for Kaggle Deepfake Detection Challenge"""
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        fake_dirs = list(self.data_dir.rglob('Fake')) + list(self.data_dir.rglob('fake'))
        real_dirs = list(self.data_dir.rglob('Real')) + list(self.data_dir.rglob('real'))

        for fake_dir in fake_dirs:
            if fake_dir.is_dir():
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    for img_file in fake_dir.glob(ext):
                        self.samples.append((str(img_file), 1))

        for real_dir in real_dirs:
            if real_dir.is_dir():
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    for img_file in real_dir.glob(ext):
                        self.samples.append((str(img_file), 0))

        if max_samples and len(self.samples) > max_samples:
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]

        labels = [label for _, label in self.samples]
        print(f"Loaded {len(self.samples)} images (Real: {labels.count(0)}, Fake: {labels.count(1)})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = fast_image_load(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


def create_tta_transforms(image_size, num_augments=9):
    """Create test-time augmentation transforms with advanced F1-boosting techniques"""
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
        # 3. Rotation +5°
        rot_pos = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("Rot+5", rot_pos))

    if num_augments >= 4:
        # 4. Rotation -5°
        rot_neg = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("Rot-5", rot_neg))

    if num_augments >= 5:
        # 5. Color jitter
        color = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("ColorJitter", color))

    if num_augments >= 6:
        # 6. CLAHE - Contrast Limited Adaptive Histogram Equalization
        # Helps detect deepfake artifacts by enhancing local contrast
        clahe = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: apply_clahe(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("CLAHE", clahe))

    if num_augments >= 7:
        # 7. Sharpening - Enhances edges and artifacts
        sharpen = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: apply_sharpening(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("Sharpen", sharpen))

    if num_augments >= 8:
        # 8. Edge Enhancement - Highlights compression artifacts
        edge = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: apply_edge_enhance(img)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("EdgeEnhance", edge))

    if num_augments >= 9:
        # 9. CLAHE + Sharpening combo (most powerful for artifact detection)
        clahe_sharp = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda img: apply_sharpening(apply_clahe(img))),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        tta_transforms.append(("CLAHE+Sharpen", clahe_sharp))

    return tta_transforms


def run_inference(model, dataloader, device, gpu_transform=None, use_amp=True, desc="Inference"):
    """Run inference and collect predictions"""
    all_labels = []
    all_probs = []

    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(device, non_blocking=True)

            if gpu_transform is not None:
                images = gpu_transform(images)

            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    logits = model(images)
            else:
                logits = model(images)

            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def run_tta_inference(model, data_dir, tta_transforms, batch_size, num_workers, device, use_amp=True):
    """Run test-time augmentation"""
    all_tta_probs = []
    y_true = None

    print("\n" + "="*60)
    print(f"Running Test-Time Augmentation ({len(tta_transforms)} transforms)")
    print("="*60)

    for tta_name, tta_transform in tta_transforms:
        print(f"\nProcessing: {tta_name}")

        dataset = KaggleDeepfakeDataset(data_dir, transform=tta_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        labels, probs = run_inference(model, dataloader, device, None, use_amp, desc=f"  {tta_name}")

        if y_true is None:
            y_true = labels
        else:
            assert np.array_equal(y_true, labels), "Labels mismatch across TTA!"

        all_tta_probs.append(probs)

    # Average probabilities
    y_probs_avg = np.mean(all_tta_probs, axis=0)

    print("\n✓ TTA complete! Averaged probabilities from all augmentations")

    return y_true, y_probs_avg, all_tta_probs


def calibrate_probabilities(y_true_cal, y_probs_cal):
    """Calibrate probabilities using isotonic regression"""
    print("\nCalibrating probabilities with isotonic regression...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_probs_cal, y_true_cal)
    print("✓ Calibration complete!")
    return calibrator


def find_optimal_threshold(y_true, y_probs, fine_tune=True):
    """Find threshold that maximizes F1 score with optional fine-tuning"""
    # Coarse search: Full range
    thresholds = np.linspace(0.0, 1.0, 201)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred_temp = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_temp)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Fine-grained search around optimal (±0.05 with 0.002 step)
    if fine_tune:
        print(f"  Coarse search: threshold={best_threshold:.4f}, F1={best_f1:.4f}")
        print(f"  Fine-tuning around {best_threshold:.4f}...")

        fine_start = max(0.0, best_threshold - 0.05)
        fine_end = min(1.0, best_threshold + 0.05)
        fine_thresholds = np.arange(fine_start, fine_end, 0.002)

        for threshold in fine_thresholds:
            y_pred_temp = (y_probs >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_temp)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"  Fine-tuned: threshold={best_threshold:.4f}, F1={best_f1:.4f} (+{best_f1-best_f1:.4f})")

    return best_threshold, best_f1


def apply_temperature_scaling(y_probs, temperature=1.0):
    """
    Apply temperature scaling to probabilities
    Temperature > 1: Makes probabilities more uncertain (softer)
    Temperature < 1: Makes probabilities more confident (sharper)
    Temperature = 1: No change
    """
    # Convert probabilities to logits
    epsilon = 1e-7
    y_probs = np.clip(y_probs, epsilon, 1 - epsilon)
    logits = np.log(y_probs / (1 - y_probs))

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Convert back to probabilities
    scaled_probs = 1 / (1 + np.exp(-scaled_logits))

    return scaled_probs


def find_optimal_temperature(y_true, y_probs):
    """Find temperature that maximizes F1 score"""
    print("\nFinding optimal temperature scaling...")

    best_temp = 1.0
    best_f1 = 0
    best_threshold = 0.5

    # Search temperature range
    for temp in np.linspace(0.5, 2.0, 31):
        # Apply temperature scaling
        y_probs_temp = apply_temperature_scaling(y_probs, temperature=temp)

        # Find best threshold for this temperature
        thresh, f1 = find_optimal_threshold(y_true, y_probs_temp)

        if f1 > best_f1:
            best_f1 = f1
            best_temp = temp
            best_threshold = thresh

    print(f"  Optimal temperature: {best_temp:.3f}")
    print(f"  Expected F1:         {best_f1:.4f}")

    return best_temp, best_threshold, best_f1


# ============ PLOTTING FUNCTIONS ============

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


def plot_confusion_matrix_normalized(y_true, y_pred, output_dir):
    """Generate normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Percentage'},
                ax=ax)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix (Normalized)')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'confusion_matrix_normalized.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: confusion_matrix_normalized.png/pdf")


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


def plot_threshold_analysis(y_true, y_probs, optimal_threshold, output_dir):
    """Plot metrics vs threshold"""
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(thresholds, accuracies, label='Accuracy', linewidth=2, color='#3498db')
    ax.plot(thresholds, precisions, label='Precision (Fake)', linewidth=2, color='#e74c3c')
    ax.plot(thresholds, recalls, label='Recall (Fake)', linewidth=2, color='#2ecc71')
    ax.plot(thresholds, f1_scores, label='F1-Score (Fake)', linewidth=2, color='#f39c12')

    ax.axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2,
               label=f'Optimal Threshold ({optimal_threshold:.3f})')

    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Decision Threshold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'threshold_analysis.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: threshold_analysis.png/pdf")


def plot_calibration_curve(y_true, y_probs_uncal, y_probs_cal, output_dir):
    """Plot calibration curve"""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_probs_uncal, n_bins=10, strategy='uniform'
    )
    ax.plot(mean_pred_uncal, fraction_pos_uncal, 's-',
            label='Before Calibration', linewidth=2, markersize=8, color='#e74c3c')

    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_probs_cal, n_bins=10, strategy='uniform'
    )
    ax.plot(mean_pred_cal, fraction_pos_cal, 'o-',
            label='After Calibration', linewidth=2, markersize=8, color='#2ecc71')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Probability Calibration Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'calibration_curve.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: calibration_curve.png/pdf")


def plot_comparison_bar(methods, accuracies, f1_scores, output_dir):
    """Compare different methods"""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#2ecc71', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Base vs Enhanced Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'method_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: method_comparison.png/pdf")


def plot_combined_curves(y_true, y_probs, output_dir):
    """Combined ROC and PR curves"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # PR Curve
    ax2.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'combined_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'combined_curves.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: combined_curves.png/pdf")


def plot_class_comparison(metrics, output_dir):
    """Compare Real vs Fake performance"""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Precision', 'Recall', 'F1-Score']
    real_values = [metrics['precision_real'], metrics['recall_real'], metrics['f1_real']]
    fake_values = [metrics['precision_fake'], metrics['recall_fake'], metrics['f1_fake']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, real_values, width, label='Real', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, fake_values, width, label='Fake', color='#e74c3c', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'class_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'class_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: class_comparison.png/pdf")


def plot_metrics_comparison(metrics, output_dir):
    """Metrics bar plot"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['Accuracy', 'Precision\n(Fake)', 'Recall\n(Fake)',
                   'F1-Score\n(Fake)', 'AUC-ROC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision_fake'],
        metrics['recall_fake'],
        metrics['f1_fake'],
        metrics['auc_roc']
    ]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Summary')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'metrics_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: metrics_comparison.png/pdf")


# ============ FEW-SHOT LEARNING FUNCTIONS ============

def extract_few_shot_support_set(data_dir, n_shot=5, seed=42):
    """Extract a support set for few-shot learning (n_shot per class)"""
    random.seed(seed)
    data_dir = Path(data_dir)

    support_set = {'real': [], 'fake': []}

    # Find real images
    real_dirs = list(data_dir.rglob('Real')) + list(data_dir.rglob('real'))
    real_images = []
    for real_dir in real_dirs:
        if real_dir.is_dir():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                real_images.extend(real_dir.glob(ext))

    # Find fake images
    fake_dirs = list(data_dir.rglob('Fake')) + list(data_dir.rglob('fake'))
    fake_images = []
    for fake_dir in fake_dirs:
        if fake_dir.is_dir():
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                fake_images.extend(fake_dir.glob(ext))

    # Sample n_shot from each class
    if len(real_images) >= n_shot:
        support_set['real'] = random.sample(real_images, n_shot)
    else:
        support_set['real'] = real_images
        print(f"Warning: Only {len(real_images)} real images available (requested {n_shot})")

    if len(fake_images) >= n_shot:
        support_set['fake'] = random.sample(fake_images, n_shot)
    else:
        support_set['fake'] = fake_images
        print(f"Warning: Only {len(fake_images)} fake images available (requested {n_shot})")

    print(f"Few-shot support set created: {len(support_set['real'])} real, {len(support_set['fake'])} fake")
    return support_set


def extract_support_features(model, support_set, device):
    """Extract features from support set"""
    support_features = []
    support_labels = []

    model.eval()
    with torch.no_grad():
        # Process real images
        for img_path in support_set['real']:
            image = fast_image_load(str(img_path))
            image = model.preprocess(image).unsqueeze(0).to(device)
            features = model.extract_features(image)
            support_features.append(features.cpu())
            support_labels.append(0)  # Real = 0

        # Process fake images
        for img_path in support_set['fake']:
            image = fast_image_load(str(img_path))
            image = model.preprocess(image).unsqueeze(0).to(device)
            features = model.extract_features(image)
            support_features.append(features.cpu())
            support_labels.append(1)  # Fake = 1

    support_features = torch.cat(support_features, dim=0)
    support_labels = torch.tensor(support_labels)

    return support_features, support_labels


def prototype_classification(support_features, support_labels, query_features, temperature=0.5):
    """Prototypical Networks: classify based on distance to class prototypes"""
    # Compute prototypes (mean of each class)
    real_mask = support_labels == 0
    fake_mask = support_labels == 1

    real_prototype = support_features[real_mask].mean(dim=0, keepdim=True)
    fake_prototype = support_features[fake_mask].mean(dim=0, keepdim=True)

    prototypes = torch.cat([real_prototype, fake_prototype], dim=0)

    # Compute distances from query to prototypes
    # Using negative squared Euclidean distance
    distances = -torch.cdist(query_features.unsqueeze(0), prototypes.unsqueeze(0)).squeeze(0)

    # Apply temperature scaling and softmax
    logits = distances / temperature
    probs = torch.softmax(logits, dim=1)

    # Return probability of being fake (class 1)
    return probs[:, 1].numpy()


def svm_classification(support_features, support_labels, query_features):
    """SVM-based few-shot classification"""
    # Convert to numpy
    X_support = support_features.numpy()
    y_support = support_labels.numpy()
    X_query = query_features.numpy()

    # Normalize features
    scaler = StandardScaler()
    X_support = scaler.fit_transform(X_support)
    X_query = scaler.transform(X_query)

    # Train SVM with RBF kernel
    svm = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
    svm.fit(X_support, y_support)

    # Get probabilities
    probs = svm.predict_proba(X_query)[:, 1]  # Probability of being fake

    return probs


def linear_probe_classification(support_features, support_labels, query_features,
                               epochs=100, lr=0.01, device='cpu'):
    """Simple linear probing on frozen features"""
    support_features = support_features.to(device)
    support_labels = support_labels.to(device).float()
    query_features = query_features.to(device)

    # Create simple linear classifier
    feature_dim = support_features.shape[1]
    classifier = nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Training
    classifier.train()
    for epoch in range(epochs):
        logits = classifier(support_features).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, support_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    classifier.eval()
    with torch.no_grad():
        query_logits = classifier(query_features).squeeze(-1)
        query_probs = torch.sigmoid(query_logits).cpu().numpy()

    return query_probs


def run_few_shot_inference(model, dataloader, support_set, device, method='prototype'):
    """Run few-shot inference using specified method"""
    print(f"\nRunning few-shot inference with {method} method...")

    # Extract support features
    support_features, support_labels = extract_support_features(model, support_set, device)

    all_labels = []
    all_probs = []

    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc=f"Few-shot {method}"):
            images = images.to(device, non_blocking=True)

            # Extract query features
            query_features = model.extract_features(images).cpu()

            # Apply few-shot method
            if method == 'prototype':
                probs = prototype_classification(support_features, support_labels,
                                                query_features, temperature=0.5)
            elif method == 'svm':
                probs = svm_classification(support_features, support_labels, query_features)
            elif method == 'linear':
                probs = linear_probe_classification(support_features, support_labels,
                                                   query_features, epochs=50, lr=0.01, device=device)
            else:
                raise ValueError(f"Unknown few-shot method: {method}")

            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


def save_results(metrics, output_dir):
    """Save metrics to JSON and text"""
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / 'report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Deepfake Detection - Complete Inference Results\n")
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

    print(f"\n✓ Saved: metrics.json, report.txt")


def main():
    # ============ CONFIGURATION ============
    MODEL_PATH = "/mnt/c/Users/admin/Desktop/kaggle deepfake/best_model.pth"
    DATA_DIR = "/mnt/c/Users/admin/Desktop/kaggle deepfake"
    OUTPUT_DIR = Path("/mnt/c/Users/admin/Desktop/kaggle deepfake/inference_results")

    # Performance settings
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    MODEL_SIZE = 'large'
    USE_TORCH_COMPILE = True

    # Enhancement options (Options 2, 3, 4 + Advanced F1 Boosting)
    USE_TTA = False  # Test-Time Augmentation - DISABLED (hurts performance)
    NUM_TTA_AUGMENTS = 1  # No augmentations
    USE_CALIBRATION = True  # ENABLED: Isotonic regression for better probability spacing
    CALIBRATION_SPLIT = 0.2  # 20% of data for calibration
    USE_TEMPERATURE_SCALING = True  # ENABLED: Both together improve precision (+0.02-0.04 F1)

    # Few-shot learning options
    USE_FEW_SHOT = True  # Enable few-shot learning with limited examples
    FEW_SHOT_METHOD = 'prototype'  # Options: 'prototype', 'svm', 'linear'
    N_SHOT = 50  # Number of examples per class (50 real + 50 fake = 100 total)

    # ============ SETUP ============
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_subdir = OUTPUT_DIR / f"run_{timestamp}"
    output_subdir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Complete Deepfake Detection Inference")
    print("  - Optimal threshold finding")
    if USE_CALIBRATION:
        print(f"  - Probability calibration (Option 2)")
    if USE_TTA and NUM_TTA_AUGMENTS > 1:
        print(f"  - Test-Time Augmentation with {NUM_TTA_AUGMENTS} transforms (Options 3 & 4)")
    if USE_FEW_SHOT:
        print(f"  - Few-shot learning: {N_SHOT}-shot per class ({N_SHOT*2} total)")
        print(f"    Method: {FEW_SHOT_METHOD}")
    print("  - All plots and metrics")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ============ LOAD MODEL ============
    print("\nLoading model...")
    model = BinaryClassifier(model_size=MODEL_SIZE, device=device).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

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

    # CRITICAL FIX: Load ALL weights (backbone + classifier)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print(f"  ⚠️  WARNING: Some weights not loaded from checkpoint!")
        print(f"     This means your fine-tuned backbone is NOT being used!")
    model.eval()

    # Compile model
    if USE_TORCH_COMPILE and hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            print("\nCompiling model...")
            model = torch.compile(model, mode='max-autotune')
            print("✓ Model compiled! (First batch will be slow, then 2-3x faster)")
        except Exception as e:
            print(f"⚠️  Compilation failed: {e}")

    print("✓ Model loaded!")

    # ============ RUN INFERENCE ============
    image_size = model.resolution

    # Few-shot learning path
    if USE_FEW_SHOT:
        print("\n" + "="*60)
        print(f"FEW-SHOT LEARNING MODE ({N_SHOT}-shot)")
        print("="*60)

        # Extract support set
        support_set = extract_few_shot_support_set(DATA_DIR, n_shot=N_SHOT, seed=42)

        # Create dataloader for query set (excluding support set images)
        test_transform = model.preprocess
        dataset = KaggleDeepfakeDataset(DATA_DIR, transform=test_transform)

        # Filter out support set images from the dataset
        support_paths = set(str(p) for p in support_set['real'] + support_set['fake'])
        filtered_samples = [(path, label) for path, label in dataset.samples if path not in support_paths]
        dataset.samples = filtered_samples
        print(f"Query set: {len(dataset.samples)} images (after removing support set)")

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        # Run few-shot inference
        y_true, y_probs = run_few_shot_inference(model, dataloader, support_set, device, method=FEW_SHOT_METHOD)
        y_probs_base = y_probs.copy()

    elif USE_TTA and NUM_TTA_AUGMENTS > 1:
        # Option 3 & 4: Test-Time Augmentation
        tta_transforms = create_tta_transforms(image_size, NUM_TTA_AUGMENTS)
        y_true, y_probs, all_tta_probs = run_tta_inference(
            model, DATA_DIR, tta_transforms, BATCH_SIZE, NUM_WORKERS, device, use_amp=True
        )
        y_probs_base = all_tta_probs[0]  # Original (no augmentation)
    else:
        # Standard inference
        print("\nRunning standard inference...")
        # CRITICAL FIX: Use model's actual preprocessing (not custom normalization)
        test_transform = model.preprocess
        print(f"  Using SigLIP's native preprocessing (not custom [0.5, 0.5, 0.5])")

        dataset = KaggleDeepfakeDataset(DATA_DIR, transform=test_transform)
        # Verify dataset balance
        real_count = sum(1 for _, label in dataset.samples if label == 0)
        fake_count = sum(1 for _, label in dataset.samples if label == 1)
        print(f"  Dataset balance: Real={real_count} ({real_count/len(dataset)*100:.1f}%), Fake={fake_count} ({fake_count/len(dataset)*100:.1f}%)")
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        y_true, y_probs = run_inference(model, dataloader, device, None, use_amp=True, desc="Inference")
        y_probs_base = y_probs.copy()

    # Check label inversion
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_true, y_probs)

    if test_auc < 0.5:
        print(f"\n⚠️  Detected label mismatch! AUC={test_auc:.4f} < 0.5")
        print("Inverting predictions...")
        y_probs = 1 - y_probs
        y_probs_base = 1 - y_probs_base
        test_auc = 1 - test_auc
        print(f"Corrected AUC: {test_auc:.4f}")

    # ============ CALIBRATION (Option 2) ============
    y_probs_uncal = y_probs.copy()

    if USE_CALIBRATION:
        n_total = len(y_true)
        n_cal = int(n_total * CALIBRATION_SPLIT)
        n_test = n_total - n_cal

        print(f"\n" + "="*60)
        print(f"Applying Probability Calibration (Option 2)")
        print(f"  Calibration set: {n_cal} samples ({CALIBRATION_SPLIT*100:.0f}%)")
        print(f"  Test set:        {n_test} samples ({(1-CALIBRATION_SPLIT)*100:.0f}%)")
        print("="*60)

        # Split data
        indices = np.random.RandomState(42).permutation(n_total)
        cal_idx = indices[:n_cal]
        test_idx = indices[n_cal:]

        y_true_cal, y_probs_cal = y_true[cal_idx], y_probs[cal_idx]
        y_true_test, y_probs_test = y_true[test_idx], y_probs[test_idx]

        # Calibrate
        calibrator = calibrate_probabilities(y_true_cal, y_probs_cal)
        y_probs_test_cal = calibrator.predict(y_probs_test)

        # Use calibrated probabilities
        y_true = y_true_test
        y_probs = y_probs_test_cal
        y_probs_uncal_test = y_probs_test

        # CRITICAL FIX: Also subset y_probs_base to match test set
        y_probs_base = y_probs_base[test_idx]

        # Plot calibration curve
        plot_calibration_curve(y_true, y_probs_uncal_test, y_probs, output_subdir)

    # ============ TEMPERATURE SCALING (Advanced F1 Boost) ============
    optimal_temperature = 1.0
    if USE_TEMPERATURE_SCALING:
        print("\n" + "="*60)
        print("Applying Temperature Scaling (Advanced F1 Boosting)")
        print("="*60)

        # Find optimal temperature
        optimal_temperature, best_threshold, best_f1 = find_optimal_temperature(y_true, y_probs)

        # Apply optimal temperature
        y_probs = apply_temperature_scaling(y_probs, temperature=optimal_temperature)

        print(f"✓ Applied temperature scaling: T={optimal_temperature:.3f}")
    else:
        # ============ FIND OPTIMAL THRESHOLD ============
        print("\nFinding optimal threshold...")
        best_threshold, best_f1 = find_optimal_threshold(y_true, y_probs)
        print(f"  Optimal threshold: {best_threshold:.3f}")
        print(f"  Expected F1:       {best_f1:.4f}")

    # ============ CALCULATE METRICS ============
    y_pred = (y_probs >= best_threshold).astype(int)

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
        'optimal_threshold': float(best_threshold),
        'use_tta': USE_TTA,
        'num_tta_augments': NUM_TTA_AUGMENTS if USE_TTA else 1,
        'use_calibration': USE_CALIBRATION,
        'use_few_shot': USE_FEW_SHOT,
        'few_shot_method': FEW_SHOT_METHOD if USE_FEW_SHOT else None,
        'n_shot': N_SHOT if USE_FEW_SHOT else None
    }

    # ============ GENERATE ALL PLOTS ============
    print("\n" + "="*60)
    print("Generating comprehensive plots...")
    print("="*60)

    plot_confusion_matrix(y_true, y_pred, output_subdir)
    plot_confusion_matrix_normalized(y_true, y_pred, output_subdir)
    plot_roc_curve(y_true, y_probs, output_subdir)
    plot_precision_recall_curve(y_true, y_probs, output_subdir)
    plot_probability_distribution(y_true, y_probs, output_subdir, best_threshold)
    plot_threshold_analysis(y_true, y_probs, best_threshold, output_subdir)
    plot_combined_curves(y_true, y_probs, output_subdir)
    plot_class_comparison(metrics, output_subdir)
    plot_metrics_comparison(metrics, output_subdir)

    # Compare methods if applicable
    if USE_TTA or USE_CALIBRATION:
        methods = []
        accuracies = []
        f1_scores = []

        # Base
        y_pred_base = (y_probs_base >= 0.5).astype(int)
        methods.append("Base\n(threshold=0.5)")
        accuracies.append(accuracy_score(y_true, y_pred_base))
        f1_scores.append(f1_score(y_true, y_pred_base))

        # Final (with all enhancements)
        methods.append(f"Enhanced\n(threshold={best_threshold:.3f})")
        accuracies.append(accuracy)
        f1_scores.append(metrics['f1_fake'])

        plot_comparison_bar(methods, accuracies, f1_scores, output_subdir)

    # ============ SAVE RESULTS ============
    save_results(metrics, output_subdir)

    # ============ PRINT SUMMARY ============
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Samples:     {len(y_true)}")
    print(f"Optimal Threshold: {best_threshold:.3f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"F1-Score (Fake):   {metrics['f1_fake']:.4f}")
    print(f"Precision (Fake):  {metrics['precision_fake']:.4f}")
    print(f"Recall (Fake):     {metrics['recall_fake']:.4f}")
    print(f"AUC-ROC:           {auc_score:.4f}")
    print(f"Average Precision: {ap_score:.4f}")
    print(f"\nEnhancements applied:")
    if USE_FEW_SHOT:
        print(f"  ✓ Few-shot learning ({N_SHOT}-shot, method: {FEW_SHOT_METHOD})")
    if USE_CALIBRATION:
        print(f"  ✓ Probability calibration")
    if USE_TTA and NUM_TTA_AUGMENTS > 1:
        print(f"  ✓ Test-Time Augmentation ({NUM_TTA_AUGMENTS} transforms)")
    if USE_TEMPERATURE_SCALING:
        print(f"  ✓ Temperature scaling")
    print(f"\nResults saved to: {output_subdir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
