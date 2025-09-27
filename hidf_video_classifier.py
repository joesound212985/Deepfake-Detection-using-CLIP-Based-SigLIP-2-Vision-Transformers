#!/usr/bin/env python3
"""
HIDF Video Binary Classifier — Fast & Comprehensive Analysis Edition
Real vs Fake video classification using SigLIP vision transformer with comprehensive training analysis.

🚀 Speed Optimizations (enabled by default):
- Fast ViT-B-16-SigLIP model (vs massive SO400M)
- Backbone freezing (train head only for speed)
- BF16 mixed precision + TF32 + cuDNN optimizations
- GPU-accelerated preprocessing with Kornia
- Optimized batch size (16) and frame sampling (4 frames)
- Model compilation with max-autotune

📊 Comprehensive Analysis (automatic):
- Real-time training curves with trend analysis
- Overfitting detection dashboard (12-panel analysis)
- Learning vs Memorization analysis (16-panel behavioral assessment)
- Publication-ready visualizations and metrics
- Statistical significance testing
- Error analysis and uncertainty quantification

Usage (fast training with full analysis):
  python hidf_video_classifier.py --data_dir "/path/to/hidf videos"

Usage (maximum accuracy - slower):
  python hidf_video_classifier.py --data_dir "/path/to/hidf videos" \
      --model_name "ViT-SO400M-16-SigLIP2-512" --image_size 512 --num_frames 8 \
      --batch_size 4 --accumulate_grad_batches 16 --freeze_backbone=False
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, cohen_kappa_score
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from scipy.ndimage import uniform_filter1d
import pandas as pd
import argparse
from tqdm import tqdm
import json
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import glob
import kornia
import kornia.augmentation as K
import math
import random

# Enable speed optimizations early
import torch.backends.cudnn as cudnn
torch.set_float32_matmul_precision('high')  # allow TF32
cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32 = True
    # Prefer Flash/Mem-efficient SDPA when available
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass

# Video processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Fast image loading optimizations
try:
    from turbojpeg import TurboJPEG
    TURBOJPEG_AVAILABLE = True
    turbo_jpeg = TurboJPEG()
except ImportError:
    TURBOJPEG_AVAILABLE = False
    turbo_jpeg = None

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("Warning: open_clip not available. Install with: pip install open-clip-torch")

# Import torch._dynamo at global scope to avoid local assignment issues
try:
    import torch._dynamo
    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False

def extract_frames_from_video(video_path, num_frames=4, target_size=224):
    """Extract frames from video using OpenCV"""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for video processing. Install with: pip install opencv-python")
    
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Sample frames uniformly across the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (target_size, target_size))
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_resized)
            frames.append(pil_frame)
        else:
            # If frame read fails, create a black frame
            black_frame = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            frames.append(Image.fromarray(black_frame))
    
    cap.release()
    
    # Ensure we have the requested number of frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.fromarray(np.zeros((target_size, target_size, 3), dtype=np.uint8)))
    
    return frames[:num_frames]

def fast_image_load(img_path):
    """Fast image loading using TurboJPEG or cv2 as fallback, else PIL."""
    try:
        if TURBOJPEG_AVAILABLE and img_path.lower().endswith(('.jpg', '.jpeg')):
            with open(img_path, 'rb') as f:
                arr = turbo_jpeg.decode(f.read())
            return Image.fromarray(arr)

        if CV2_AVAILABLE:
            arr = cv2.imread(img_path)
            if arr is not None:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(arr)

        return Image.open(img_path).convert('RGB')
    except Exception:
        return Image.open(img_path).convert('RGB')

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

class HIDFVideoDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, gpu_transform=None, 
                 num_frames=8, cache_tensors=False):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.gpu_transform = gpu_transform
        self.num_frames = num_frames
        self.cache_tensors = cache_tensors
        self.tensor_cache = {} if cache_tensors else None

        real_dir = os.path.join(data_dir, split.upper(), 'REAL')
        fake_dir = os.path.join(data_dir, split.upper(), 'FAKE')

        self.samples = []
        self.labels = []

        # Video extensions
        video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv')

        # Load REAL videos (label=0)
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(video_exts):
                    self.samples.append(os.path.join(real_dir, filename))
                    self.labels.append(0)

        # Load FAKE videos (label=1) 
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.lower().endswith(video_exts):
                    self.samples.append(os.path.join(fake_dir, filename))
                    self.labels.append(1)

        print(f"Loaded {len(self.samples)} {split} videos: {sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = self.samples[idx]
        label = self.labels[idx]

        # Check tensor cache first
        if self.cache_tensors and video_path in self.tensor_cache:
            return self.tensor_cache[video_path], label

        try:
            # Extract frames from video
            frames = extract_frames_from_video(video_path, self.num_frames)
            
            # Process frames and average their features
            processed_frames = []
            for frame in frames:
                if self.transform:
                    frame_tensor = self.transform(frame)
                    processed_frames.append(frame_tensor)
            
            # Stack frames: (num_frames, channels, height, width)
            if processed_frames:
                frames_tensor = torch.stack(processed_frames)
            else:
                # Fallback: create random noise frames  
                frames_tensor = torch.randn(self.num_frames, 3, 224, 224)

            # Cache the tensor if enabled
            if self.cache_tensors:
                self.tensor_cache[video_path] = frames_tensor

            return frames_tensor, label

        except Exception as e:
            print(f"Warning: Error processing video {video_path}: {e}")
            # Return random noise frames as fallback
            fallback_frames = torch.randn(self.num_frames, 3, 224, 224)
            return fallback_frames, label

class BinaryVideoClassifier(nn.Module):
    def __init__(self, model_name='ViT-B-16-SigLIP', device='cuda', num_frames=4, dropout_rate=0.3):
        super().__init__()
        
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open_clip required. Install with: pip install open-clip-torch")
        
        self.num_frames = num_frames
        
        # Load SigLIP vision encoder
        self.vision_encoder, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained='webli', device=device
        )
        
        # Get feature dimension from visual encoder
        if hasattr(self.vision_encoder, 'embed_dim'):
            self.feature_dim = self.vision_encoder.embed_dim
        elif hasattr(self.vision_encoder, 'num_features'):
            self.feature_dim = self.vision_encoder.num_features
        else:
            # Fallback: get dimension from a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224, device=device)
                dummy_features = self.vision_encoder.encode_image(dummy_input)
                self.feature_dim = dummy_features.shape[-1]
        
        # Temporal aggregation layer
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Binary classifier head with configurable regularization
        self.binary_classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),  # Configurable dropout
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),  # Gradually reduce dropout
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),  # Additional layer for capacity
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.33),  # Lower dropout in final layers
            nn.Linear(self.feature_dim // 4, 1)
        )
        
        # Initialize classifier layers
        for layer in self.binary_classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        print(f"Model: {model_name}, Feature dim: {self.feature_dim}, Frames: {num_frames}")

    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to process all frames together
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features for all frames
        frame_features = self.vision_encoder.encode_image(x)
        frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)  # Normalize
        
        # Reshape back to separate frames
        frame_features = frame_features.view(batch_size, num_frames, self.feature_dim)
        
        # Temporal pooling (average across frames)
        # frame_features: (batch_size, num_frames, feature_dim) -> (batch_size, feature_dim, num_frames)
        frame_features = frame_features.transpose(1, 2)
        pooled_features = self.temporal_pool(frame_features).squeeze(-1)  # (batch_size, feature_dim)
        
        # Binary classification
        logits = self.binary_classifier(pooled_features)
        return logits.squeeze(-1)

def evaluate(model, dataloader, criterion, device, gpu_transform=None, amp_dtype=torch.float16):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for frames_batch, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            frames_batch = frames_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if gpu_transform is not None:
                # Apply GPU transforms to each frame
                batch_size, num_frames, channels, height, width = frames_batch.shape
                frames_batch = frames_batch.view(batch_size * num_frames, channels, height, width)
                frames_batch = gpu_transform(frames_batch)
                frames_batch = frames_batch.view(batch_size, num_frames, channels, height, width)

            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(frames_batch)
                    loss = criterion(logits, labels.float())
            else:
                logits = model(frames_batch)
                loss = criterion(logits, labels.float())

            total_loss += loss.item()
            probs = torch.sigmoid(logits)

            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    # Calculate metrics - convert BF16 to float32 for numpy compatibility
    labels_np = torch.cat(all_labels).numpy()
    probs_np = torch.cat(all_probs).float().numpy()
    preds_np = (probs_np > 0.5).astype(int)

    accuracy = accuracy_score(labels_np, preds_np)
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds_np, average='binary', zero_division=0)
    auc = roc_auc_score(labels_np, probs_np)
    ap = average_precision_score(labels_np, probs_np)  # Average Precision
    mcc = matthews_corrcoef(labels_np, preds_np)
    cm = confusion_matrix(labels_np, preds_np)

    avg_loss = total_loss / max(1, len(dataloader))

    return avg_loss, accuracy, balanced_acc, precision, recall, f1, auc, ap, mcc, cm, labels_np, probs_np

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, 
                accumulate_grad_batches=1, gpu_transform=None, amp_dtype=torch.float16, 
                track_gradients=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    gradient_norms = []
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (frames_batch, labels) in enumerate(pbar):
        frames_batch = frames_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if gpu_transform is not None:
            # Apply GPU transforms to each frame
            batch_size, num_frames, channels, height, width = frames_batch.shape
            frames_batch = frames_batch.view(batch_size * num_frames, channels, height, width)
            frames_batch = gpu_transform(frames_batch)
            frames_batch = frames_batch.view(batch_size, num_frames, channels, height, width)

        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(frames_batch)
                loss = criterion(logits, labels.float()) / accumulate_grad_batches
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                scaler.unscale_(optimizer)
                if track_gradients:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    gradient_norms.append(grad_norm.item())
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(frames_batch)
            loss = criterion(logits, labels.float()) / accumulate_grad_batches
            loss.backward()
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                if track_gradients:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    gradient_norms.append(grad_norm.item())
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulate_grad_batches
        predicted = (torch.sigmoid(logits) > 0.5).long()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'Loss': f'{loss.item() * accumulate_grad_batches:.4f}',
                          'Acc': f'{100.*correct/total:.2f}%'})
    
    if track_gradients:
        return total_loss / max(1, len(dataloader)), 100. * correct / max(1, total), gradient_norms
    else:
        return total_loss / max(1, len(dataloader)), 100. * correct / max(1, total), []

# Plotting functions (same as image classifier)
def save_plots(labels, probs, save_dir, epoch=None):
    """Save separate publication-quality plots"""
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Determine filename prefix
    if epoch is not None:
        if isinstance(epoch, str):
            prefix = f'{epoch}_'
        else:
            prefix = f'epoch_{epoch+1}_'
    else:
        prefix = ''
    
    # Calculate metrics for plots
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    
    # 1. Confusion Matrix - Normalized
    plt.figure(figsize=(8, 6))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                cbar_kws={'label': 'Normalized Count'})
    plt.title('Normalized Confusion Matrix (Video)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Real', 'Fake'])
    plt.yticks([0.5, 1.5], ['Real', 'Fake'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Video Classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Metrics Bar Chart
    plt.figure(figsize=(12, 8))
    metrics = ['Accuracy', 'Bal. Acc', 'Precision', 'Recall', 'F1', 'AUC', 'AP']
    ap_metric = average_precision_score(labels, probs)
    values = [accuracy_score(labels, preds), balanced_accuracy_score(labels, preds),
              precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)[0],
              precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)[1],
              precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)[2],
              auc, ap_metric]
    
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf'])
    plt.ylabel('Score')
    plt.title('Performance Metrics (Video Classification)')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, linewidth=3, label=f'PR Curve (AP = {ap:.3f})')
    plt.axhline(y=labels.mean(), color='k', linestyle='--', linewidth=2, label=f'Random (AP = {labels.mean():.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Video Classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Prediction Distribution
    plt.figure(figsize=(10, 6))
    real_probs = probs[labels == 0]
    fake_probs = probs[labels == 1]
    
    plt.subplot(1, 2, 1)
    plt.hist(real_probs, bins=30, alpha=0.7, color='blue', label='Real Videos', density=True)
    plt.hist(fake_probs, bins=30, alpha=0.7, color='red', label='Fake Videos', density=True)
    plt.xlabel('Predicted Probability (Fake)')
    plt.ylabel('Density')
    plt.title('Distribution of Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([real_probs, fake_probs], labels=['Real', 'Fake'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
    plt.ylabel('Predicted Probability (Fake)')
    plt.title('Prediction Boxplots by Class')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}prediction_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Calibration Plot
    plt.figure(figsize=(8, 6))
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(labels, probs, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth=3, label='Model Calibration')
        plt.plot([0, 1], [0, 1], "k--", linewidth=2, label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot (Reliability Curve)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
    except Exception as e:
        # Fallback if calibration fails
        plt.text(0.5, 0.5, f'Calibration plot failed: {str(e)}', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Calibration Plot (Failed)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Raw Confusion Matrix (with counts)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Real', 'Fake'])
    plt.yticks([0.5, 1.5], ['Real', 'Fake'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Class-wise Performance Breakdown
    plt.figure(figsize=(12, 6))
    
    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    real_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  
    real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall) if (real_precision + real_recall) > 0 else 0
    
    fake_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall) if (fake_precision + fake_recall) > 0 else 0
    
    classes = ['Real', 'Fake']
    precisions = [real_precision, fake_precision]
    recalls = [real_recall, fake_recall]
    f1_scores = [real_f1, fake_f1]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precisions, width, label='Precision', color='#1f77b4', alpha=0.8)
    plt.bar(x, recalls, width, label='Recall', color='#ff7f0e', alpha=0.8)
    plt.bar(x + width, f1_scores, width, label='F1-Score', color='#2ca02c', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
        plt.text(i - width, p + 0.02, f'{p:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.text(i + width, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}class_performance_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. Threshold Analysis
    plt.figure(figsize=(10, 6))
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        pred_thresh = (probs > thresh).astype(int)
        if len(np.unique(pred_thresh)) > 1:  # Avoid division by zero
            acc = accuracy_score(labels, pred_thresh)
            prec = precision_recall_fscore_support(labels, pred_thresh, average='binary', zero_division=0)[0]
            rec = precision_recall_fscore_support(labels, pred_thresh, average='binary', zero_division=0)[1]
            f1 = precision_recall_fscore_support(labels, pred_thresh, average='binary', zero_division=0)[2]
        else:
            acc = prec = rec = f1 = 0
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    plt.plot(thresholds, accuracies, linewidth=2, label='Accuracy', color='blue')
    plt.plot(thresholds, precisions, linewidth=2, label='Precision', color='green')
    plt.plot(thresholds, recalls, linewidth=2, label='Recall', color='red')
    plt.plot(thresholds, f1s, linewidth=2, label='F1-Score', color='orange')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Score')
    plt.title('Performance vs Decision Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved 9 detailed plots to {save_dir}/ with prefix '{prefix}'")

def save_learning_rate_plot(learning_rates, save_dir):
    """Save learning rate schedule plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })
    
    epochs = range(1, len(learning_rates) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, linewidth=3, color='purple', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate_schedule.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved learning rate plot to {save_dir}/")

def save_gradient_norms_plot(gradient_norms, save_dir):
    """Save gradient norms plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })
    
    steps = range(1, len(gradient_norms) + 1)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, gradient_norms, linewidth=2, color='red', alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms During Training')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.hist(gradient_norms, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Gradient Norm')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gradient Norms')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradient_norms.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved gradient norms plot to {save_dir}/")

def bootstrap_confidence_intervals(labels, probs, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence intervals for metrics"""
    np.random.seed(42)  # For reproducibility
    
    metrics = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = resample(range(len(labels)), random_state=np.random.randint(0, 10000))
        y_boot = labels[indices]
        p_boot = probs[indices]
        pred_boot = (p_boot > 0.5).astype(int)
        
        # Calculate metrics
        if len(np.unique(y_boot)) > 1 and len(np.unique(pred_boot)) > 1:
            acc = accuracy_score(y_boot, pred_boot)
            prec, rec, f1, _ = precision_recall_fscore_support(y_boot, pred_boot, average='binary', zero_division=0)
            auc = roc_auc_score(y_boot, p_boot)
            metrics.append([acc, prec, rec, f1, auc])
        else:
            metrics.append([0, 0, 0, 0, 0.5])
    
    metrics = np.array(metrics)
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(metrics, lower_percentile, axis=0)
    ci_upper = np.percentile(metrics, upper_percentile, axis=0)
    
    return ci_lower, ci_upper

def statistical_significance_tests(labels, probs, save_dir):
    """Perform statistical significance tests and save results"""
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    
    results = {}
    
    # Chi-square test for independence
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(cm)
        results['chi_square'] = {'statistic': chi2, 'p_value': p_chi2, 'dof': dof}
    except:
        results['chi_square'] = {'statistic': np.nan, 'p_value': np.nan, 'dof': np.nan}
    
    # Fisher's exact test (for 2x2 contingency table)
    try:
        oddsratio, p_fisher = fisher_exact(cm)
        results['fisher_exact'] = {'odds_ratio': oddsratio, 'p_value': p_fisher}
    except:
        results['fisher_exact'] = {'odds_ratio': np.nan, 'p_value': np.nan}
    
    # McNemar's test (comparing against random classifier)
    random_preds = np.random.binomial(1, 0.5, len(labels))
    try:
        # Create contingency table for McNemar's test
        correct_model = (preds == labels)
        correct_random = (random_preds == labels)
        
        both_correct = np.sum(correct_model & correct_random)
        model_correct_random_wrong = np.sum(correct_model & ~correct_random)
        model_wrong_random_correct = np.sum(~correct_model & correct_random)
        both_wrong = np.sum(~correct_model & ~correct_random)
        
        mcnemar_table = np.array([[both_correct, model_wrong_random_correct],
                                 [model_correct_random_wrong, both_wrong]])
        
        # McNemar statistic
        b = model_wrong_random_correct
        c = model_correct_random_wrong
        if b + c > 0:
            mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
            p_mcnemar = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            mcnemar_stat = 0
            p_mcnemar = 1.0
            
        results['mcnemar'] = {'statistic': mcnemar_stat, 'p_value': p_mcnemar}
    except:
        results['mcnemar'] = {'statistic': np.nan, 'p_value': np.nan}
    
    # Bootstrap confidence intervals
    ci_lower, ci_upper = bootstrap_confidence_intervals(labels, probs)
    results['confidence_intervals'] = {
        'accuracy': {'lower': ci_lower[0], 'upper': ci_upper[0]},
        'precision': {'lower': ci_lower[1], 'upper': ci_upper[1]},
        'recall': {'lower': ci_lower[2], 'upper': ci_upper[2]},
        'f1': {'lower': ci_lower[3], 'upper': ci_upper[3]},
        'auc': {'lower': ci_lower[4], 'upper': ci_upper[4]}
    }
    
    # Save results to JSON
    with open(os.path.join(save_dir, 'statistical_tests.json'), 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    return results

def save_publication_table(labels, probs, save_dir, prefix=""):
    """Generate publication-ready performance table"""
    preds = (probs > 0.5).astype(int)
    
    # Calculate all metrics
    accuracy = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    
    # Confusion matrix metrics
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value (same as precision)
    
    # Bootstrap confidence intervals
    ci_lower, ci_upper = bootstrap_confidence_intervals(labels, probs)
    
    # Create DataFrame
    metrics_data = {
        'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision (PPV)', 'Recall (Sensitivity)', 
                  'Specificity', 'NPV', 'F1-Score', 'AUC-ROC', 'AP', 'MCC', 'Cohen\'s κ'],
        'Value': [accuracy, balanced_acc, precision, sensitivity, specificity, npv, f1, auc, ap, mcc, kappa],
        '95% CI Lower': [ci_lower[0], np.nan, ci_lower[1], ci_lower[2], np.nan, np.nan, ci_lower[3], ci_lower[4], np.nan, np.nan, np.nan],
        '95% CI Upper': [ci_upper[0], np.nan, ci_upper[1], ci_upper[2], np.nan, np.nan, ci_upper[3], ci_upper[4], np.nan, np.nan, np.nan]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Format for publication
    df['Formatted'] = df.apply(lambda row: 
        f"{row['Value']:.3f} ({row['95% CI Lower']:.3f}-{row['95% CI Upper']:.3f})" 
        if not pd.isna(row['95% CI Lower']) 
        else f"{row['Value']:.3f}", axis=1)
    
    # Save as CSV
    df.to_csv(os.path.join(save_dir, f'{prefix}performance_table.csv'), index=False)
    
    # Save as LaTeX table
    latex_table = df[['Metric', 'Formatted']].to_latex(
        index=False, 
        column_format='lc',
        caption='Performance Metrics with 95\\% Bootstrap Confidence Intervals',
        label='tab:performance',
        escape=False
    )
    
    with open(os.path.join(save_dir, f'{prefix}performance_table.tex'), 'w') as f:
        f.write(latex_table)
    
    return df

def save_error_analysis_plots(labels, probs, save_dir, prefix=""):
    """Analyze and visualize prediction errors"""
    preds = (probs > 0.5).astype(int)
    
    # Identify different types of predictions
    correct = (preds == labels)
    incorrect = ~correct
    
    # True/False Positives/Negatives
    tp_mask = (labels == 1) & (preds == 1)
    fp_mask = (labels == 0) & (preds == 1)
    tn_mask = (labels == 0) & (preds == 0)
    fn_mask = (labels == 1) & (preds == 0)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })
    
    # 1. Error Analysis by Confidence
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Confidence distribution by prediction type
    plt.subplot(2, 2, 1)
    tp_probs = probs[tp_mask]
    fp_probs = probs[fp_mask]
    tn_probs = 1 - probs[tn_mask]  # Convert to confidence in "real" class
    fn_probs = 1 - probs[fn_mask]
    
    plt.hist([tp_probs, fp_probs, tn_probs, fn_probs], bins=20, alpha=0.7, 
             label=['True Pos', 'False Pos', 'True Neg', 'False Neg'],
             color=['green', 'red', 'blue', 'orange'])
    plt.xlabel('Model Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution by Prediction Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Confidence vs Accuracy
    confidence_bins = np.linspace(0, 1, 11)
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (probs >= confidence_bins[i]) & (probs < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(correct[mask])
            bin_accuracies.append(bin_acc)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    plt.subplot(2, 2, 2)
    plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8, color='purple')
    plt.xlabel('Confidence Bin Center')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Model Confidence')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Subplot 3: Sample counts per bin
    plt.subplot(2, 2, 3)
    plt.bar(bin_centers, bin_counts, width=0.08, alpha=0.7, color='cyan')
    plt.xlabel('Confidence Bin Center')
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution Across Confidence Bins')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Error rates by confidence quartiles
    plt.subplot(2, 2, 4)
    quartiles = np.percentile(probs, [25, 50, 75])
    q_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    q_errors = []
    
    for i in range(4):
        if i == 0:
            mask = probs <= quartiles[0]
        elif i == 1:
            mask = (probs > quartiles[0]) & (probs <= quartiles[1])
        elif i == 2:
            mask = (probs > quartiles[1]) & (probs <= quartiles[2])
        else:
            mask = probs > quartiles[2]
        
        if np.sum(mask) > 0:
            error_rate = 1 - np.mean(correct[mask])
            q_errors.append(error_rate)
        else:
            q_errors.append(0)
    
    plt.bar(q_labels, q_errors, color=['red', 'orange', 'yellow', 'green'], alpha=0.7)
    plt.xlabel('Confidence Quartile')
    plt.ylabel('Error Rate')
    plt.title('Error Rate by Confidence Quartile')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'confidence_accuracy_correlation': np.corrcoef(probs, correct.astype(int))[0, 1],
        'high_confidence_accuracy': np.mean(correct[probs > 0.9]) if np.sum(probs > 0.9) > 0 else np.nan,
        'low_confidence_accuracy': np.mean(correct[probs < 0.1]) if np.sum(probs < 0.1) > 0 else np.nan
    }

def save_uncertainty_analysis(model, dataloader, device, save_dir, gpu_transform=None, 
                            amp_dtype=torch.float16, n_passes=10):
    """Analyze model uncertainty using Monte Carlo Dropout"""
    model.train()  # Enable dropout for uncertainty estimation
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for frames_batch, labels in tqdm(dataloader, desc="Uncertainty Analysis", leave=False):
            frames_batch = frames_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if gpu_transform is not None:
                batch_size, num_frames, channels, height, width = frames_batch.shape
                frames_batch = frames_batch.view(batch_size * num_frames, channels, height, width)
                frames_batch = gpu_transform(frames_batch)
                frames_batch = frames_batch.view(batch_size, num_frames, channels, height, width)
            
            # Multiple forward passes for uncertainty estimation
            predictions = []
            for _ in range(n_passes):
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(frames_batch)
                        probs = torch.sigmoid(logits)
                else:
                    logits = model(frames_batch)
                    probs = torch.sigmoid(logits)
                
                predictions.append(probs.cpu().numpy())
            
            predictions = np.array(predictions)  # Shape: (n_passes, batch_size)
            all_predictions.append(predictions)
            all_labels.append(labels.cpu().numpy())
    
    # Combine all predictions
    all_predictions = np.concatenate(all_predictions, axis=1)  # Shape: (n_passes, total_samples)
    all_labels = np.concatenate(all_labels)
    
    # Calculate uncertainty metrics
    mean_predictions = np.mean(all_predictions, axis=0)
    prediction_std = np.std(all_predictions, axis=0)
    
    # Epistemic uncertainty (model uncertainty)
    epistemic_uncertainty = prediction_std
    
    # Aleatoric uncertainty (data uncertainty) - approximated
    aleatoric_uncertainty = mean_predictions * (1 - mean_predictions)
    
    # Total uncertainty
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
    
    # Plot uncertainty analysis
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Epistemic uncertainty distribution
    axes[0, 0].hist(epistemic_uncertainty, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Epistemic Uncertainty')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Epistemic Uncertainty')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Aleatoric uncertainty distribution
    axes[0, 1].hist(aleatoric_uncertainty, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Aleatoric Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Aleatoric Uncertainty')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Total uncertainty distribution
    axes[0, 2].hist(total_uncertainty, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 2].set_xlabel('Total Uncertainty')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Total Uncertainty')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Uncertainty vs Accuracy
    correct = (mean_predictions > 0.5) == all_labels
    
    # Bin by uncertainty and calculate accuracy
    uncertainty_bins = np.linspace(0, total_uncertainty.max(), 10)
    bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2
    bin_accuracies = []
    
    for i in range(len(uncertainty_bins) - 1):
        mask = (total_uncertainty >= uncertainty_bins[i]) & (total_uncertainty < uncertainty_bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(correct[mask])
            bin_accuracies.append(bin_acc)
        else:
            bin_accuracies.append(0)
    
    axes[1, 0].plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Total Uncertainty')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Accuracy vs Uncertainty')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Prediction confidence vs uncertainty
    axes[1, 1].scatter(mean_predictions, total_uncertainty, alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Mean Prediction')
    axes[1, 1].set_ylabel('Total Uncertainty')
    axes[1, 1].set_title('Prediction Confidence vs Uncertainty')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Uncertainty by class
    real_uncertainty = total_uncertainty[all_labels == 0]
    fake_uncertainty = total_uncertainty[all_labels == 1]
    
    axes[1, 2].boxplot([real_uncertainty, fake_uncertainty], labels=['Real', 'Fake'], 
                      patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[1, 2].set_ylabel('Total Uncertainty')
    axes[1, 2].set_title('Uncertainty by True Class')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_epistemic_uncertainty': np.mean(epistemic_uncertainty),
        'mean_aleatoric_uncertainty': np.mean(aleatoric_uncertainty),
        'mean_total_uncertainty': np.mean(total_uncertainty),
        'uncertainty_accuracy_correlation': np.corrcoef(total_uncertainty, correct.astype(int))[0, 1]
    }

def save_temporal_analysis(model, video_paths, labels, save_dir, device, num_frames=8):
    """Analyze temporal patterns in video classification"""
    model.eval()
    
    # Sample a few videos for detailed analysis
    sample_indices = np.random.choice(len(video_paths), min(20, len(video_paths)), replace=False)
    sample_paths = [video_paths[i] for i in sample_indices]
    sample_labels = [labels[i] for i in sample_indices]
    
    frame_predictions = []
    frame_positions = []
    
    with torch.no_grad():
        for video_path, label in zip(sample_paths, sample_labels):
            try:
                # Extract frames at different positions
                cap = cv2.VideoCapture(video_path) if CV2_AVAILABLE else None
                if cap is None or not cap.isOpened():
                    continue
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames == 0:
                    cap.release()
                    continue
                
                # Extract frames at regular intervals
                frame_indices = np.linspace(0, total_frames - 1, num_frames * 2, dtype=int)
                
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # Convert and process single frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (224, 224))
                        pil_frame = Image.fromarray(frame_resized)
                        
                        # Transform frame
                        transform = transforms.Compose([
                            transforms.Resize((224, 224), antialias=True),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])
                        
                        frame_tensor = transform(pil_frame).unsqueeze(0).unsqueeze(0).to(device)
                        
                        # Get prediction for single frame (treat as 1-frame video)
                        logits = model(frame_tensor)
                        prob = torch.sigmoid(logits).item()
                        
                        frame_predictions.append(prob)
                        frame_positions.append(frame_idx / total_frames)  # Normalize position
                
                cap.release()
                
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                continue
    
    if frame_predictions:
        frame_predictions = np.array(frame_predictions)
        frame_positions = np.array(frame_positions)
        
        # Plot temporal analysis
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Prediction vs temporal position
        axes[0, 0].scatter(frame_positions, frame_predictions, alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Temporal Position (0=start, 1=end)')
        axes[0, 0].set_ylabel('Fake Probability')
        axes[0, 0].set_title('Predictions Across Video Timeline')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Temporal position distribution
        axes[0, 1].hist(frame_positions, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Temporal Position')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Analyzed Frame Positions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction distribution by temporal region
        early_preds = frame_predictions[frame_positions < 0.33]
        middle_preds = frame_predictions[(frame_positions >= 0.33) & (frame_positions < 0.67)]
        late_preds = frame_predictions[frame_positions >= 0.67]
        
        axes[1, 0].boxplot([early_preds, middle_preds, late_preds], 
                          labels=['Early', 'Middle', 'Late'],
                          patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        axes[1, 0].set_ylabel('Fake Probability')
        axes[1, 0].set_title('Predictions by Video Region')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Temporal consistency (prediction variance within videos)
        # This would require more complex tracking of which frames belong to which video
        axes[1, 1].hist(frame_predictions, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Fake Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Overall Prediction Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'temporal_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'temporal_correlation': np.corrcoef(frame_positions, frame_predictions)[0, 1],
            'early_region_mean': np.mean(early_preds) if len(early_preds) > 0 else np.nan,
            'middle_region_mean': np.mean(middle_preds) if len(middle_preds) > 0 else np.nan,
            'late_region_mean': np.mean(late_preds) if len(late_preds) > 0 else np.nan
        }
    
    return None

def save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, save_dir, epoch=None):
    """Save comprehensive training curves with real-time updates"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.facecolor': 'white'
    })
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create a comprehensive 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress - Video Classification', fontsize=20, fontweight='bold')
    
    # 1. Loss Curves
    ax1.plot(epochs, train_losses, 'b-', linewidth=3, label='Training Loss', marker='o', markersize=4, alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=4, alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Add current epoch marker if specified
    if epoch is not None and epoch < len(train_losses):
        ax1.axvline(x=epoch+1, color='purple', linestyle='--', alpha=0.7, linewidth=2, label=f'Current Epoch')
        ax1.legend()
    
    # 2. Accuracy Curves (fix the scaling issue)
    train_acc_normalized = [acc/100 if acc > 1 else acc for acc in train_accs]  # Handle percentage vs decimal
    val_acc_normalized = [acc*100 if acc <= 1 else acc for acc in val_accs]    # Handle decimal vs percentage
    
    ax2.plot(epochs, [acc*100 if acc <= 1 else acc for acc in train_acc_normalized], 'b-', linewidth=3, 
             label='Training Accuracy', marker='o', markersize=4, alpha=0.8)
    ax2.plot(epochs, val_acc_normalized, 'r-', linewidth=3, 
             label='Validation Accuracy', marker='s', markersize=4, alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    if epoch is not None and epoch < len(train_accs):
        ax2.axvline(x=epoch+1, color='purple', linestyle='--', alpha=0.7, linewidth=2)
    
    # 3. F1 Score and other metrics
    ax3.plot(epochs, val_f1s, 'g-', linewidth=3, label='Validation F1 Score', marker='^', markersize=4, alpha=0.8)
    
    # Add best F1 line
    if val_f1s:
        best_f1 = max(val_f1s)
        best_epoch = val_f1s.index(best_f1) + 1
        ax3.axhline(y=best_f1, color='gold', linestyle=':', alpha=0.8, linewidth=2, label=f'Best F1: {best_f1:.3f}')
        ax3.axvline(x=best_epoch, color='gold', linestyle=':', alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('Validation F1 Score Progression')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    if epoch is not None and epoch < len(val_f1s):
        ax3.axvline(x=epoch+1, color='purple', linestyle='--', alpha=0.7, linewidth=2)
    
    # 4. Combined Loss and Accuracy (dual y-axis)
    ax4_twin = ax4.twinx()
    
    # Plot loss on left y-axis
    line1 = ax4.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)
    line2 = ax4.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='black')
    ax4.set_yscale('log')
    ax4.tick_params(axis='y', labelcolor='black')
    
    # Plot F1 on right y-axis
    line3 = ax4_twin.plot(epochs, val_f1s, 'g-', linewidth=3, label='Val F1', marker='o', markersize=3, alpha=0.8)
    ax4_twin.set_ylabel('F1 Score', color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    ax4_twin.set_ylim([0, 1])
    
    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    ax4.set_title('Loss vs F1 Score Comparison')
    ax4.grid(True, alpha=0.3)
    
    if epoch is not None and epoch < len(epochs):
        ax4.axvline(x=epoch+1, color='purple', linestyle='--', alpha=0.7, linewidth=2)
        ax4_twin.axvline(x=epoch+1, color='purple', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    
    # Save with epoch info if provided
    if epoch is not None:
        filename = f'training_curves_epoch_{epoch+1}.png'
    else:
        filename = 'training_curves_final.png'
    
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save individual plots for clarity
    save_individual_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, save_dir, epoch)
    
    if epoch is not None:
        print(f"Saved training curves (epoch {epoch+1}) to {save_dir}/")
    else:
        print(f"Saved final training curves to {save_dir}/")

def save_individual_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, save_dir, epoch=None):
    """Save individual curve plots for detailed analysis"""
    epochs = range(1, len(train_losses) + 1)
    
    # Individual Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=3, label='Training Loss', marker='o', markersize=6, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=6, alpha=0.8)
    
    # Add smoothed trend lines
    if len(epochs) > 5:
        from scipy.ndimage import uniform_filter1d
        smooth_train = uniform_filter1d(train_losses, size=min(5, len(train_losses)//2))
        smooth_val = uniform_filter1d(val_losses, size=min(5, len(val_losses)//2))
        plt.plot(epochs, smooth_train, 'b--', alpha=0.5, linewidth=2, label='Train Trend')
        plt.plot(epochs, smooth_val, 'r--', alpha=0.5, linewidth=2, label='Val Trend')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Video Classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Accuracy Curves
    plt.figure(figsize=(10, 6))
    train_acc_plot = [acc/100 if acc > 1 else acc*100 for acc in train_accs]
    val_acc_plot = [acc*100 if acc <= 1 else acc for acc in val_accs]
    
    plt.plot(epochs, train_acc_plot, 'b-', linewidth=3, label='Training Accuracy', marker='o', markersize=6, alpha=0.8)
    plt.plot(epochs, val_acc_plot, 'r-', linewidth=3, label='Validation Accuracy', marker='s', markersize=6, alpha=0.8)
    
    # Add smoothed trend lines
    if len(epochs) > 5:
        smooth_train_acc = uniform_filter1d(train_acc_plot, size=min(5, len(train_acc_plot)//2))
        smooth_val_acc = uniform_filter1d(val_acc_plot, size=min(5, len(val_acc_plot)//2))
        plt.plot(epochs, smooth_train_acc, 'b--', alpha=0.5, linewidth=2, label='Train Trend')
        plt.plot(epochs, smooth_val_acc, 'r--', alpha=0.5, linewidth=2, label='Val Trend')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy (Video Classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual F1 Score Curve with additional metrics
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1s, 'g-', linewidth=3, label='Validation F1 Score', marker='^', markersize=6, alpha=0.8)
    
    # Add best F1 annotations
    if val_f1s:
        best_f1 = max(val_f1s)
        best_epoch = val_f1s.index(best_f1) + 1
        plt.axhline(y=best_f1, color='gold', linestyle=':', alpha=0.8, linewidth=2, label=f'Best F1: {best_f1:.3f} @ Epoch {best_epoch}')
        plt.axvline(x=best_epoch, color='gold', linestyle=':', alpha=0.8, linewidth=2)
        
        # Annotate the best point
        plt.annotate(f'Best: {best_f1:.3f}', xy=(best_epoch, best_f1), 
                    xytext=(best_epoch + len(epochs)*0.1, best_f1 + 0.05),
                    arrowprops=dict(arrowstyle='->', color='gold', alpha=0.8),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.3))
    
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Progression (Video Classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_metrics_plot(train_losses, train_accs, val_losses, val_accs, val_f1s, 
                              val_aucs, val_precisions, val_recalls, save_dir):
    """Save comprehensive metrics comparison plot"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'figure.facecolor': 'white'
    })
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create a comprehensive 2x3 subplot figure for detailed metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Training Metrics - Video Classification', fontsize=20, fontweight='bold')
    
    # 1. Loss comparison with error bands (if we had multiple runs)
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=3, label='Training Loss', marker='o', markersize=4, alpha=0.8)
    axes[0, 0].plot(epochs, val_losses, 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=4, alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Accuracy comparison
    train_acc_plot = [acc/100 if acc > 1 else acc*100 for acc in train_accs]
    val_acc_plot = [acc*100 if acc <= 1 else acc for acc in val_accs]
    
    axes[0, 1].plot(epochs, train_acc_plot, 'b-', linewidth=3, label='Training Accuracy', marker='o', markersize=4, alpha=0.8)
    axes[0, 1].plot(epochs, val_acc_plot, 'r-', linewidth=3, label='Validation Accuracy', marker='s', markersize=4, alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # 3. F1 Score with best marker
    axes[0, 2].plot(epochs, val_f1s, 'g-', linewidth=3, label='F1 Score', marker='^', markersize=4, alpha=0.8)
    if val_f1s:
        best_f1 = max(val_f1s)
        best_epoch = val_f1s.index(best_f1) + 1
        axes[0, 2].scatter([best_epoch], [best_f1], color='gold', s=100, zorder=5, label=f'Best: {best_f1:.3f}')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score Progression')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1])
    
    # 4. Precision, Recall, F1 comparison
    axes[1, 0].plot(epochs, val_precisions, 'purple', linewidth=2, label='Precision', marker='d', markersize=3, alpha=0.8)
    axes[1, 0].plot(epochs, val_recalls, 'orange', linewidth=2, label='Recall', marker='v', markersize=3, alpha=0.8)
    axes[1, 0].plot(epochs, val_f1s, 'green', linewidth=2, label='F1 Score', marker='^', markersize=3, alpha=0.8)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision, Recall, F1 Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # 5. AUC progression
    axes[1, 1].plot(epochs, val_aucs, 'navy', linewidth=3, label='AUC-ROC', marker='h', markersize=4, alpha=0.8)
    if val_aucs:
        best_auc = max(val_aucs)
        best_auc_epoch = val_aucs.index(best_auc) + 1
        axes[1, 1].scatter([best_auc_epoch], [best_auc], color='gold', s=100, zorder=5, label=f'Best: {best_auc:.3f}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC Score')
    axes[1, 1].set_title('AUC-ROC Progression')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0.5, 1])  # AUC should be > 0.5 for meaningful models
    
    # 6. Training dynamics (loss vs accuracy relationship)
    # Normalize data for comparison (avoid division by zero)
    train_loss_range = max(train_losses) - min(train_losses)
    val_acc_range = max(val_acc_plot) - min(val_acc_plot)
    
    if train_loss_range > 1e-8:
        norm_train_loss = [(l - min(train_losses)) / train_loss_range for l in train_losses]
    else:
        norm_train_loss = [0.5] * len(train_losses)  # Flat line if no variation
        
    if val_acc_range > 1e-8:
        norm_val_acc = [(a - min(val_acc_plot)) / val_acc_range for a in val_acc_plot]
    else:
        norm_val_acc = [0.5] * len(val_acc_plot)  # Flat line if no variation
    
    axes[1, 2].plot(epochs, norm_train_loss, 'b--', linewidth=2, label='Normalized Train Loss', alpha=0.7)
    axes[1, 2].plot(epochs, norm_val_acc, 'r-', linewidth=2, label='Normalized Val Accuracy', alpha=0.7)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Normalized Value')
    axes[1, 2].set_title('Training Dynamics')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_metrics_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detailed metrics curves to {save_dir}/")

def calculate_overfitting_metrics(train_losses, val_losses, train_accs, val_accs, val_f1s, window_size=5):
    """Calculate comprehensive overfitting detection metrics"""
    
    metrics = {}
    
    if len(train_losses) < 2 or len(val_losses) < 2:
        return metrics
    
    # 1. Loss Gap Analysis
    current_train_loss = train_losses[-1]
    current_val_loss = val_losses[-1]
    loss_gap = current_val_loss - current_train_loss
    loss_ratio = current_val_loss / max(current_train_loss, 1e-8)
    
    metrics['loss_gap'] = loss_gap
    metrics['loss_ratio'] = loss_ratio
    
    # 2. Accuracy Gap Analysis  
    current_train_acc = train_accs[-1] / 100 if train_accs[-1] > 1 else train_accs[-1]
    current_val_acc = val_accs[-1] * 100 if val_accs[-1] <= 1 else val_accs[-1]
    current_val_acc = current_val_acc / 100  # Normalize to [0,1]
    
    acc_gap = current_train_acc - current_val_acc
    metrics['accuracy_gap'] = acc_gap
    
    # 3. Trend Analysis (using moving averages)
    if len(train_losses) >= window_size:
        # Recent trends
        recent_train_loss_trend = np.mean(train_losses[-window_size:]) - np.mean(train_losses[-window_size*2:-window_size]) if len(train_losses) >= window_size*2 else 0
        recent_val_loss_trend = np.mean(val_losses[-window_size:]) - np.mean(val_losses[-window_size*2:-window_size]) if len(val_losses) >= window_size*2 else 0
        
        metrics['train_loss_trend'] = recent_train_loss_trend
        metrics['val_loss_trend'] = recent_val_loss_trend
        
        # Divergence: validation loss increasing while training loss decreasing
        is_diverging = recent_train_loss_trend < 0 and recent_val_loss_trend > 0
        metrics['is_diverging'] = is_diverging
        
        # Trend difference
        trend_difference = recent_val_loss_trend - recent_train_loss_trend
        metrics['trend_difference'] = trend_difference
    
    # 4. Best vs Current Performance Gap
    if val_f1s:
        best_val_f1 = max(val_f1s)
        current_val_f1 = val_f1s[-1]
        f1_degradation = best_val_f1 - current_val_f1
        metrics['f1_degradation'] = f1_degradation
        
        # Epochs since best performance
        best_f1_epoch = val_f1s.index(best_val_f1)
        epochs_since_best = len(val_f1s) - 1 - best_f1_epoch
        metrics['epochs_since_best'] = epochs_since_best
    
    # 5. Validation Loss Plateau Detection
    if len(val_losses) >= window_size:
        recent_val_losses = val_losses[-window_size:]
        val_loss_std = np.std(recent_val_losses)
        val_loss_variance = np.var(recent_val_losses)
        
        metrics['val_loss_std'] = val_loss_std
        metrics['val_loss_variance'] = val_loss_variance
        
        # Check if validation loss is plateauing (low variance)
        is_plateauing = val_loss_std < 0.01  # Threshold for plateau
        metrics['is_plateauing'] = is_plateauing
    
    # 6. Overall Overfitting Score (composite metric)
    overfitting_score = 0
    
    # Loss-based indicators
    if loss_gap > 0.1:  # Significant gap
        overfitting_score += min(loss_gap * 10, 3)  # Cap at 3 points
    
    if loss_ratio > 1.2:  # Val loss > 20% higher than train loss
        overfitting_score += min((loss_ratio - 1) * 5, 2)  # Cap at 2 points
    
    # Accuracy-based indicators  
    if acc_gap > 0.05:  # 5% accuracy gap
        overfitting_score += min(acc_gap * 20, 2)  # Cap at 2 points
    
    # Trend-based indicators
    if metrics.get('is_diverging', False):
        overfitting_score += 2
    
    if metrics.get('trend_difference', 0) > 0.05:
        overfitting_score += 1
    
    # Performance degradation
    if metrics.get('f1_degradation', 0) > 0.02:  # 2% F1 drop from best
        overfitting_score += min(metrics['f1_degradation'] * 50, 2)
    
    if metrics.get('epochs_since_best', 0) > 5:  # No improvement for 5 epochs
        overfitting_score += 1
    
    metrics['overfitting_score'] = min(overfitting_score, 10)  # Cap at 10
    
    # 7. Overfitting Risk Level
    if len(train_losses) < 3:  # Not enough data for reliable assessment
        risk_level = "UNKNOWN"
    elif overfitting_score < 2:
        risk_level = "LOW"
    elif overfitting_score < 5:
        risk_level = "MODERATE"
    elif overfitting_score < 8:
        risk_level = "HIGH"
    else:
        risk_level = "SEVERE"
    
    metrics['risk_level'] = risk_level
    
    return metrics

def save_overfitting_analysis(train_losses, train_accs, val_losses, val_accs, val_f1s, save_dir, epoch=None):
    """Create comprehensive overfitting analysis dashboard"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.facecolor': 'white'
    })
    
    epochs = range(1, len(train_losses) + 1)
    
    # Calculate overfitting metrics
    overfitting_metrics = calculate_overfitting_metrics(train_losses, val_losses, train_accs, val_accs, val_f1s)
    
    # Create comprehensive overfitting analysis figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main title with overfitting risk
    risk_level = overfitting_metrics.get('risk_level', 'UNKNOWN')
    risk_color = {'LOW': 'green', 'MODERATE': 'orange', 'HIGH': 'red', 'SEVERE': 'darkred'}.get(risk_level, 'black')
    
    fig.suptitle(f'Overfitting Analysis Dashboard - Risk Level: {risk_level}', 
                fontsize=18, fontweight='bold', color=risk_color)
    
    # 1. Loss Gap Analysis (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    # Fill area between curves to show gap
    ax1.fill_between(epochs, train_losses, val_losses, alpha=0.2, color='red')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Gap Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add gap annotation
    if overfitting_metrics.get('loss_gap'):
        gap_text = f"Gap: {overfitting_metrics['loss_gap']:.3f}"
        ax1.text(0.02, 0.98, gap_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                verticalalignment='top')
    
    # 2. Accuracy Gap Analysis (top-center-left)  
    ax2 = fig.add_subplot(gs[0, 1])
    train_acc_plot = [acc/100 if acc > 1 else acc*100 for acc in train_accs]
    val_acc_plot = [acc*100 if acc <= 1 else acc for acc in val_accs]
    
    ax2.plot(epochs, train_acc_plot, 'b-', linewidth=2, label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs, val_acc_plot, 'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
    
    # Fill area between curves
    ax2.fill_between(epochs, train_acc_plot, val_acc_plot, alpha=0.2, color='blue')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Gap Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    if overfitting_metrics.get('accuracy_gap'):
        gap_text = f"Gap: {overfitting_metrics['accuracy_gap']*100:.1f}%"
        ax2.text(0.02, 0.02, gap_text, transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # 3. Loss Ratio Over Time (top-center-right)
    ax3 = fig.add_subplot(gs[0, 2])
    loss_ratios = [val_losses[i] / max(train_losses[i], 1e-8) for i in range(len(epochs))]
    ax3.plot(epochs, loss_ratios, 'purple', linewidth=2, marker='o', markersize=3)
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal (Ratio=1)')
    ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Warning (Ratio=1.2)')
    ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Danger (Ratio=1.5)')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Val Loss / Train Loss')
    ax3.set_title('Loss Ratio Trend')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score Degradation (top-right)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(epochs, val_f1s, 'g-', linewidth=2, marker='^', markersize=3, label='Validation F1')
    
    if val_f1s:
        best_f1 = max(val_f1s)
        best_epoch = val_f1s.index(best_f1) + 1
        ax4.axhline(y=best_f1, color='gold', linestyle=':', alpha=0.8, label=f'Best F1: {best_f1:.3f}')
        ax4.axvline(x=best_epoch, color='gold', linestyle=':', alpha=0.8)
        ax4.scatter([best_epoch], [best_f1], color='gold', s=100, zorder=5)
        
        # Show degradation
        current_f1 = val_f1s[-1]
        degradation = best_f1 - current_f1
        if degradation > 0.01:
            ax4.annotate(f'Degradation: -{degradation:.3f}', 
                        xy=(len(epochs), current_f1), xytext=(len(epochs)*0.7, current_f1-0.1),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Degradation')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    # 5. Loss Trends (Moving Averages) (middle-left)
    ax5 = fig.add_subplot(gs[1, 0])
    window = 5
    if len(epochs) >= window:
        train_ma = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_ma = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        
        ax5.plot(ma_epochs, train_ma, 'b-', linewidth=2, label=f'Train MA({window})')
        ax5.plot(ma_epochs, val_ma, 'r-', linewidth=2, label=f'Val MA({window})')
        
        # Check for divergence
        if len(ma_epochs) > window:
            train_slope = np.polyfit(ma_epochs[-window:], train_ma[-window:], 1)[0]
            val_slope = np.polyfit(ma_epochs[-window:], val_ma[-window:], 1)[0]
            
            if train_slope < 0 and val_slope > 0:
                ax5.text(0.5, 0.95, 'DIVERGING!', transform=ax5.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.8),
                        ha='center', va='top', fontweight='bold', color='white')
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss (Moving Average)')
    ax5.set_title('Loss Trends (Smoothed)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_yscale('log')
    
    # 6. Overfitting Score Over Time (middle-center-left)
    ax6 = fig.add_subplot(gs[1, 1])
    
    overfitting_scores = []
    for i in range(3, len(epochs) + 1):  # Start from epoch 3
        temp_metrics = calculate_overfitting_metrics(
            train_losses[:i], val_losses[:i], train_accs[:i], val_accs[:i], val_f1s[:i]
        )
        overfitting_scores.append(temp_metrics.get('overfitting_score', 0))
    
    score_epochs = epochs[2:] if len(epochs) > 2 else epochs
    if overfitting_scores:
        ax6.plot(score_epochs, overfitting_scores, 'red', linewidth=3, marker='o', markersize=4)
        ax6.axhline(y=2, color='green', linestyle='--', alpha=0.7, label='Low Risk')
        ax6.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk')
        ax6.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='High Risk')
        
        # Current score highlight
        if overfitting_scores:
            current_score = overfitting_scores[-1]
            ax6.scatter([score_epochs[-1]], [current_score], color='darkred', s=100, zorder=5)
            ax6.text(score_epochs[-1], current_score + 0.2, f'{current_score:.1f}', 
                    ha='center', fontweight='bold')
    
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Overfitting Score')
    ax6.set_title('Overfitting Score Progression')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 10])
    
    # 7. Validation Loss Variance (middle-center-right)
    ax7 = fig.add_subplot(gs[1, 2])
    
    window = 5
    val_loss_variances = []
    variance_epochs = []
    
    for i in range(window, len(val_losses) + 1):
        window_losses = val_losses[i-window:i]
        variance = np.var(window_losses)
        val_loss_variances.append(variance)
        variance_epochs.append(i)
    
    if val_loss_variances:
        ax7.plot(variance_epochs, val_loss_variances, 'orange', linewidth=2, marker='s', markersize=3)
        ax7.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Plateau Threshold')
    
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel(f'Val Loss Variance (Window={window})')
    ax7.set_title('Loss Variance (Plateau Detection)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')
    
    # 8. Current Metrics Summary (middle-right)
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    # Create metrics summary text
    summary_text = f"""CURRENT METRICS SUMMARY
    
Loss Gap: {overfitting_metrics.get('loss_gap', 0):.3f}
Loss Ratio: {overfitting_metrics.get('loss_ratio', 1):.3f}
Accuracy Gap: {overfitting_metrics.get('accuracy_gap', 0)*100:.1f}%
F1 Degradation: {overfitting_metrics.get('f1_degradation', 0):.3f}
Epochs Since Best: {overfitting_metrics.get('epochs_since_best', 0)}

RISK INDICATORS:
Diverging Trends: {'YES' if overfitting_metrics.get('is_diverging', False) else 'NO'}
Loss Plateauing: {'YES' if overfitting_metrics.get('is_plateauing', False) else 'NO'}

OVERFITTING SCORE: {overfitting_metrics.get('overfitting_score', 0):.1f}/10
RISK LEVEL: {overfitting_metrics.get('risk_level', 'UNKNOWN')}"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # 9. Loss Distribution Comparison (bottom-left)
    ax9 = fig.add_subplot(gs[2, 0])
    
    # Recent losses for distribution
    recent_window = min(10, len(train_losses))
    recent_train = train_losses[-recent_window:]
    recent_val = val_losses[-recent_window:]
    
    ax9.hist(recent_train, bins=10, alpha=0.7, label='Train Loss Dist', color='blue', density=True)
    ax9.hist(recent_val, bins=10, alpha=0.7, label='Val Loss Dist', color='red', density=True)
    ax9.set_xlabel('Loss Value')
    ax9.set_ylabel('Density')
    ax9.set_title(f'Loss Distribution (Last {recent_window} Epochs)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Generalization Gap Over Time (bottom-center-left)
    ax10 = fig.add_subplot(gs[2, 1])
    
    gen_gaps = []
    for i in range(len(epochs)):
        train_acc_norm = train_accs[i] / 100 if train_accs[i] > 1 else train_accs[i]
        val_acc_norm = val_accs[i] * 100 if val_accs[i] <= 1 else val_accs[i]
        val_acc_norm = val_acc_norm / 100
        gap = train_acc_norm - val_acc_norm
        gen_gaps.append(gap)
    
    ax10.plot(epochs, gen_gaps, 'purple', linewidth=2, marker='d', markersize=3)
    ax10.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='Perfect Generalization')
    ax10.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Warning (5%)')
    ax10.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Overfitting (10%)')
    
    ax10.set_xlabel('Epoch')
    ax10.set_ylabel('Generalization Gap')
    ax10.set_title('Generalization Gap Trend')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    # 11. Early Stopping Signal (bottom-center-right)
    ax11 = fig.add_subplot(gs[2, 2])
    
    # Calculate early stopping signals
    patience_scores = []
    for i in range(len(val_f1s)):
        if i == 0:
            patience_scores.append(0)
        else:
            # Score based on epochs without improvement
            best_so_far = max(val_f1s[:i+1])
            current = val_f1s[i]
            if current >= best_so_far:
                patience_scores.append(0)  # Reset
            else:
                patience_scores.append(patience_scores[i-1] + 1)  # Increment
    
    ax11.plot(epochs, patience_scores, 'darkred', linewidth=2, marker='v', markersize=4)
    ax11.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Warning (5 epochs)')
    ax11.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Stop (10 epochs)')
    
    ax11.set_xlabel('Epoch')
    ax11.set_ylabel('Epochs Without Improvement')
    ax11.set_title('Early Stopping Signal')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Recommendations (bottom-right)
    ax12 = fig.add_subplot(gs[2, 3])
    ax12.axis('off')
    
    # Generate recommendations
    recommendations = []
    score = overfitting_metrics.get('overfitting_score', 0)
    
    if score < 2:
        recommendations.append("✅ Training looks healthy")
        recommendations.append("• Continue training")
        recommendations.append("• Monitor loss gap")
    elif score < 5:
        recommendations.append("⚠️ Moderate overfitting risk")
        recommendations.append("• Consider early stopping")
        recommendations.append("• Add regularization")
        recommendations.append("• Reduce learning rate")
    elif score < 8:
        recommendations.append("🚨 High overfitting risk")
        recommendations.append("• Stop training soon")
        recommendations.append("• Increase dropout")
        recommendations.append("• Add weight decay")
        recommendations.append("• Use data augmentation")
    else:
        recommendations.append("🛑 SEVERE overfitting")
        recommendations.append("• STOP training now")
        recommendations.append("• Use previous checkpoint")
        recommendations.append("• Redesign architecture")
        recommendations.append("• Collect more data")
    
    rec_text = "RECOMMENDATIONS:\n\n" + "\n".join(recommendations)
    
    ax12.text(0.05, 0.95, rec_text, transform=ax12.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    # Save the plot
    if epoch is not None:
        filename = f'overfitting_analysis_epoch_{epoch+1}.png'
    else:
        filename = 'overfitting_analysis_final.png'
        
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save metrics to JSON
    metrics_filename = 'overfitting_metrics.json' if epoch is None else f'overfitting_metrics_epoch_{epoch+1}.json'
    with open(os.path.join(save_dir, metrics_filename), 'w') as f:
        # Convert numpy types for JSON serialization
        json_metrics = {}
        for k, v in overfitting_metrics.items():
            if isinstance(v, np.floating):
                json_metrics[k] = float(v)
            elif isinstance(v, np.integer):
                json_metrics[k] = int(v)
            elif isinstance(v, np.bool_):
                json_metrics[k] = bool(v)
            elif isinstance(v, bool):
                json_metrics[k] = v
            else:
                json_metrics[k] = v
        json.dump(json_metrics, f, indent=2)
    
    if epoch is not None:
        print(f"Saved overfitting analysis (epoch {epoch+1}) - Risk Level: {risk_level}")
    else:
        print(f"Saved final overfitting analysis - Risk Level: {risk_level}")
    
    return overfitting_metrics

def calculate_learning_vs_memorization_metrics(model, train_loader, val_loader, test_loader, device, 
                                             criterion, train_losses, val_losses, train_accs, val_accs,
                                             gpu_transform=None, amp_dtype=torch.float16):
    """Comprehensive analysis to distinguish learning from memorization"""
    
    learning_metrics = {}
    
    print("🧠 Analyzing Learning vs Memorization...")
    
    # 1. Generalization Gap Analysis
    if len(train_losses) > 0 and len(val_losses) > 0:
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        generalization_gap = final_val_loss - final_train_loss
        
        learning_metrics['generalization_gap'] = generalization_gap
        learning_metrics['generalization_ratio'] = final_val_loss / max(final_train_loss, 1e-8)
    
    # 2. Learning Curve Shape Analysis
    if len(train_losses) >= 10:
        # Smooth learning curves
        window = min(5, len(train_losses) // 3)
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        
        # Calculate learning rates (slopes)
        mid_point = len(train_smooth) // 2
        early_train_slope = np.polyfit(range(mid_point), train_smooth[:mid_point], 1)[0]
        late_train_slope = np.polyfit(range(mid_point, len(train_smooth)), train_smooth[mid_point:], 1)[0]
        early_val_slope = np.polyfit(range(mid_point), val_smooth[:mid_point], 1)[0]
        late_val_slope = np.polyfit(range(mid_point, len(val_smooth)), val_smooth[mid_point:], 1)[0]
        
        learning_metrics['early_train_slope'] = early_train_slope
        learning_metrics['late_train_slope'] = late_train_slope
        learning_metrics['early_val_slope'] = early_val_slope
        learning_metrics['late_val_slope'] = late_val_slope
        
        # Learning consistency
        learning_consistency = abs(early_train_slope) / max(abs(late_train_slope), 1e-8)
        learning_metrics['learning_consistency'] = learning_consistency
        
        # Validation following training (good sign)
        val_follows_train = (early_train_slope < 0 and early_val_slope < 0) and (late_train_slope < 0 and late_val_slope < 0)
        learning_metrics['val_follows_train'] = val_follows_train
    
    # 3. Data Efficiency Analysis - Sample a subset and retrain quickly
    print("   Analyzing data efficiency...")
    
    # Create smaller dataset (25% of training data) to test data efficiency
    small_indices = torch.randperm(len(train_loader.dataset))[:len(train_loader.dataset) // 4]
    small_dataset = torch.utils.data.Subset(train_loader.dataset, small_indices.tolist())
    small_loader = DataLoader(small_dataset, batch_size=train_loader.batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    
    # Quick test: train on small dataset for few epochs
    model.train()
    optimizer_temp = torch.optim.AdamW(model.parameters(), lr=1e-4)
    small_losses = []
    small_accs = []
    
    for epoch in range(3):  # Quick test
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (frames_batch, labels) in enumerate(small_loader):
            if batch_idx >= 10:  # Limit batches for speed
                break
                
            frames_batch = frames_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if gpu_transform is not None:
                batch_size, num_frames, channels, height, width = frames_batch.shape
                frames_batch = frames_batch.view(batch_size * num_frames, channels, height, width)
                frames_batch = gpu_transform(frames_batch)
                frames_batch = frames_batch.view(batch_size, num_frames, channels, height, width)
            
            optimizer_temp.zero_grad()
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(frames_batch)
                    loss = criterion(logits, labels.float())
            else:
                logits = model(frames_batch)
                loss = criterion(logits, labels.float())
                
            loss.backward()
            optimizer_temp.step()
            
            epoch_loss += loss.item()
            predicted = (torch.sigmoid(logits) > 0.5).long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        if total > 0:
            small_losses.append(epoch_loss / min(10, len(small_loader)))
            small_accs.append(100. * correct / total)
    
    # Data efficiency metrics
    if len(small_losses) >= 2:
        small_improvement = small_losses[0] - small_losses[-1]
        learning_metrics['data_efficiency'] = small_improvement
        learning_metrics['small_dataset_final_acc'] = small_accs[-1]
        
        # Quick learner vs memorizer indicator
        if small_improvement > 0.1:
            learning_metrics['quick_learner'] = True
        else:
            learning_metrics['quick_learner'] = False
    
    # 4. Feature Learning Analysis - Activation statistics
    print("   Analyzing feature learning...")
    
    model.eval()
    activation_stats = []
    
    # Hook to capture intermediate activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks on classifier layers
    handles = []
    if hasattr(model, 'binary_classifier'):
        for i, layer in enumerate(model.binary_classifier):
            if isinstance(layer, torch.nn.Linear):
                handle = layer.register_forward_hook(get_activation(f'classifier_{i}'))
                handles.append(handle)
    
    # Sample activations from validation set
    val_activations = []
    with torch.no_grad():
        for batch_idx, (frames_batch, labels) in enumerate(val_loader):
            if batch_idx >= 5:  # Limit for speed
                break
                
            frames_batch = frames_batch.to(device, non_blocking=True)
            
            if gpu_transform is not None:
                batch_size, num_frames, channels, height, width = frames_batch.shape
                frames_batch = frames_batch.view(batch_size * num_frames, channels, height, width)
                frames_batch = gpu_transform(frames_batch)
                frames_batch = frames_batch.view(batch_size, num_frames, channels, height, width)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    _ = model(frames_batch)
            else:
                _ = model(frames_batch)
                
            # Collect activation statistics
            for name, activation in activations.items():
                if activation is not None:
                    val_activations.append({
                        'name': name,
                        'mean': torch.mean(activation).item(),
                        'std': torch.std(activation).item(),
                        'sparsity': (activation == 0).float().mean().item()
                    })
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    # Feature diversity metric
    if val_activations:
        mean_std = np.mean([act['std'] for act in val_activations])
        mean_sparsity = np.mean([act['sparsity'] for act in val_activations])
        
        learning_metrics['activation_diversity'] = mean_std
        learning_metrics['activation_sparsity'] = mean_sparsity
        
        # High diversity + low sparsity = good feature learning
        learning_metrics['feature_learning_score'] = mean_std * (1 - mean_sparsity)
    
    # 5. Prediction Confidence Analysis
    print("   Analyzing prediction confidence...")
    
    model.eval()
    train_confidences = []
    val_confidences = []
    
    # Sample confidences from training set
    with torch.no_grad():
        for batch_idx, (frames_batch, labels) in enumerate(train_loader):
            if batch_idx >= 10:  # Limit for speed
                break
                
            frames_batch = frames_batch.to(device, non_blocking=True)
            
            if gpu_transform is not None:
                batch_size, num_frames, channels, height, width = frames_batch.shape
                frames_batch = frames_batch.view(batch_size * num_frames, channels, height, width)
                frames_batch = gpu_transform(frames_batch)
                frames_batch = frames_batch.view(batch_size, num_frames, channels, height, width)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(frames_batch)
            else:
                logits = model(frames_batch)
                
            probs = torch.sigmoid(logits)
            # Confidence = distance from 0.5 (uncertainty)
            confidences = torch.abs(probs - 0.5).float().cpu().numpy()
            train_confidences.extend(confidences.tolist())
    
    # Sample confidences from validation set  
    with torch.no_grad():
        for batch_idx, (frames_batch, labels) in enumerate(val_loader):
            if batch_idx >= 10:  # Limit for speed
                break
                
            frames_batch = frames_batch.to(device, non_blocking=True)
            
            if gpu_transform is not None:
                batch_size, num_frames, channels, height, width = frames_batch.shape
                frames_batch = frames_batch.view(batch_size * num_frames, channels, height, width)
                frames_batch = gpu_transform(frames_batch)
                frames_batch = frames_batch.view(batch_size, num_frames, channels, height, width)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(frames_batch)
            else:
                logits = model(frames_batch)
                
            probs = torch.sigmoid(logits)
            confidences = torch.abs(probs - 0.5).float().cpu().numpy()
            val_confidences.extend(confidences.tolist())
    
    # Confidence analysis
    if train_confidences and val_confidences:
        train_conf_mean = np.mean(train_confidences)
        val_conf_mean = np.mean(val_confidences)
        
        learning_metrics['train_confidence'] = train_conf_mean
        learning_metrics['val_confidence'] = val_conf_mean
        learning_metrics['confidence_gap'] = train_conf_mean - val_conf_mean
        
        # Memorizers are often overconfident on training data
        learning_metrics['overconfidence_ratio'] = train_conf_mean / max(val_conf_mean, 1e-8)
    
    # 6. Learning vs Memorization Score (0-10 scale)
    memorization_score = 0
    
    # High generalization gap indicates memorization
    if learning_metrics.get('generalization_gap', 0) > 0.2:
        memorization_score += min(learning_metrics['generalization_gap'] * 10, 3)
    
    # Poor data efficiency indicates memorization
    if not learning_metrics.get('quick_learner', True):
        memorization_score += 2
    
    # Low feature diversity indicates memorization
    if learning_metrics.get('feature_learning_score', 1) < 0.1:
        memorization_score += 2
    
    # Overconfidence on training data indicates memorization
    if learning_metrics.get('overconfidence_ratio', 1) > 1.5:
        memorization_score += min((learning_metrics['overconfidence_ratio'] - 1) * 2, 2)
    
    # Poor validation following indicates memorization
    if not learning_metrics.get('val_follows_train', True):
        memorization_score += 1
    
    learning_metrics['memorization_score'] = min(memorization_score, 10)
    
    # Learning score (inverse of memorization)
    learning_metrics['learning_score'] = 10 - learning_metrics['memorization_score']
    
    # Classification
    if memorization_score < 3:
        learning_type = "LEARNING"
        confidence = "HIGH"
    elif memorization_score < 5:
        learning_type = "MIXED"
        confidence = "MODERATE"
    elif memorization_score < 7:
        learning_type = "MEMORIZING"
        confidence = "MODERATE"
    else:
        learning_type = "MEMORIZING"
        confidence = "HIGH"
    
    learning_metrics['learning_type'] = learning_type
    learning_metrics['confidence'] = confidence
    
    print(f"   Learning Analysis Complete: {learning_type} ({confidence} confidence)")
    
    return learning_metrics

def save_learning_vs_memorization_analysis(learning_metrics, train_losses, val_losses, 
                                         train_accs, val_accs, save_dir):
    """Create comprehensive learning vs memorization analysis dashboard"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.facecolor': 'white'
    })
    
    # Create comprehensive analysis figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    # Main title with learning classification
    learning_type = learning_metrics.get('learning_type', 'UNKNOWN')
    confidence = learning_metrics.get('confidence', 'UNKNOWN')
    learning_score = learning_metrics.get('learning_score', 0)
    
    color_map = {'LEARNING': 'green', 'MIXED': 'orange', 'MEMORIZING': 'red'}
    title_color = color_map.get(learning_type, 'black')
    
    fig.suptitle(f'Learning vs Memorization Analysis - {learning_type} (Score: {learning_score:.1f}/10)', 
                fontsize=18, fontweight='bold', color=title_color)
    
    epochs = range(1, len(train_losses) + 1) if train_losses else [1]
    
    # 1. Generalization Gap Evolution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if len(train_losses) > 0 and len(val_losses) > 0:
        gen_gaps = [val_losses[i] - train_losses[i] for i in range(len(epochs))]
        ax1.plot(epochs, gen_gaps, 'purple', linewidth=3, marker='o', markersize=4, alpha=0.8)
        ax1.axhline(y=0, color='green', linestyle='-', alpha=0.7, label='Perfect Generalization')
        ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Warning')
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Memorization')
        
        # Highlight current gap
        current_gap = learning_metrics.get('generalization_gap', 0)
        ax1.scatter([epochs[-1]], [current_gap], color='red', s=100, zorder=5)
        ax1.text(epochs[-1], current_gap + max(gen_gaps) * 0.1, f'{current_gap:.3f}', 
                ha='center', fontweight='bold')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Generalization Gap (Val - Train Loss)')
    ax1.set_title('Generalization Gap Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning Curve Shape Analysis (top-center-left)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if len(train_losses) >= 10:
        # Show smooth curves with slopes
        window = min(5, len(train_losses) // 3)
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = epochs[window-1:]
        
        ax2.plot(smooth_epochs, train_smooth, 'b-', linewidth=2, label='Train (Smoothed)', alpha=0.8)
        ax2.plot(smooth_epochs, val_smooth, 'r-', linewidth=2, label='Val (Smoothed)', alpha=0.8)
        
        # Add trend lines
        mid_point = len(smooth_epochs) // 2
        if mid_point > 1:
            early_epochs = smooth_epochs[:mid_point]
            late_epochs = smooth_epochs[mid_point:]
            
            early_train_fit = np.polyfit(range(len(early_epochs)), train_smooth[:mid_point], 1)
            late_train_fit = np.polyfit(range(len(late_epochs)), train_smooth[mid_point:], 1)
            
            ax2.plot(early_epochs, np.poly1d(early_train_fit)(range(len(early_epochs))), 
                    'b--', alpha=0.6, linewidth=2, label='Early Trend')
            ax2.plot(late_epochs, np.poly1d(late_train_fit)(range(len(late_epochs))), 
                    'b:', alpha=0.6, linewidth=2, label='Late Trend')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (Smoothed)')
    ax2.set_title('Learning Curve Shape')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Confidence Analysis (top-center-right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    train_conf = learning_metrics.get('train_confidence', 0)
    val_conf = learning_metrics.get('val_confidence', 0)
    
    categories = ['Training', 'Validation']
    confidences = [train_conf, val_conf]
    colors = ['blue', 'red']
    
    bars = ax3.bar(categories, confidences, color=colors, alpha=0.7)
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Maximum Confidence')
    
    # Add overconfidence indicator
    if learning_metrics.get('overconfidence_ratio', 1) > 1.2:
        ax3.text(0.5, max(confidences) * 0.8, 'OVERCONFIDENT', 
                transform=ax3.transData, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                fontweight='bold', color='white')
    
    for bar, conf in zip(bars, confidences):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_ylabel('Average Confidence')
    ax3.set_title('Prediction Confidence Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.5])
    
    # 4. Learning vs Memorization Score (top-right)
    ax4 = fig.add_subplot(gs[0, 3])
    
    learning_score = learning_metrics.get('learning_score', 0)
    memorization_score = learning_metrics.get('memorization_score', 0)
    
    # Gauge-style visualization
    scores = [learning_score, memorization_score]
    labels = ['Learning\nScore', 'Memorization\nScore']
    colors = ['green', 'red']
    
    bars = ax4.bar(labels, scores, color=colors, alpha=0.7)
    ax4.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Threshold')
    
    for bar, score in zip(bars, scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax4.set_ylabel('Score (0-10)')
    ax4.set_title('Learning vs Memorization Scores')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 10])
    
    # 5. Data Efficiency Indicator (second-row-left)
    ax5 = fig.add_subplot(gs[1, 0])
    
    quick_learner = learning_metrics.get('quick_learner', False)
    data_eff = learning_metrics.get('data_efficiency', 0)
    small_acc = learning_metrics.get('small_dataset_final_acc', 0)
    
    efficiency_text = f"""DATA EFFICIENCY ANALYSIS
    
Quick Learner: {'YES' if quick_learner else 'NO'}
Small Dataset Improvement: {data_eff:.3f}
Small Dataset Final Acc: {small_acc:.1f}%

INTERPRETATION:
{'✅ Efficient learner' if quick_learner else '❌ Requires lots of data'}
{'Good generalization' if data_eff > 0.1 else 'May be memorizing'}"""
    
    ax5.text(0.05, 0.95, efficiency_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    ax5.axis('off')
    ax5.set_title('Data Efficiency Analysis')
    
    # 6. Feature Learning Quality (second-row-center-left)
    ax6 = fig.add_subplot(gs[1, 1])
    
    diversity = learning_metrics.get('activation_diversity', 0)
    sparsity = learning_metrics.get('activation_sparsity', 0)
    feature_score = learning_metrics.get('feature_learning_score', 0)
    
    metrics = ['Activation\nDiversity', 'Activation\nSparsity', 'Feature Learning\nScore']
    values = [diversity, sparsity, feature_score]
    colors = ['blue', 'orange', 'green']
    
    bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
    
    for bar, value in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax6.set_ylabel('Score')
    ax6.set_title('Feature Learning Quality')
    ax6.grid(True, alpha=0.3)
    
    # 7. Learning Consistency (second-row-center-right)
    ax7 = fig.add_subplot(gs[1, 2])
    
    if learning_metrics.get('learning_consistency') is not None:
        consistency = learning_metrics['learning_consistency']
        early_slope = learning_metrics.get('early_train_slope', 0)
        late_slope = learning_metrics.get('late_train_slope', 0)
        
        phases = ['Early Phase', 'Late Phase']
        slopes = [abs(early_slope), abs(late_slope)]
        
        bars = ax7.bar(phases, slopes, color=['lightgreen', 'darkgreen'], alpha=0.7)
        
        for bar, slope in zip(bars, slopes):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(slopes) * 0.02,
                    f'{slope:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Add consistency indicator
        if consistency > 2:
            ax7.text(0.5, 0.8, 'INCONSISTENT', transform=ax7.transAxes, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                    fontweight='bold', color='white')
        elif consistency > 0.5:
            ax7.text(0.5, 0.8, 'CONSISTENT', transform=ax7.transAxes, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                    fontweight='bold', color='white')
    
    ax7.set_ylabel('Learning Rate (|slope|)')
    ax7.set_title(f'Learning Consistency (Ratio: {learning_metrics.get("learning_consistency", 0):.2f})')
    ax7.grid(True, alpha=0.3)
    
    # 8. Diagnostic Summary (second-row-right)
    ax8 = fig.add_subplot(gs[1, 3])
    
    # Create diagnostic summary
    diagnostics = []
    
    if learning_metrics.get('generalization_gap', 0) > 0.2:
        diagnostics.append("❌ High generalization gap")
    else:
        diagnostics.append("✅ Good generalization")
    
    if learning_metrics.get('quick_learner', False):
        diagnostics.append("✅ Data efficient")
    else:
        diagnostics.append("❌ Poor data efficiency")
        
    if learning_metrics.get('val_follows_train', False):
        diagnostics.append("✅ Val follows train loss")
    else:
        diagnostics.append("❌ Val diverges from train")
    
    if learning_metrics.get('overconfidence_ratio', 1) < 1.3:
        diagnostics.append("✅ Well-calibrated confidence")
    else:
        diagnostics.append("❌ Overconfident on training")
        
    if learning_metrics.get('feature_learning_score', 0) > 0.1:
        diagnostics.append("✅ Good feature diversity")
    else:
        diagnostics.append("❌ Poor feature learning")
    
    diagnostic_text = "DIAGNOSTIC SUMMARY:\n\n" + "\n".join(diagnostics)
    
    ax8.text(0.05, 0.95, diagnostic_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    ax8.axis('off')
    ax8.set_title('Diagnostic Summary')
    
    # 9-12. Bottom row: Detailed metric breakdowns
    
    # 9. Generalization Trajectory (third-row-left)
    ax9 = fig.add_subplot(gs[2, 0])
    
    if len(train_losses) > 0 and len(val_losses) > 0:
        gen_ratios = [val_losses[i] / max(train_losses[i], 1e-8) for i in range(len(epochs))]
        ax9.plot(epochs, gen_ratios, 'purple', linewidth=2, marker='d', markersize=3)
        ax9.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Perfect')
        ax9.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Warning')
        ax9.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Overfitting')
        
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Val Loss / Train Loss')
        ax9.set_title('Generalization Ratio Trajectory')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    # 10. Learning Rate Analysis (third-row-center-left)
    ax10 = fig.add_subplot(gs[2, 1])
    
    if len(train_losses) >= 10:
        # Calculate instantaneous learning rates
        train_rates = [-np.diff(train_losses)[i] for i in range(len(np.diff(train_losses)))]
        val_rates = [-np.diff(val_losses)[i] for i in range(len(np.diff(val_losses)))]
        rate_epochs = epochs[1:]
        
        ax10.plot(rate_epochs, train_rates, 'b-', alpha=0.7, label='Train Learning Rate')
        ax10.plot(rate_epochs, val_rates, 'r-', alpha=0.7, label='Val Learning Rate')
        ax10.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('Learning Rate (Loss Decrease)')
        ax10.set_title('Instantaneous Learning Rates')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
    
    # 11. Memorization Indicators (third-row-center-right)
    ax11 = fig.add_subplot(gs[2, 2])
    
    indicators = ['Gen Gap', 'Confidence\nGap', 'Data\nInefficiency', 'Feature\nPoor']
    indicator_scores = [
        min(learning_metrics.get('generalization_gap', 0) * 5, 3),
        min(learning_metrics.get('confidence_gap', 0) * 10, 2),
        0 if learning_metrics.get('quick_learner', False) else 2,
        2 if learning_metrics.get('feature_learning_score', 1) < 0.1 else 0
    ]
    
    colors = ['red' if score > 1 else 'green' for score in indicator_scores]
    bars = ax11.bar(indicators, indicator_scores, color=colors, alpha=0.7)
    
    ax11.set_ylabel('Memorization Indicator Strength')
    ax11.set_title('Individual Memorization Indicators')
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim([0, 3])
    
    # 12. Recommendations (third-row-right)
    ax12 = fig.add_subplot(gs[2, 3])
    
    recommendations = []
    learning_score = learning_metrics.get('learning_score', 0)
    
    if learning_score >= 7:
        recommendations.append("🎯 EXCELLENT LEARNING")
        recommendations.append("• Model is generalizing well")
        recommendations.append("• Continue current approach")
        recommendations.append("• Ready for deployment")
    elif learning_score >= 5:
        recommendations.append("✅ GOOD LEARNING")
        recommendations.append("• Minor memorization detected")
        recommendations.append("• Consider light regularization")
        recommendations.append("• Monitor generalization gap")
    elif learning_score >= 3:
        recommendations.append("⚠️ MIXED BEHAVIOR")
        recommendations.append("• Significant memorization risk")
        recommendations.append("• Add dropout/weight decay")
        recommendations.append("• Use data augmentation")
        recommendations.append("• Collect more diverse data")
    else:
        recommendations.append("🚨 STRONG MEMORIZATION")
        recommendations.append("• Model not learning patterns")
        recommendations.append("• Reduce model complexity")
        recommendations.append("• Increase regularization")
        recommendations.append("• Review data quality")
        recommendations.append("• Consider architecture change")
    
    rec_text = "\n".join(recommendations)
    
    ax12.text(0.05, 0.95, rec_text, transform=ax12.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.9))
    ax12.axis('off')
    ax12.set_title('Recommendations')
    
    # 13-16. Bottom row: Advanced analysis
    
    # 13. Loss Landscape Analysis (bottom-left)
    ax13 = fig.add_subplot(gs[3, 0])
    
    if len(train_losses) > 5:
        # Analyze loss landscape smoothness
        train_second_deriv = np.diff(train_losses, n=2)
        val_second_deriv = np.diff(val_losses, n=2)
        
        ax13.plot(epochs[2:], train_second_deriv, 'b-', alpha=0.7, label='Train Curvature')
        ax13.plot(epochs[2:], val_second_deriv, 'r-', alpha=0.7, label='Val Curvature')
        ax13.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax13.set_xlabel('Epoch')
        ax13.set_ylabel('Second Derivative (Curvature)')
        ax13.set_title('Loss Landscape Curvature')
        ax13.legend()
        ax13.grid(True, alpha=0.3)
    
    # 14. Validation Following Score (bottom-center-left)
    ax14 = fig.add_subplot(gs[3, 1])
    
    if len(train_losses) > 1 and len(val_losses) > 1:
        # Calculate correlation between train and val loss changes
        train_changes = np.diff(train_losses)
        val_changes = np.diff(val_losses)
        
        if len(train_changes) > 1 and len(val_changes) > 1:
            correlation = np.corrcoef(train_changes, val_changes)[0, 1]
            
            # Scatter plot of changes
            ax14.scatter(train_changes, val_changes, alpha=0.6, s=30)
            
            # Add correlation line
            z = np.polyfit(train_changes, val_changes, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(train_changes), max(train_changes), 100)
            ax14.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
            
            ax14.set_xlabel('Train Loss Change')
            ax14.set_ylabel('Val Loss Change')
            ax14.set_title(f'Val Following Train (r={correlation:.3f})')
            ax14.grid(True, alpha=0.3)
            
            # Add correlation interpretation
            if correlation > 0.8:
                ax14.text(0.05, 0.95, 'STRONG FOLLOWING', transform=ax14.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                         color='white', fontweight='bold')
            elif correlation > 0.5:
                ax14.text(0.05, 0.95, 'MODERATE FOLLOWING', transform=ax14.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7),
                         color='white', fontweight='bold')
            else:
                ax14.text(0.05, 0.95, 'POOR FOLLOWING', transform=ax14.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                         color='white', fontweight='bold')
    
    # 15. Complexity vs Performance (bottom-center-right)
    ax15 = fig.add_subplot(gs[3, 2])
    
    # Simple complexity analysis based on available metrics
    complexity_metrics = ['Model Size', 'Data Efficiency', 'Feature Diversity']
    complexity_values = [
        5,  # Placeholder for model size (normalized)
        learning_metrics.get('data_efficiency', 0) * 10,  # Scale data efficiency
        learning_metrics.get('activation_diversity', 0) * 20  # Scale diversity
    ]
    
    bars = ax15.bar(complexity_metrics, complexity_values, 
                   color=['gray', 'blue', 'green'], alpha=0.7)
    
    ax15.set_ylabel('Normalized Score')
    ax15.set_title('Model Complexity Factors')
    ax15.grid(True, alpha=0.3)
    
    # 16. Final Learning Assessment (bottom-right)
    ax16 = fig.add_subplot(gs[3, 3])
    
    # Create a comprehensive assessment
    assessment_text = f"""FINAL ASSESSMENT
    
Learning Type: {learning_type}
Confidence: {confidence}
Learning Score: {learning_score:.1f}/10
Memorization Score: {memorization_score:.1f}/10

KEY FINDINGS:
Gen. Gap: {learning_metrics.get('generalization_gap', 0):.3f}
Data Efficient: {'Yes' if learning_metrics.get('quick_learner', False) else 'No'}
Overconfident: {'Yes' if learning_metrics.get('overconfidence_ratio', 1) > 1.3 else 'No'}

CONCLUSION:
{'Model is learning patterns' if learning_score >= 5 else 'Model is memorizing data'}"""
    
    ax16.text(0.05, 0.95, assessment_text, transform=ax16.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", 
                      facecolor=color_map.get(learning_type, 'lightgray'), alpha=0.8))
    ax16.axis('off')
    ax16.set_title('Final Assessment')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_vs_memorization_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to JSON
    with open(os.path.join(save_dir, 'learning_vs_memorization_metrics.json'), 'w') as f:
        # Convert numpy types for JSON serialization
        json_metrics = {}
        for k, v in learning_metrics.items():
            if isinstance(v, np.floating):
                json_metrics[k] = float(v)
            elif isinstance(v, np.integer):
                json_metrics[k] = int(v)
            elif isinstance(v, np.bool_):
                json_metrics[k] = bool(v)
            else:
                json_metrics[k] = v
        json.dump(json_metrics, f, indent=2)
    
    print(f"Saved learning vs memorization analysis - {learning_type} behavior detected")
    
    return learning_metrics

def main():
    parser = argparse.ArgumentParser(description='HIDF Video Binary Classifier — Fast Training with Comprehensive Analysis')
    parser.add_argument('--real_dir', type=str, default='/mnt/c/Users/admin/Desktop/Real-vid/', help='Path to real videos directory')
    parser.add_argument('--fake_dir', type=str, default='/mnt/c/Users/admin/Desktop/Fake-vid/', help='Path to fake videos directory')
    parser.add_argument('--data_dir', type=str, default='/mnt/c/Users/admin/Desktop/hidf videos/', help='Dataset directory with TRAIN/VAL/TEST')
    parser.add_argument('--model_name', type=str, default='ViT-B-16-SigLIP', help='SigLIP model variant (default: fast ViT-B-16)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (optimized for ViT-B-16)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (reduced to prevent overfitting)')
    parser.add_argument('--save_dir', type=str, default='./hidf_video_checkpoints', help='Save directory')
    parser.add_argument('--accumulate_grad_batches', type=int, default=2, help='Gradient accumulation steps (optimized for ViT-B-16)')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs for learning rate')
    parser.add_argument('--eval_every_n_epochs', type=int, default=2, help='Evaluate every N epochs for speed')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, help='Freeze vision backbone for faster training')
    parser.add_argument('--compile_mode', type=str, default='max-autotune', choices=['default','reduce-overhead','max-autotune'], help='torch.compile mode')
    parser.add_argument('--image_size', type=int, default=224, help='Image size (224 for ViT-B-16, 512 for larger models)')
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['fp16','bf16'], help='AMP precision on CUDA')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers (lower for videos)')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames per video (4 for speed, 8 for accuracy)')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate using the best checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path to load (optional)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for regularization (0.01-0.1)')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for classifier head (0.1-0.5)')
    parser.add_argument('--data_augmentation', action='store_true', default=True, help='Enable data augmentation')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not os.path.exists(os.path.join(args.data_dir, 'TRAIN')):
        print(f"ERROR: No TRAIN split found in {args.data_dir}")
        return

    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' and device.type == 'cuda' else torch.float16
    print(f"Using AMP dtype: {amp_dtype}")

    # Print speed optimization status
    optimizations = []
    if device.type == 'cuda':
        optimizations.append("✅ GPU-accelerated preprocessing with kornia")
        optimizations.append(f"✅ {args.amp_dtype.upper()} mixed precision training")
        optimizations.append("✅ TF32 enabled for faster matmul")
    if CV2_AVAILABLE:
        optimizations.append("✅ OpenCV video processing")
    if TURBOJPEG_AVAILABLE:
        optimizations.append("✅ TurboJPEG fast frame decoding")

    if optimizations:
        print("🚀 Speed optimizations enabled:")
        for opt in optimizations:
            print(f"   {opt}")
    else:
        print("⚠️  No speed optimizations available")

    print(f"Using existing splits in: {args.data_dir}")

    # GPU-accelerated transforms with optional augmentation
    if device.type == 'cuda':
        if args.data_augmentation:
            print("🎲 Data augmentation enabled for better generalization")
            gpu_transform = nn.Sequential(
                K.Resize(args.image_size, antialias=True),
                K.RandomHorizontalFlip(p=0.5),  # Data augmentation
                K.RandomRotation(degrees=5, p=0.3),  # Slight rotation
                K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),  # Color augmentation
                K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ).to(device)
        else:
            gpu_transform = nn.Sequential(
                K.Resize(args.image_size, antialias=True),
                K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ).to(device)
        cpu_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor()
        ])
        print(f"🚀 Using GPU-accelerated preprocessing with kornia @ {args.image_size}px")
    else:
        gpu_transform = None
        cpu_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"Using CPU preprocessing @ {args.image_size}px (GPU not available)")

    # Datasets (no caching for videos due to memory)
    train_ds = HIDFVideoDataset(args.data_dir, 'train', cpu_transform, gpu_transform, args.num_frames, cache_tensors=False)
    val_ds   = HIDFVideoDataset(args.data_dir, 'val',   cpu_transform, gpu_transform, args.num_frames, cache_tensors=False)
    test_ds  = HIDFVideoDataset(args.data_dir, 'test',  cpu_transform, gpu_transform, args.num_frames, cache_tensors=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                              prefetch_factor=args.prefetch_factor, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                            prefetch_factor=args.prefetch_factor)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=True,
                             prefetch_factor=args.prefetch_factor)

    # Model
    model = BinaryVideoClassifier(args.model_name, device, args.num_frames, args.dropout_rate).to(device)

    # Freeze backbone for faster training if enabled
    if args.freeze_backbone:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen - training linear head only for faster training")

    # Compile model for speed (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            mode = None if args.compile_mode == 'default' else args.compile_mode
            model = torch.compile(model, mode=mode)
            print(f"✅ Model compiled for faster training (mode={args.compile_mode})")
        except Exception as e:
            print(f"⚠️  Model compilation failed ({e}), falling back to eager mode")
            if DYNAMO_AVAILABLE:
                torch._dynamo.config.suppress_errors = True

    # Convert model to channels_last for faster training
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    # Loss with automatic pos_weight for class imbalance
    neg_count = sum(1 for label in train_ds.labels if label == 0)
    pos_count = sum(1 for label in train_ds.labels if label == 1)
    pos_weight_val = neg_count / max(1, pos_count) if pos_count > 0 else 1.0
    pos_weight = torch.tensor(pos_weight_val, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"📊 Auto pos_weight = {pos_weight_val:.4f} (neg={neg_count}, pos={pos_count})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / max(1, args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    ensure_dir(args.save_dir)

    # Load checkpoint if provided
    start_epoch = 0
    best_f1 = -1.0

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if not args.evaluate_only and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_f1 = ckpt.get('val_f1', best_f1)
        print(f"Loaded checkpoint from {args.checkpoint} (epoch={start_epoch}, best_f1={best_f1:.4f})")

    # Evaluate-only mode
    if args.evaluate_only:
        print("Evaluate-only mode: loading best checkpoint if available and running on TEST...")
        if not args.checkpoint:
            default_ckpt = os.path.join(args.save_dir, 'best_hidf_video_binary.pth')
            if os.path.exists(default_ckpt):
                args.checkpoint = default_ckpt
                ckpt = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded {default_ckpt}")
        
        test_loss, test_acc, test_balanced_acc, test_precision, test_recall, test_f1, test_auc, test_ap, test_mcc, test_cm, test_labels, test_probs = evaluate(model, test_loader, criterion, device, gpu_transform, amp_dtype)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1: {test_f1:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        print(f"  AP: {test_ap:.4f}")
        print(f"  MCC: {test_mcc:.4f}")
        
        save_plots(test_labels, test_probs, args.save_dir, "test_best")
        return

    # Training loop with enhanced tracking
    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []
    val_aucs, val_precisions, val_recalls = [], [], []  # Additional metrics tracking
    learning_rates = []
    all_gradient_norms = []
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train (track gradients for first 10 epochs and last 5 epochs)
        track_grads = epoch < 10 or epoch >= args.epochs - 5
        train_loss, train_acc, gradient_norms = train_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                                           args.accumulate_grad_batches, gpu_transform, amp_dtype, track_grads)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        if gradient_norms:
            all_gradient_norms.extend(gradient_norms)
        
        # Record learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Evaluate on validation set (every N epochs for speed, plus first/last epochs)
        should_eval = (epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % args.eval_every_n_epochs == 0)
        
        if should_eval:
            val_loss, val_acc, val_balanced_acc, val_precision, val_recall, val_f1, val_auc, val_ap, val_mcc, val_cm, val_labels, val_probs = evaluate(model, val_loader, criterion, device, gpu_transform, amp_dtype)
        else:
            # Skip expensive validation, use previous values
            val_loss = val_losses[-1] if val_losses else 0.5
            val_acc = val_accs[-1] if val_accs else 0.5
            val_f1 = val_f1s[-1] if val_f1s else 0.5
            val_balanced_acc = val_precision = val_recall = val_auc = val_ap = val_mcc = 0.0
            
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Track additional metrics when available
        if should_eval:
            val_aucs.append(val_auc)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
        else:
            # Use previous values when not evaluating
            val_aucs.append(val_aucs[-1] if val_aucs else 0.5)
            val_precisions.append(val_precisions[-1] if val_precisions else 0.5)
            val_recalls.append(val_recalls[-1] if val_recalls else 0.5)
        
        # Save training curves and overfitting analysis every few epochs for monitoring
        if epoch % max(1, args.eval_every_n_epochs) == 0 or epoch == args.epochs - 1:
            save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir, epoch)
            
            # Generate overfitting analysis
            overfitting_metrics = save_overfitting_analysis(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir, epoch)
            
            # Check for severe overfitting and recommend stopping
            if overfitting_metrics.get('overfitting_score', 0) >= 8:
                print(f"\n🛑 SEVERE OVERFITTING DETECTED! Score: {overfitting_metrics['overfitting_score']:.1f}/10")
                print("   Recommendation: Consider stopping training and using an earlier checkpoint")
                
            elif overfitting_metrics.get('overfitting_score', 0) >= 5:
                print(f"\n🚨 HIGH OVERFITTING RISK! Score: {overfitting_metrics['overfitting_score']:.1f}/10")
                print("   Recommendation: Monitor closely, consider adding regularization")

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        if should_eval:
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Balanced Acc: {val_balanced_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            print(f"Val AUC: {val_auc:.4f}, AP: {val_ap:.4f}, MCC: {val_mcc:.4f}")
            print(f"Confusion Matrix:")
            print(f"  [[{val_cm[0,0]:5d} {val_cm[0,1]:5d}]]  (Real: True Neg, False Pos)")
            print(f"  [[{val_cm[1,0]:5d} {val_cm[1,1]:5d}]]  (Fake: False Neg, True Pos)")
            
            # Save best model and early stopping (only when evaluating)
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': val_f1,
                    'val_acc': val_acc
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'best_hidf_video_binary.pth'))
                print(f"🎯 New best F1: {val_f1:.4f} - model saved")
                save_plots(val_labels, val_probs, args.save_dir, epoch)
                # Update training curves with latest data
                save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir, epoch)
                
                # Generate overfitting analysis for best model
                overfitting_metrics = save_overfitting_analysis(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir, epoch)
                print(f"   Overfitting Score: {overfitting_metrics.get('overfitting_score', 0):.1f}/10 ({overfitting_metrics.get('risk_level', 'UNKNOWN')})")
            else:
                patience_counter += 1
                
                # Enhanced early stopping with overfitting detection
                overfitting_metrics = calculate_overfitting_metrics(train_losses, val_losses, train_accs, val_accs, val_f1s)
                overfitting_score = overfitting_metrics.get('overfitting_score', 0)
                
                # Reduce patience if overfitting is detected
                effective_patience = args.early_stopping_patience
                if overfitting_score >= 8:  # Severe overfitting
                    effective_patience = max(1, args.early_stopping_patience // 3)
                    print(f"   ⚠️ Severe overfitting detected - reducing patience to {effective_patience}")
                elif overfitting_score >= 5:  # High overfitting
                    effective_patience = max(2, args.early_stopping_patience // 2) 
                    print(f"   ⚠️ High overfitting risk - reducing patience to {effective_patience}")
                
                if patience_counter >= effective_patience:
                    stop_reason = "standard early stopping"
                    if overfitting_score >= 5:
                        stop_reason = f"overfitting detection (score: {overfitting_score:.1f}/10)"
                    print(f"Early stopping triggered after {epoch+1} epochs due to {stop_reason}")
                    break
        else:
            print(f"Val (cached): Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            patience_counter += 1  # Increment patience when skipping eval

    # Final test evaluation
    print(f"\n🧪 Final Test Evaluation:")
    test_loss, test_acc, test_balanced_acc, test_precision, test_recall, test_f1, test_auc, test_ap, test_mcc, test_cm, test_labels, test_probs = evaluate(model, test_loader, criterion, device, gpu_transform, amp_dtype)
    print(f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, AP: {test_ap:.4f}")

    # Save final comprehensive training curves and test plots
    save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir)
    
    # Save additional detailed metrics plot
    save_detailed_metrics_plot(train_losses, train_accs, val_losses, val_accs, val_f1s, 
                               val_aucs, val_precisions, val_recalls, args.save_dir)
    
    # Final overfitting analysis
    final_overfitting_metrics = save_overfitting_analysis(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir)
    print(f"\n📊 Final Overfitting Analysis:")
    print(f"   Score: {final_overfitting_metrics.get('overfitting_score', 0):.1f}/10")
    print(f"   Risk Level: {final_overfitting_metrics.get('risk_level', 'UNKNOWN')}")
    print(f"   Loss Gap: {final_overfitting_metrics.get('loss_gap', 0):.3f}")
    print(f"   Accuracy Gap: {final_overfitting_metrics.get('accuracy_gap', 0)*100:.1f}%")
    if final_overfitting_metrics.get('is_diverging', False):
        print(f"   ⚠️ Warning: Training and validation losses are diverging!")
    if final_overfitting_metrics.get('epochs_since_best', 0) > 3:
        print(f"   📉 Performance plateau: {final_overfitting_metrics['epochs_since_best']} epochs since best F1")
    save_plots(test_labels, test_probs, args.save_dir, "final_test")
    
    # Save additional analysis plots
    if learning_rates:
        save_learning_rate_plot(learning_rates, args.save_dir)
    if all_gradient_norms:
        save_gradient_norms_plot(all_gradient_norms, args.save_dir)
    
    # Publication-quality analysis
    print("\n📊 Generating publication-quality analysis...")
    
    # Learning vs Memorization Analysis
    learning_metrics = calculate_learning_vs_memorization_metrics(
        model, train_loader, val_loader, test_loader, device, criterion,
        train_losses, val_losses, train_accs, val_accs, gpu_transform, amp_dtype
    )
    save_learning_vs_memorization_analysis(learning_metrics, train_losses, val_losses, train_accs, val_accs, args.save_dir)
    
    # Statistical significance tests
    statistical_results = statistical_significance_tests(test_labels, test_probs, args.save_dir)
    
    # Performance table
    performance_df = save_publication_table(test_labels, test_probs, args.save_dir, "final_")
    
    # Error analysis
    error_results = save_error_analysis_plots(test_labels, test_probs, args.save_dir, "final_")
    
    # Uncertainty analysis (Monte Carlo Dropout)
    try:
        uncertainty_results = save_uncertainty_analysis(model, test_loader, device, args.save_dir, 
                                                       gpu_transform, amp_dtype, n_passes=10)
        print(f"✅ Uncertainty analysis completed")
    except Exception as e:
        print(f"⚠️  Uncertainty analysis failed: {e}")
    
    # Temporal analysis
    if CV2_AVAILABLE:
        try:
            temporal_results = save_temporal_analysis(model, test_ds.samples, test_ds.labels, 
                                                    args.save_dir, device, args.num_frames)
            if temporal_results:
                print(f"✅ Temporal analysis completed")
            else:
                print(f"⚠️  Temporal analysis had no valid results")
        except Exception as e:
            print(f"⚠️  Temporal analysis failed: {e}")
    
    # Summary of all analyses
    analysis_summary = {
        'dataset_stats': {
            'total_samples': len(test_ds),
            'fake_samples': sum(test_ds.labels),
            'real_samples': len(test_ds.labels) - sum(test_ds.labels)
        },
        'performance_metrics': {
            'accuracy': float(test_acc),
            'f1_score': float(test_f1),
            'auc_roc': float(test_auc),
            'average_precision': float(test_ap),
            'matthews_correlation': float(test_mcc)
        },
        'statistical_tests': statistical_results,
        'error_analysis': error_results
    }
    
    # Add learning vs memorization results
    analysis_summary['learning_analysis'] = learning_metrics
    
    # Add uncertainty results if available
    if 'uncertainty_results' in locals():
        analysis_summary['uncertainty_analysis'] = uncertainty_results
    
    # Add temporal results if available
    if 'temporal_results' in locals() and temporal_results:
        analysis_summary['temporal_analysis'] = temporal_results
    
    # Save comprehensive analysis summary
    with open(os.path.join(args.save_dir, 'comprehensive_analysis.json'), 'w') as f:
        json.dump(analysis_summary, f, indent=2, default=str)
    
    print(f"\n🎯 Analysis Complete! Generated comprehensive publication-ready materials:")
    print(f"   📈 {19 + (6 if 'uncertainty_results' in locals() else 0) + (4 if 'temporal_results' in locals() and temporal_results else 0)} visualization plots")
    print(f"   🔍 Overfitting Analysis Dashboard (12-panel comprehensive analysis)")
    print(f"   🧠 Learning vs Memorization Dashboard (16-panel behavioral analysis)")
    print(f"   📋 Performance table (CSV + LaTeX)")
    print(f"   📊 Statistical significance tests")
    print(f"   🔍 Error analysis and model insights")
    print(f"   🧠 Learning vs Memorization Analysis (16-panel dashboard)")
    print(f"   📄 Comprehensive analysis summary (JSON)")
    print(f"   📊 Overfitting metrics and recommendations (JSON)")
    print(f"   🎯 Learning behavior classification and metrics (JSON)")
    print(f"   📁 All files saved in: {args.save_dir}/")
    
    # Print key performance metrics with confidence intervals
    # Print learning analysis summary
    learning_type = learning_metrics.get('learning_type', 'UNKNOWN')
    learning_score = learning_metrics.get('learning_score', 0)
    memorization_score = learning_metrics.get('memorization_score', 0)
    
    print(f"\n🧠 Learning Analysis Summary:")
    print(f"   Learning Type: {learning_type}")
    print(f"   Learning Score: {learning_score:.1f}/10")
    print(f"   Memorization Score: {memorization_score:.1f}/10")
    print(f"   Generalization Gap: {learning_metrics.get('generalization_gap', 0):.3f}")
    print(f"   Data Efficient: {'Yes' if learning_metrics.get('quick_learner', False) else 'No'}")
    print(f"   Feature Learning Quality: {learning_metrics.get('feature_learning_score', 0):.3f}")
    
    if learning_score >= 7:
        print(f"   ✅ Excellent: Model is learning genuine patterns!")
    elif learning_score >= 5:
        print(f"   ✅ Good: Model shows healthy learning behavior")
    elif learning_score >= 3:
        print(f"   ⚠️ Mixed: Some memorization detected, monitor closely")
    else:
        print(f"   🚨 Warning: Strong memorization behavior detected!")
    
    print(f"\n🏆 Key Results (with 95% confidence intervals):")
    for _, row in performance_df.head(7).iterrows():  # Show main metrics
        print(f"   {row['Metric']:20s}: {row['Formatted']}")
    
    # Print statistical significance
    if statistical_results.get('fisher_exact', {}).get('p_value'):
        p_val = statistical_results['fisher_exact']['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n📈 Statistical Significance: p = {p_val:.6f} {significance}")
        print(f"   (Fisher's exact test vs. random classification)")
    
    # Final learning assessment
    if learning_type == "LEARNING":
        print(f"\n✨ EXCELLENT! Model demonstrates genuine learning behavior! ✨")
        print(f"   Your model is ready for publication and deployment.")
    elif learning_type == "MIXED":
        print(f"\n⚠️ CAUTION: Mixed learning/memorization behavior detected.")
        print(f"   Consider additional regularization before deployment.")
    else:
        print(f"\n🚨 WARNING: Strong memorization detected!")
        print(f"   Review model architecture and training approach.")
        
    print(f"\n✨ Comprehensive analysis complete - ready for publication! ✨")

if __name__ == '__main__':
    main()