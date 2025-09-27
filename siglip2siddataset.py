#!/usr/bin/env python3
# SigLIP-2 (ViT) + SegFormer-style decoder for SID_Set
# Logs paper metrics: Detection (Acc, F1) and Localization (AUC, F1, IoU)

import os, math, argparse, random, csv, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import kornia as K
import kornia.augmentation as KA
import kornia.morphology as KM
try:
    import kornia.enhance as KE
    KORNIA_CLAHE_AVAILABLE = hasattr(KE, 'Clahe') or hasattr(KE, 'clahe')
except ImportError:
    KE = None
    KORNIA_CLAHE_AVAILABLE = False

from datasets import load_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Comprehensive dual-memory optimizations
if torch.cuda.is_available():
    # Flash-Attention and SDPA optimizations
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    # TensorFloat-32 for speed with minimal precision loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True   # Enable for optimal kernel selection
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    # Memory allocation optimizations
    torch.cuda.set_per_process_memory_fraction(0.9)
    torch.cuda.empty_cache()

# CUDA memory allocator optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
from transformers import SiglipVisionModel, SiglipImageProcessor, get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, eps=1e-8):
    """Focal Loss for handling hard examples and class imbalance."""
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    return focal_loss.mean()

def morphological_postprocess(pred_masks, kernel_size=3, iterations=1):
    """Apply morphological operations to refine mask boundaries."""
    try:
        # Use Kornia morphological operations if available
        kernel = torch.ones(kernel_size, kernel_size, device=pred_masks.device)
        
        # Close small holes
        closed = KM.closing(pred_masks, kernel)
        
        # Remove small noise
        opened = KM.opening(closed, kernel)
        
        return opened
    except:
        # Fallback: simple smoothing
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=pred_masks.device) / (kernel_size**2)
        smoothed = F.conv2d(pred_masks, kernel, padding=kernel_size//2)
        return (smoothed > 0.5).float()

def boundary_aware_loss(logits, targets, kernel_size=3, eps=1e-8):
    """Enhanced boundary-aware loss using morphological operations."""
    try:
        # Use Kornia morphological operations for better boundary detection
        kernel = torch.ones(kernel_size, kernel_size, device=targets.device)
        dilated = KM.dilation(targets, kernel)
        eroded = KM.erosion(targets, kernel) 
        boundary_mask = (dilated - eroded).detach()
    except:
        # Fallback: convolution-based morphology
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=targets.device)
        dilated = F.conv2d(targets, kernel, padding=kernel_size//2) > 0
        eroded = F.conv2d(targets, kernel, padding=kernel_size//2) == kernel_size**2
        boundary_mask = (dilated.float() - eroded.float()).detach()
    
    # Apply higher weight to boundary regions
    boundary_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    boundary_weighted = boundary_loss * (1 + 3 * boundary_mask)  # 4x weight on boundaries
    return boundary_weighted.mean()

def morphological_loss(logits, targets, kernel_size=3):
    """Morphological consistency loss to encourage shape coherence."""
    p = torch.sigmoid(logits)
    
    try:
        kernel = torch.ones(kernel_size, kernel_size, device=p.device)
        
        # Morphological opening should be consistent
        p_opened = KM.opening(p, kernel)
        t_opened = KM.opening(targets, kernel)
        opening_loss = F.mse_loss(p_opened, t_opened)
        
        # Morphological closing should be consistent  
        p_closed = KM.closing(p, kernel)
        t_closed = KM.closing(targets, kernel)
        closing_loss = F.mse_loss(p_closed, t_closed)
        
        return (opening_loss + closing_loss) / 2
    except:
        # Fallback: smoothness penalty
        grad_x = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
        grad_y = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
        return grad_x.mean() + grad_y.mean()

def iou_loss(logits, targets, smooth=1e-6):
    """Direct IoU loss optimization."""
    p = torch.sigmoid(logits)
    intersection = (p * targets).sum(dim=(1,2,3))
    union = p.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection + smooth
    iou = intersection / union
    return 1 - iou.mean()

def combined_segmentation_loss(logits, targets, bce_w=0.4, focal_w=0.3, dice_w=0.5, 
                              boundary_w=0.4, iou_w=0.4, morph_w=0.2, eps=1e-6):
    """Enhanced multi-component loss with morphological consistency."""
    # Traditional losses
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    
    # Dice loss
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1,2,3))
    denom = p.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    dice = 1 - (2 * inter / denom).mean()
    
    # Advanced losses for better boundaries and IoU
    focal = focal_loss(logits, targets)
    boundary = boundary_aware_loss(logits, targets)
    iou = iou_loss(logits, targets)
    morph = morphological_loss(logits, targets)
    
    # Combine all losses
    total_loss = (bce_w * bce + focal_w * focal + dice_w * dice + 
                  boundary_w * boundary + iou_w * iou + morph_w * morph)
    
    return total_loss

def bce_dice_loss(logits, targets, bce_w=1.0, dice_w=0.5, eps=1e-6):
    """Legacy BCE+Dice loss (kept for compatibility)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(1,2,3))
    denom = p.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    dice = 1 - (2 * inter / denom).mean()
    return bce_w * bce + dice_w * dice

def dice_iou_from_logits(logits, targets, thr=0.5, eps=1e-6):
    p_bin = (torch.sigmoid(logits) > thr).float()
    inter = (p_bin*targets).sum(dim=(1,2,3))
    union = (p_bin + targets - p_bin*targets).sum(dim=(1,2,3)) + eps
    dice = (2*inter / (p_bin.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps))
    iou  = (inter / union)
    return dice.detach().cpu().tolist(), iou.detach().cpu().tolist(), p_bin

def color_overlay_fp_fn(img_rgb: np.ndarray, gt: np.ndarray, pr: np.ndarray, alpha=0.45):
    # TP=green, FP=red, FN=blue overlay
    img = img_rgb.astype(np.float32)
    tp = (pr==1) & (gt==1)
    fp = (pr==1) & (gt==0)
    fn = (pr==0) & (gt==1)
    overlay = np.zeros_like(img)
    overlay[...,1] += tp*255
    overlay[...,0] += fp*255
    overlay[...,2] += fn*255
    out = (1-alpha)*img + alpha*overlay
    return np.clip(out,0,255).astype(np.uint8)

def sweep_mask_thresholds(seg_logits_list: List[torch.Tensor], masks_list: List[torch.Tensor], 
                         thr_min=0.1, thr_max=0.9, thr_steps=17):
    """Sweep mask thresholds to find optimal F1, Dice, and IoU."""
    thresholds = np.linspace(thr_min, thr_max, thr_steps)
    best_metrics = {'f1': 0, 'dice': 0, 'iou': 0, 'thr_f1': 0.5, 'thr_dice': 0.5, 'thr_iou': 0.5}
    
    # Concatenate all logits and masks
    all_logits = torch.cat(seg_logits_list, dim=0)  # (N, 1, H, W)
    all_masks = torch.cat(masks_list, dim=0)        # (N, 1, H, W)
    
    for thr in thresholds:
        dice_scores, iou_scores, pred_bin = dice_iou_from_logits(all_logits, all_masks, thr=thr)
        
        # Calculate F1 score for segmentation masks
        pred_flat = pred_bin.flatten().cpu().numpy()
        mask_flat = all_masks.flatten().cpu().numpy()
        
        if len(np.unique(mask_flat)) > 1:  # Avoid division by zero
            f1 = f1_score(mask_flat, pred_flat, zero_division=0)
        else:
            f1 = 0
            
        avg_dice = np.mean(dice_scores) if dice_scores else 0
        avg_iou = np.mean(iou_scores) if iou_scores else 0
        
        # Track best metrics
        if f1 > best_metrics['f1']:
            best_metrics['f1'] = f1
            best_metrics['thr_f1'] = thr
        if avg_dice > best_metrics['dice']:
            best_metrics['dice'] = avg_dice
            best_metrics['thr_dice'] = thr
        if avg_iou > best_metrics['iou']:
            best_metrics['iou'] = avg_iou
            best_metrics['thr_iou'] = thr
    
    return best_metrics

# -----------------------------
# Visualization and Analysis Functions
# -----------------------------
def create_results_table(metrics_csv, output_dir):
    """Create formatted results table from metrics CSV."""
    try:
        df = pd.read_csv(metrics_csv)
        # Round values for better readability
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        
        # Save as enhanced CSV
        results_csv = os.path.join(output_dir, "results_table.csv")
        df.to_csv(results_csv, index=False)
        
        # Create markdown table
        markdown_path = os.path.join(output_dir, "results_table.md")
        with open(markdown_path, 'w') as f:
            f.write("# Training Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Best Metrics\n")
            if len(df) > 0:
                # Handle NaN values in idxmax to avoid FutureWarning
                if not df['f1'].isna().all():
                    best_f1_idx = df['f1'].idxmax()
                    f.write(f"- **Best F1**: {df.loc[best_f1_idx, 'f1']:.4f} (Epoch {df.loc[best_f1_idx, 'epoch']})\n")
                else:
                    f.write("- **Best F1**: No valid F1 scores recorded\n")
                
                if 'dice' in df.columns and not df['dice'].isna().all():
                    best_dice_idx = df['dice'].idxmax()
                    f.write(f"- **Best Dice**: {df.loc[best_dice_idx, 'dice']:.4f} (Epoch {df.loc[best_dice_idx, 'epoch']})\n")
                else:
                    f.write("- **Best Dice**: No valid Dice scores recorded\n")
        
        print(f"📊 Results table saved: {results_csv} & {markdown_path}")
        return df
    except Exception as e:
        print(f"⚠️ Failed to create results table: {e}")
        return None

def create_overlay_collage(images, gt_masks, pred_masks, ids, epoch, save_dir, max_samples=10):
    """Create validation overlay collage with GT, predictions, error analysis, and IoU values."""
    try:
        if len(images) == 0:
            return
            
        # Calculate metrics for all samples to select the best ones
        sample_metrics = []
        for i in range(len(images)):
            gt = gt_masks[i]
            pred = pred_masks[i]
            
            # Calculate IoU
            intersection = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            iou = intersection / (union + 1e-8) if union > 0 else 0.0
            
            # Calculate Dice score
            dice = (2 * intersection) / (gt.sum() + pred.sum() + 1e-8) if (gt.sum() + pred.sum()) > 0 else 0.0
            
            # Calculate error diversity (higher is better for visualization)
            tp = (gt == 1) & (pred == 1)
            fp = (gt == 0) & (pred == 1)
            fn = (gt == 1) & (pred == 0)
            error_ratio = (fp.sum() + fn.sum()) / (gt.size + 1e-8)
            
            sample_metrics.append((i, iou, dice, error_ratio))
        
        # Sort by IoU score (descending), then by error ratio for diversity
        sample_metrics.sort(key=lambda x: (x[1], x[3]), reverse=True)
        
        # Select best samples (top IoU scores with some diversity)
        selected_indices = [x[0] for x in sample_metrics[:max_samples]]
        n_samples = len(selected_indices)
        
        # Create compact figure
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3.5*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
            
        iou_scores = []
        for plot_idx, sample_idx in enumerate(selected_indices):
            img = images[sample_idx]
            gt = gt_masks[sample_idx]
            pred = pred_masks[sample_idx]
            
            # Calculate IoU for this sample
            intersection = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            iou = intersection / (union + 1e-8) if union > 0 else 0.0
            iou_scores.append(iou)
            
            # Calculate Dice score
            dice = (2 * intersection) / (gt.sum() + pred.sum() + 1e-8) if (gt.sum() + pred.sum()) > 0 else 0.0
            
            # Create error overlay (TP=Green, FP=Red, FN=Blue)
            overlay = img.copy().astype(np.float32)
            tp = (gt == 1) & (pred == 1)
            fp = (gt == 0) & (pred == 1) 
            fn = (gt == 1) & (pred == 0)
            
            # Apply color coding with transparency
            overlay[tp] = [0, 200, 0]    # Green for TP
            overlay[fp] = [200, 0, 0]    # Red for FP  
            overlay[fn] = [0, 100, 200]  # Blue for FN
            overlay = (0.65 * img + 0.35 * overlay).astype(np.uint8)
            
            # Plot: Original, GT, Prediction, Overlay
            axes[plot_idx, 0].imshow(img)
            axes[plot_idx, 0].set_title(f'Original\n{ids[sample_idx][:12]}', fontsize=10)
            axes[plot_idx, 0].axis('off')
            
            axes[plot_idx, 1].imshow(gt, cmap='gray')
            axes[plot_idx, 1].set_title('Ground Truth', fontsize=10)
            axes[plot_idx, 1].axis('off')
            
            axes[plot_idx, 2].imshow(pred, cmap='gray')
            axes[plot_idx, 2].set_title('Prediction', fontsize=10)
            axes[plot_idx, 2].axis('off')
            
            axes[plot_idx, 3].imshow(overlay)
            axes[plot_idx, 3].set_title(f'Error Analysis\nIoU: {iou:.3f} | Dice: {dice:.3f}', fontsize=10)
            axes[plot_idx, 3].axis('off')
        
        # Add compact title and legend
        fig.suptitle(f'Validation Results - Epoch {epoch} | Avg IoU: {np.mean(iou_scores) if iou_scores else 0.0:.3f}', 
                     fontsize=14, y=0.95)
        
        # Create compact legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='TP'),
            Patch(facecolor='red', alpha=0.7, label='FP'),  
            Patch(facecolor='blue', alpha=0.7, label='FN'),
            Patch(facecolor='lightgray', alpha=0.7, label='TN')
        ]
        
        # Add legend below title
        fig.legend(handles=legend_elements, loc='upper center', ncol=4, 
                   bbox_to_anchor=(0.5, 0.89), fontsize=10, frameon=False)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.87])  # Minimal margins
        
        collage_path = os.path.join(save_dir, f'validation_collage_epoch_{epoch:02d}.png')
        plt.savefig(collage_path, dpi=150, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        
        avg_iou = np.mean(iou_scores) if iou_scores else 0.0
        print(f"🖼️ Validation collage saved: {collage_path} (Avg IoU: {avg_iou:.3f})")
        
    except Exception as e:
        print(f"⚠️ Failed to create overlay collage: {e}")

def plot_training_curves(metrics_csv, output_dir):
    """Plot comprehensive training curves."""
    try:
        # Handle potential CSV format issues
        df = pd.read_csv(metrics_csv, on_bad_lines='skip')
        if len(df) < 1:
            print("No data available for plotting yet")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = df['epoch']
        
        # Plot losses (training vs validation)
        if 'train_loss' in df.columns:
            axes[0, 0].plot(epochs, df['train_loss'], 'r-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, df['val_loss'], 'b-', label='Validation Loss', linewidth=2)
        if 'test_loss' in df.columns:
            axes[0, 0].plot(epochs, df['test_loss'], 'g-', label='Test Loss', linewidth=2)
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot classification metrics
        axes[0, 1].plot(epochs, df['acc'], 'b-', label='Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, df['f1'], 'r-', label='F1', linewidth=2)
        axes[0, 1].set_title('Classification Metrics')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot AUC metrics
        axes[0, 2].plot(epochs, df['auc'], 'purple', label='AUC', linewidth=2)
        axes[0, 2].plot(epochs, df['ap'], 'orange', label='AP', linewidth=2)
        if 'pix_auc' in df.columns:
            axes[0, 2].plot(epochs, df['pix_auc'], 'brown', label='Pixel AUC', alpha=0.7)
        axes[0, 2].set_title('AUC Metrics')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot segmentation metrics
        axes[1, 0].plot(epochs, df['dice'], 'g-', label='Dice', linewidth=2)
        axes[1, 0].plot(epochs, df['iou'], 'b-', label='IoU', linewidth=2)
        axes[1, 0].set_title('Segmentation Metrics')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot threshold sweep results if available
        if 'best_dice' in df.columns:
            axes[1, 1].plot(epochs, df['dice'], 'g--', label='Fixed Thresh Dice', alpha=0.7)
            axes[1, 1].plot(epochs, df['best_dice'], 'g-', label='Optimal Dice', linewidth=2)
            axes[1, 1].plot(epochs, df['iou'], 'b--', label='Fixed Thresh IoU', alpha=0.7)
            axes[1, 1].plot(epochs, df['best_iou'], 'b-', label='Optimal IoU', linewidth=2)
            axes[1, 1].set_title('Threshold Optimization')
            axes[1, 1].legend()
        else:
            axes[1, 1].plot(epochs, df['dice'], 'g-', label='Dice')
            axes[1, 1].plot(epochs, df['iou'], 'b-', label='IoU')
            axes[1, 1].set_title('Segmentation Quality')
            axes[1, 1].legend()
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        axes[1, 2].plot(epochs, [0.001] * len(epochs), 'k--', alpha=0.5, label='Base LR')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training curves saved: {curves_path}")
    except Exception as e:
        print(f"⚠️ Failed to plot training curves: {e}")

def create_confusion_matrix(y_true, y_pred, epoch, save_dir):
    """Create and save confusion matrix with numerical values."""
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Plot heatmap with numbers
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Authentic', 'Tampered'],
                   yticklabels=['Authentic', 'Tampered'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Classification Confusion Matrix - Epoch {epoch}', fontsize=14, weight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add accuracy text
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        plt.figtext(0.5, 0.02, f'Accuracy: {accuracy:.3f}', ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        
        # Save matrix
        matrix_path = os.path.join(save_dir, f'confusion_matrix_classification_epoch_{epoch:02d}.png')
        plt.savefig(matrix_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Also save numerical values as text
        values_path = os.path.join(save_dir, f'confusion_matrix_values_epoch_{epoch:02d}.txt')
        with open(values_path, 'w') as f:
            f.write(f"Classification Confusion Matrix - Epoch {epoch}\n")
            f.write("=" * 40 + "\n")
            f.write(f"True Negatives (Authentic→Authentic): {cm[0,0]}\n")
            f.write(f"False Positives (Authentic→Tampered): {cm[0,1]}\n") 
            f.write(f"False Negatives (Tampered→Authentic): {cm[1,0]}\n")
            f.write(f"True Positives (Tampered→Tampered): {cm[1,1]}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {cm[1,1]/(cm[1,1]+cm[0,1]+1e-8):.4f}\n")
            f.write(f"Recall: {cm[1,1]/(cm[1,1]+cm[1,0]+1e-8):.4f}\n")
        
        print(f"📊 Classification confusion matrix saved: {matrix_path}")
        print(f"📝 Matrix values saved: {values_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to create confusion matrix: {e}")

def create_3class_confusion_matrix(y_true, y_pred, epoch, save_dir):
    """Create and save 3-class confusion matrix (Real/Synthetic/Tampered)."""
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns
        
        # Calculate 3x3 confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        class_names = ['Real', 'Synthetic', 'Tampered']
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap with numbers
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'3-Class Confusion Matrix - Epoch {epoch}', fontsize=16, weight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        
        # Calculate overall accuracy
        accuracy = np.trace(cm) / cm.sum()
        plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.3f}', ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        
        # Save 3-class matrix
        matrix_path = os.path.join(save_dir, f'confusion_matrix_3class_epoch_{epoch:02d}.png')
        plt.savefig(matrix_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Generate detailed classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Save detailed metrics as text
        values_path = os.path.join(save_dir, f'classification_report_3class_epoch_{epoch:02d}.txt')
        with open(values_path, 'w') as f:
            f.write(f"3-Class Classification Report - Epoch {epoch}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("                 Predicted\n")
            f.write("          Real  Synthetic  Tampered\n")
            f.write(f"Real      {cm[0,0]:4d}      {cm[0,1]:4d}      {cm[0,2]:4d}\n")
            f.write(f"Synthetic {cm[1,0]:4d}      {cm[1,1]:4d}      {cm[1,2]:4d}\n") 
            f.write(f"Tampered  {cm[2,0]:4d}      {cm[2,1]:4d}      {cm[2,2]:4d}\n\n")
            
            f.write("Per-Class Metrics:\n")
            for i, class_name in enumerate(class_names):
                if str(i) in report:
                    metrics = report[str(i)]
                    f.write(f"{class_name:>10}: Precision={metrics['precision']:.4f}, "
                           f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}, "
                           f"Support={metrics['support']:d}\n")
            
            f.write(f"\nOverall Metrics:\n")
            f.write(f"Accuracy: {report['accuracy']:.4f}\n")
            f.write(f"Macro Avg: Precision={report['macro avg']['precision']:.4f}, "
                   f"Recall={report['macro avg']['recall']:.4f}, F1={report['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted Avg: Precision={report['weighted avg']['precision']:.4f}, "
                   f"Recall={report['weighted avg']['recall']:.4f}, F1={report['weighted avg']['f1-score']:.4f}\n")
        
        print(f"🎯 3-Class confusion matrix saved: {matrix_path}")
        print(f"📊 Detailed classification report saved: {values_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to create 3-class confusion matrix: {e}")

def create_iou_confusion_matrix(seg_masks_gt, seg_masks_pred, seg_dice_list, seg_iou_list, epoch, save_dir):
    """Create IoU-based confusion matrix for segmentation performance."""
    try:
        import seaborn as sns
        
        if not seg_masks_gt or not seg_masks_pred:
            return
            
        # Calculate pixel-level confusion matrix for segmentation
        all_gt = np.concatenate([mask.flatten() for mask in seg_masks_gt])
        all_pred = np.concatenate([mask.flatten() for mask in seg_masks_pred])
        
        from sklearn.metrics import confusion_matrix
        seg_cm = confusion_matrix(all_gt, all_pred)
        
        # Create segmentation confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pixel-level confusion matrix
        sns.heatmap(seg_cm, annot=True, fmt='d', cmap='Oranges', 
                   xticklabels=['Background', 'Tampered'],
                   yticklabels=['Background', 'Tampered'],
                   cbar_kws={'label': 'Pixel Count'}, ax=axes[0])
        
        axes[0].set_title(f'Pixel-Level Segmentation - Epoch {epoch}', fontsize=12, weight='bold')
        axes[0].set_xlabel('Predicted Pixels')
        axes[0].set_ylabel('True Pixels')
        
        # IoU distribution histogram
        if seg_iou_list:
            axes[1].hist(seg_iou_list, bins=20, color='orange', alpha=0.7, edgecolor='black')
            axes[1].axvline(np.mean(seg_iou_list), color='red', linestyle='--', linewidth=2, 
                           label=f'Mean IoU: {np.mean(seg_iou_list):.3f}')
            axes[1].set_title(f'IoU Distribution - Epoch {epoch}', fontsize=12, weight='bold')
            axes[1].set_xlabel('IoU Score')
            axes[1].set_ylabel('Sample Count')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save segmentation matrix
        seg_matrix_path = os.path.join(save_dir, f'confusion_matrix_segmentation_epoch_{epoch:02d}.png')
        plt.savefig(seg_matrix_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # Save IoU statistics
        iou_stats_path = os.path.join(save_dir, f'iou_statistics_epoch_{epoch:02d}.txt')
        with open(iou_stats_path, 'w') as f:
            f.write(f"IoU Statistics - Epoch {epoch}\n")
            f.write("=" * 30 + "\n")
            f.write(f"Pixel-Level Segmentation Confusion Matrix:\n")
            f.write(f"True Negatives (BG→BG): {seg_cm[0,0]:,}\n")
            f.write(f"False Positives (BG→Tampered): {seg_cm[0,1]:,}\n")
            f.write(f"False Negatives (Tampered→BG): {seg_cm[1,0]:,}\n") 
            f.write(f"True Positives (Tampered→Tampered): {seg_cm[1,1]:,}\n")
            
            pixel_acc = (seg_cm[0,0] + seg_cm[1,1]) / seg_cm.sum()
            pixel_prec = seg_cm[1,1] / (seg_cm[1,1] + seg_cm[0,1] + 1e-8)
            pixel_recall = seg_cm[1,1] / (seg_cm[1,1] + seg_cm[1,0] + 1e-8)
            
            f.write(f"\nPixel-Level Metrics:\n")
            f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")
            f.write(f"Pixel Precision: {pixel_prec:.4f}\n")
            f.write(f"Pixel Recall: {pixel_recall:.4f}\n")
            
            if seg_iou_list:
                f.write(f"\nIoU Statistics:\n")
                f.write(f"Mean IoU: {np.mean(seg_iou_list):.4f}\n")
                f.write(f"Std IoU: {np.std(seg_iou_list):.4f}\n")
                f.write(f"Min IoU: {np.min(seg_iou_list):.4f}\n")
                f.write(f"Max IoU: {np.max(seg_iou_list):.4f}\n")
                f.write(f"Median IoU: {np.median(seg_iou_list):.4f}\n")
                
            if seg_dice_list:
                f.write(f"\nDice Statistics:\n")
                f.write(f"Mean Dice: {np.mean(seg_dice_list):.4f}\n")
                f.write(f"Std Dice: {np.std(seg_dice_list):.4f}\n")
        
        print(f"🎯 Segmentation confusion matrix saved: {seg_matrix_path}")
        print(f"📊 IoU statistics saved: {iou_stats_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to create IoU confusion matrix: {e}")

# -----------------------------
# SegFormer-style decoder (strong)
# -----------------------------
class LinearProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.proj(x)

class SegFormerStrongDecoder(nn.Module):
    def __init__(self, in_dims: List[int], embed_dim=256, dropout_rate=0.0):
        super().__init__()
        K = len(in_dims)
        self.projs = nn.ModuleList([LinearProj(d, embed_dim) for d in in_dims])
        self.dropout_rate = dropout_rate
        
        # Add dropout to smooth layers if specified
        smooth_layers = []
        for _ in in_dims:
            layers = [
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.GELU(),
            ]
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(p=dropout_rate))
            smooth_layers.append(nn.Sequential(*layers))
        self.smooth = nn.ModuleList(smooth_layers)
        
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(embed_dim*K, (embed_dim*K)//4, 1), nn.GELU(),
            nn.Conv2d((embed_dim*K)//4, embed_dim*K, 1), nn.Sigmoid()
        )
        
        # Final fusion layers with optional dropout
        fuse_layers = [nn.Conv2d(embed_dim*K, embed_dim, 1)]
        if dropout_rate > 0:
            fuse_layers.append(nn.Dropout2d(p=dropout_rate))
        self.fuse = nn.Sequential(*fuse_layers)
        
        self.head = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, hidden_list: List[torch.Tensor], grid_hw: Tuple[int,int], target_size: int = 448):
        H, W = grid_hw
        feats = []
        for i, h in enumerate(hidden_list):           # (B,N,C)
            x = self.projs[i](h).transpose(1,2)       # (B,E,N)
            B,E,N = x.shape
            x = x.reshape(B,E,H,W)                    # (B,E,H,W)
            x = self.smooth[i](x)
            feats.append(x)
        x = torch.cat(feats, dim=1)                   # (B,E*K,H,W)
        x = self.fuse_attn(x) * x
        x = self.fuse(x)
        # Upsample to target image size instead of fixed scale_factor
        x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return self.head(x)                           # (B,1,target_size,target_size)

# -----------------------------
# Model: SigLIP-2 + SegFormer
# -----------------------------
class SigLIP2_MTL(nn.Module):
    def __init__(self, siglip_ckpt="google/siglip-base-patch16-224", seg_layers=(2,6,10,-1), embed_dim=256, dropout_rate=0.0):
        super().__init__()
        self.encoder = SiglipVisionModel.from_pretrained(siglip_ckpt)
        hid = self.encoder.config.hidden_size
        
        # 3-class classification head (Real/Synthetic/Tampered) with optional dropout
        if dropout_rate > 0:
            self.cls_head = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(hid, 3)  # 3 classes
            )
        else:
            self.cls_head = nn.Linear(hid, 3)  # 3 classes
            
        self.seg_layers = seg_layers
        self.decoder = SegFormerStrongDecoder([hid]*len(seg_layers), embed_dim=embed_dim, dropout_rate=dropout_rate)

    def forward(self, pixel_values):
        # Enable position embedding interpolation for different image sizes
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(1)
        cls_logit = self.cls_head(pooled).squeeze(1)
        hs = out.hidden_states                           # [emb, h1..hL]
        last = len(hs) - 1
        idxs = [(i+1 if i>=0 else last) for i in self.seg_layers]
        feats = [hs[i] for i in idxs]                    # each (B,N,C)
        N = feats[0].shape[1]; H = int(math.isqrt(N)); W = H
        if H*W != N:
            # Handle non-square grids by finding closest square
            H = W = int(math.sqrt(N))
            if H*W != N:
                raise ValueError(f"Cannot reshape {N} tokens into square grid. Try using square input images.")
        # Pass target size to decoder for proper upsampling
        img_size = int(pixel_values.shape[-1])  # Get actual image size
        seg_logits = self.decoder(feats, (H,W), target_size=img_size)
        return cls_logit, seg_logits

# -----------------------------
# Data
# -----------------------------
def build_transforms(img_size: int):
    train_tf = A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0),
        A.OneOf([A.ImageCompression(quality_range=(75, 95), p=0.5), A.GaussNoise(p=0.5)], p=0.5),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0),
        ToTensorV2(),
    ])
    return train_tf, val_tf

class KorniaAugmentation:
    """GPU-accelerated Kornia augmentations for dual memory optimization"""
    def __init__(self, img_size=224, prob=0.3, enhanced=False, use_clahe=False, 
                 clahe_clip_limit=2.0, clahe_tile_size=(8, 8)):
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        
        if enhanced:
            # Enhanced augmentations for better robustness
            self.augment = KA.AugmentationSequential(
                KA.RandomHorizontalFlip(p=0.5),
                KA.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5),
                KA.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
                KA.RandomGaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0), p=prob * 0.4),
                KA.RandomGaussianNoise(mean=0.0, std=0.05, p=prob * 0.3),
                KA.RandomMotionBlur(kernel_size=5, angle=(-20, 20), direction=(-1, 1), p=prob * 0.3),
                KA.RandomPerspective(distortion_scale=0.1, p=prob * 0.2),
                KA.RandomSharpness(sharpness=2.0, p=prob * 0.2),
                data_keys=["input"],
                random_apply=2,  # Apply 2 random augmentations
            )
        else:
            # Original augmentations
            self.augment = KA.AugmentationSequential(
                KA.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=prob),
                KA.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=prob * 0.3),
                KA.RandomGaussianNoise(mean=0.0, std=0.01, p=prob * 0.2),
                KA.RandomMotionBlur(kernel_size=3, angle=(-15, 15), direction=(-1, 1), p=prob * 0.2),
                data_keys=["input"],
                random_apply=1,
            )
        
    def __call__(self, tensor_batch):
        """Apply Kornia augmentations to batch of tensors on GPU"""
        if tensor_batch.device.type == 'cpu':
            tensor_batch = tensor_batch.cuda()
            
        # Ensure tensor is contiguous for Kornia operations (fixes channels-last compatibility)
        tensor_batch = tensor_batch.contiguous()
            
        # Apply standard augmentations first
        augmented = self.augment(tensor_batch)
        
        # Apply CLAHE if enabled (helps reveal tampering artifacts)
        if self.use_clahe:
            # Ensure contiguous before CLAHE
            augmented = augmented.contiguous()
            augmented = apply_clahe_gpu(
                augmented, 
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_size,
                prob=0.2  # Apply enhancement to 20% of augmented images
            )
            
        return augmented

class SIDSetDS(torch.utils.data.Dataset):
    def __init__(self, hf_ds, tfm):
        self.ds = hf_ds; self.tfm = tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex["image"].convert("RGB")
        lab = ex["label"]
        lbl = lab if isinstance(lab, str) else ["real","fully_synthetic","tampered"][int(lab)]
        
        # 3-class classification: 0=real, 1=fully_synthetic, 2=tampered
        if lbl == "real":
            y_class = 0
            y_binary = 0  # For segmentation (real)
        elif lbl == "fully_synthetic":
            y_class = 1
            y_binary = 1  # For segmentation (fake)
        else:  # "tampered"
            y_class = 2
            y_binary = 1  # For segmentation (fake)

        # mask (dataset provides 'mask')
        m = ex["mask"] if "mask" in ex else None

        img_np = np.array(img)
        if m is not None:
            m_np = np.array(m)
            if m_np.ndim == 3: m_np = m_np[...,0]
            m_np = (m_np > 127).astype(np.uint8)
            
            # Ensure mask matches image dimensions
            if img_np.shape[:2] != m_np.shape[:2]:
                from PIL import Image as PILImage
                m_pil = PILImage.fromarray(m_np).resize((img_np.shape[1], img_np.shape[0]), PILImage.NEAREST)
                m_np = np.array(m_pil)
            
            try:
                out = self.tfm(image=img_np, mask=m_np)
                img_t = out["image"]; mask_t = out["mask"][None].float(); has_mask = 1
            except ValueError as e:
                # Fallback: apply transforms to image only, create zero mask
                print(f"Warning: Mask-image shape mismatch, using zero mask. Error: {e}")
                out = self.tfm(image=img_np)
                img_t = out["image"]; H,W = img_t.shape[1:]
                mask_t = torch.zeros(1,H,W).float(); has_mask = 0
        else:
            out = self.tfm(image=img_np)
            img_t = out["image"]; H,W = img_t.shape[1:]
            mask_t = torch.zeros(1,H,W).float(); has_mask = 0

        return {
            "id": str(ex.get("img_id", f"idx_{i:08d}")),
            "orig": img_np,
            "pixel_values": img_t,
            "y_fake": torch.tensor(y_binary, dtype=torch.float32),  # Binary for segmentation
            "y_class": torch.tensor(y_class, dtype=torch.long),     # 3-class for classification
            "mask": mask_t,
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "label_str": lbl,
        }

def collate_norm(batch, processor):
    imgs = torch.stack([b["pixel_values"] for b in batch], 0) / 255.0
    mean = torch.tensor(processor.image_mean).view(1,3,1,1)
    std  = torch.tensor(processor.image_std).view(1,3,1,1)
    imgs = (imgs - mean) / std
    return {
        "ids": [b["id"] for b in batch],
        "orig": [b["orig"] for b in batch],
        "pixel_values": imgs.float(),
        "y_fake": torch.stack([b["y_fake"] for b in batch], 0).float(),    # Binary for segmentation
        "y_class": torch.stack([b["y_class"] for b in batch], 0).long(),   # 3-class for classification
        "mask": torch.stack([b["mask"] for b in batch], 0).float(),
        "has_mask": torch.stack([b["has_mask"] for b in batch], 0),
        "label_str": [b["label_str"] for b in batch],
    }

def make_loaders(img_size, bs, workers, subset_train=None, subset_val=None, siglip_ckpt="google/siglip2-base-patch16-224"):
    tr_split = subset_train if subset_train else "train"
    va_split = subset_val if subset_val else "validation"
    dtr = load_dataset("saberzl/SID_Set", split=tr_split)
    dva = load_dataset("saberzl/SID_Set", split=va_split)
    proc = SiglipImageProcessor.from_pretrained(siglip_ckpt)
    tf_tr, tf_va = build_transforms(img_size)
    ds_tr = SIDSetDS(dtr, tf_tr); ds_va = SIDSetDS(dva, tf_va)
    coll = lambda b: collate_norm(b, proc)
    # Optimized DataLoaders with dual-memory features
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=workers, 
                       pin_memory=True, persistent_workers=True, prefetch_factor=4, 
                       collate_fn=coll)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=workers, 
                       pin_memory=True, persistent_workers=True, prefetch_factor=4, 
                       collate_fn=coll)
    return dl_tr, dl_va, proc

def get_progressive_img_size(epoch, args):
    """Get image size based on progressive resize schedule with memory limits."""
    if args.no_progressive_resize:
        return min(args.img, args.max_img_size)
    
    prog_epochs = sorted(args.prog_epochs)
    if epoch <= prog_epochs[0]:
        return args.prog_start_size
    elif len(prog_epochs) > 1 and epoch <= prog_epochs[1]:
        mid_size = args.img // 2 if args.img > args.prog_start_size else args.prog_start_size
        return min(mid_size, args.max_img_size)
    else:
        return min(args.img, args.max_img_size)

def get_dynamic_loss_weights(epoch, args):
    """Get dynamic loss weights based on epoch."""
    if args.no_dynamic_loss_weights:
        return args.bce_w, args.dice_w
    
    # Early epochs: focus more on classification, later: balance segmentation
    progress = min(epoch / (args.epochs * 0.7), 1.0)
    dice_w = args.dice_w * (0.3 + 0.7 * progress)  # Gradually increase dice weight
    bce_w = args.bce_w * (1.2 - 0.2 * progress)   # Slightly decrease bce weight
    return bce_w, dice_w

def apply_clahe_gpu(image_tensor, clip_limit=2.0, tile_grid_size=(8, 8), prob=1.0):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on GPU.
    
    Args:
        image_tensor: (B, C, H, W) tensor of normalized images [0,1]
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_grid_size: Grid size for local histogram equalization  
        prob: Probability of applying CLAHE
    
    Returns:
        Enhanced image tensor with improved local contrast
    """
    if torch.rand(1).item() > prob:
        return image_tensor
    
    # Skip CLAHE if not available in this Kornia version
    if not KORNIA_CLAHE_AVAILABLE:
        # Alternative: Simple adaptive histogram equalization using torch operations
        return apply_simple_adaptive_enhancement(image_tensor, clip_limit)
    
    try:
        # Ensure tensor is contiguous and in standard format
        image_tensor = image_tensor.contiguous()
        
        # Try different Kornia CLAHE APIs
        if hasattr(KE, 'Clahe'):
            clahe = KE.Clahe(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
            enhanced = clahe(image_tensor)
        elif hasattr(KE, 'clahe'):
            enhanced = KE.clahe(image_tensor, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        else:
            # Fallback to simple enhancement
            return apply_simple_adaptive_enhancement(image_tensor, clip_limit)
            
        return enhanced.clamp(0, 1)  # Ensure output is in [0,1] range
    except Exception as e:
        # Fallback: simple adaptive enhancement
        return apply_simple_adaptive_enhancement(image_tensor, clip_limit)

def apply_simple_adaptive_enhancement(image_tensor, clip_limit=2.0):
    """Simple GPU-based adaptive enhancement as CLAHE fallback."""
    try:
        # Convert to grayscale for histogram operations
        if image_tensor.size(1) == 3:  # RGB
            gray = 0.299 * image_tensor[:, 0] + 0.587 * image_tensor[:, 1] + 0.114 * image_tensor[:, 2]
            gray = gray.unsqueeze(1)  # Add channel dim back
        else:
            gray = image_tensor
        
        # Simple adaptive enhancement: enhance contrast based on local statistics
        # Apply per-channel enhancement
        enhanced = image_tensor.clone()
        for c in range(image_tensor.size(1)):
            channel = image_tensor[:, c:c+1]
            # Local contrast enhancement
            mean = channel.mean(dim=(2, 3), keepdim=True)
            std = channel.std(dim=(2, 3), keepdim=True) + 1e-8
            # Adaptive normalization with clipping
            normalized = (channel - mean) / std
            clipped = torch.clamp(normalized, -clip_limit, clip_limit)
            enhanced[:, c:c+1] = (clipped * std * 0.5 + mean).clamp(0, 1)
            
        return enhanced
    except Exception:
        # Ultimate fallback: return original
        return image_tensor

# -----------------------------
# Pixel-AUC reservoir sampler (to keep memory bounded)
# -----------------------------
class PixelAUCBuffer:
    def __init__(self, max_pixels:int=400_000, seed:int=42):
        self.max = max_pixels
        self.count = 0
        self.logits = None
        self.targets = None
        random.seed(seed)

    def add_batch(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits, targets shape: (B,1,H,W) for tampered subset
        l = logits.detach().float().cpu().flatten()
        t = targets.detach().float().cpu().flatten()
        if self.logits is None:
            take = min(self.max, l.numel())
            idx = torch.randperm(l.numel())[:take]
            self.logits = l[idx]; self.targets = t[idx]; self.count = take
            return
        # reservoir sampling
        for i in range(l.numel()):
            self.count += 1
            if self.count <= self.max:
                # append until full
                self.logits = torch.cat([self.logits, l[i:i+1]], dim=0)
                self.targets = torch.cat([self.targets, t[i:i+1]], dim=0)
            else:
                j = random.randint(0, self.count-1)
                if j < self.max:
                    self.logits[j]  = l[i]
                    self.targets[j] = t[i]

    def auc(self):
        if self.logits is None or self.targets is None or self.targets.sum()==0:
            return float("nan")
        try:
            return float(roc_auc_score(self.targets.numpy(), torch.sigmoid(self.logits).numpy()))
        except Exception:
            return float("nan")

# -----------------------------
# Train/Eval loop
# -----------------------------
def train_one(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out); ensure_dir(args.overlay_dir)

    # CSV header
    if not os.path.exists(args.metrics_csv):
        with open(args.metrics_csv, "w", newline="") as f:
            header = ["epoch","train_loss","train_cls","train_seg","val_loss","val_cls","val_seg","acc","f1","auc","ap","dice","iou","pix_auc"]
            if not args.no_sweep_mask_thr:
                header.extend(["best_f1_seg","thr_f1","best_dice","thr_dice","best_iou","thr_iou"])
            csv.writer(f).writerow(header)

    # Select appropriate SigLIP model size
    siglip_model = "google/siglip2-base-patch16-224" if args.use_base_siglip else args.siglip_ckpt
    print(f"🔧 Using SigLIP model: {siglip_model}")
    
    dl_tr, dl_va, _ = make_loaders(args.img, args.bs, args.workers, args.subset_train, args.subset_val, siglip_model)
    # Configure decoder size (ultra-large decoder enabled by default for maximum IoU)
    if args.standard_decoder:
        embed_dim = args.embed_dim
        seg_layers = args.seg_layers
        print(f"🔩 Using standard decoder: embed_dim={embed_dim}, layers={seg_layers}")
    elif args.large_decoder:
        embed_dim = 384  # Large decoder
        seg_layers = [2, 4, 6, 8, 10, -1]  # More layers for finer segmentation
        print(f"✨ Using large decoder: embed_dim={embed_dim}, layers={seg_layers}")
    else:
        embed_dim = 512  # Ultra-large for maximum IoU (DEFAULT)
        seg_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1]  # All layers for finest segmentation
        print(f"🚀 Using ULTRA-LARGE decoder (default): embed_dim={embed_dim}, layers={seg_layers}")
        
    model = SigLIP2_MTL(siglip_model, seg_layers=tuple(seg_layers), embed_dim=embed_dim, dropout_rate=args.dropout).to(device)
    if args.dropout > 0:
        print(f"✓ Dropout regularization enabled: {args.dropout}")
    
    # Resume from checkpoint if requested
    start_epoch = 1
    if args.resume or args.resume_ckpt:
        resume_path = args.resume_ckpt if args.resume_ckpt else os.path.join(args.out, "best_siglip2_segformer.pt")
        if os.path.exists(resume_path):
            print(f"🔄 Resuming training from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            
            # Handle both old and new checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]  # Old format
            else:
                raise KeyError("No model state found in checkpoint")
            
            # Handle torch.compile _orig_mod prefixes
            if any(key.startswith("decoder._orig_mod.") for key in state_dict.keys()):
                print("🔄 Detected compiled model checkpoint, removing _orig_mod prefixes...")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("decoder._orig_mod."):
                        new_key = key.replace("decoder._orig_mod.", "decoder.")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict)
            
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"📍 Resuming from epoch {start_epoch}")
            if "metrics" in checkpoint:
                # Restore best metrics for early stopping
                best_metrics = checkpoint["metrics"]
                if "f1" in best_metrics:
                    train_one._best_f1 = best_metrics["f1"]
                    print(f"🎯 Best F1 so far: {train_one._best_f1:.4f}")
        else:
            print(f"⚠️ Checkpoint not found at {resume_path}, starting from scratch")
    
    # Channels-last memory format for CUDA efficiency (enabled by default)
    if not args.no_channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("✓ Channels-last memory format enabled")
    
    # Enable gradient checkpointing for massive VRAM savings (25-35%)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled on encoder")
    
    # Compile decoder for speed optimization (enabled by default)
    if not args.no_compile_decoder:
        try:
            model.decoder = torch.compile(model.decoder, mode="max-autotune")
            print("✓ Decoder compiled with torch.compile")
        except Exception as e:
            print(f"⚠ Failed to compile decoder: {e}")
    
    # Initialize Kornia GPU augmentations (enhanced + CLAHE enabled by default)
    kornia_aug = KorniaAugmentation(
        img_size=args.img, 
        prob=0.3, 
        enhanced=not args.no_enhanced_aug,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=(args.clahe_tile_size, args.clahe_tile_size)
    ) if device.type == 'cuda' else None
    
    if args.clahe:
        if KORNIA_CLAHE_AVAILABLE:
            print(f"✨ CLAHE enabled: clip_limit={args.clahe_clip_limit}, tile_size={args.clahe_tile_size}x{args.clahe_tile_size}")
        else:
            print(f"✨ Simple adaptive enhancement enabled (CLAHE fallback): clip_limit={args.clahe_clip_limit}")
    if not args.no_enhanced_aug:
        print("✨ Enhanced augmentations enabled")
    
    if args.use_enhanced_loss:
        print(f"🎯 Enhanced loss: BCE={args.bce_w}, Focal={args.focal_w}, Dice={args.dice_w}, Boundary={args.boundary_w}, IoU={args.iou_w}, Morph={args.morph_w}")
    else:
        print(f"🔩 Basic BCE+Dice loss: BCE={args.bce_w}, Dice={args.dice_w}")
        
    if args.use_morphological_postprocess:
        print(f"🔷 Morphological post-processing enabled (kernel={args.morph_kernel_size}x{args.morph_kernel_size})")
    else:
        print("🚫 Morphological post-processing disabled")

    # Optional 8-bit optimizer for 40-50% optimizer state memory savings
    try:
        import bitsandbytes as bnb
        OptimClass = bnb.optim.AdamW8bit
        print("✓ Using bitsandbytes AdamW8bit optimizer")
    except ImportError:
        OptimClass = torch.optim.AdamW
        print("✓ Using standard AdamW optimizer (install bitsandbytes for memory savings)")
    
    optim = OptimClass(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = (len(dl_tr) // max(1,args.grad_accum)) * args.epochs
    warm = int(args.warmup * total_steps)
    # Learning rate scheduler selection (cosine annealing by default)
    if args.use_plateau_scheduler:
        sched = ReduceLROnPlateau(optim, 'min', patience=args.plateau_patience, factor=0.5, verbose=True)
        print(f"✓ Using ReduceLROnPlateau scheduler (patience={args.plateau_patience})")
    else:
        sched = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 0.01)
        print("✓ Using CosineAnnealingLR scheduler")
    
    # Load optimizer and scheduler state if resuming (only available in new checkpoints)
    if (args.resume or args.resume_ckpt) and 'checkpoint' in locals():
        if "optimizer_state_dict" in checkpoint:
            try:
                optim.load_state_dict(checkpoint["optimizer_state_dict"])
                print("✓ Optimizer state restored")
            except Exception as e:
                print(f"⚠️ Could not restore optimizer state: {e}")
        if "scheduler_state_dict" in checkpoint:
            try:
                sched.load_state_dict(checkpoint["scheduler_state_dict"])
                print("✓ Scheduler state restored")
            except Exception as e:
                print(f"⚠️ Could not restore scheduler state: {e}")
        
        if "optimizer_state_dict" not in checkpoint:
            print("ℹ️ Old checkpoint format - optimizer/scheduler will restart from default")
    
    # Use BF16 by default if supported, otherwise FP16
    use_bf16 = not args.no_bf16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=not args.no_amp)
    if use_bf16:
        print("✓ Using BF16 precision for training")
    elif not args.no_amp:
        print("✓ Using FP16 precision for training")

    # Early stopping and tracking variables
    best_f1 = -1.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    current_img_size = args.img
    current_grad_accum = args.grad_accum
    epoch_metrics = []  # For visualization
    
    # Memory management setup
    if args.memory_efficient and not args.no_memory_efficient:
        print(f"💾 Memory-efficient training enabled (max_size={args.max_img_size})")
    
    for ep in range(start_epoch, args.epochs+1):
        # Progressive resize logic with memory management
        new_img_size = get_progressive_img_size(ep, args) if not args.no_progressive_resize else min(args.img, args.max_img_size)
        if new_img_size != current_img_size and not args.no_progressive_resize:
            print(f"🔄 Progressive resize: {current_img_size} -> {new_img_size}")
            
            # Memory-efficient batch size adjustment
            if args.memory_efficient and not args.no_memory_efficient:
                if new_img_size >= 512:
                    effective_bs = max(1, args.bs // 4)  # Reduce batch size for 512px+
                    effective_grad_accum = args.grad_accum * 4
                elif new_img_size >= 448:
                    effective_bs = max(2, args.bs // 2)  # Reduce batch size for 448px+
                    effective_grad_accum = args.grad_accum * 2
                else:
                    effective_bs = args.bs
                    effective_grad_accum = args.grad_accum
                print(f"💾 Memory optimization: batch_size={effective_bs}, grad_accum={effective_grad_accum}")
            else:
                effective_bs = args.bs
                effective_grad_accum = args.grad_accum
            
            current_img_size = new_img_size
            dl_tr, dl_va, _ = make_loaders(new_img_size, effective_bs, args.workers, 
                                         args.subset_train, args.subset_val, siglip_model)
            
            # Clear cache before loading new data
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Update Kornia augmentations for new size
            kornia_aug = KorniaAugmentation(
                img_size=new_img_size, 
                prob=0.3, 
                enhanced=not args.no_enhanced_aug,
                use_clahe=args.clahe,
                clahe_clip_limit=args.clahe_clip_limit,
                clahe_tile_size=(args.clahe_tile_size, args.clahe_tile_size)
            ) if device.type == 'cuda' else None
            
            # Update gradient accumulation for memory efficiency
            current_grad_accum = effective_grad_accum
        else:
            current_grad_accum = args.grad_accum
        
        # Dynamic loss weighting (enabled by default)
        current_bce_w, current_dice_w = get_dynamic_loss_weights(ep, args) if not args.no_dynamic_loss_weights else (args.bce_w, args.dice_w)
        
        # Loss weighting configuration (IoU-focused by default for large model)
        current_iou_w = args.iou_w
        if args.balanced_loss:
            print(f"⚖️ Balanced loss weights: maintaining standard ratios")
        else:
            # IoU-focused mode (DEFAULT)
            current_iou_w = min(1.5, args.iou_w * 1.5)  # Boost IoU weight by 50%
            current_bce_w *= 0.7  # Reduce BCE weight
            current_dice_w *= 0.8  # Slightly reduce Dice weight
            print(f"🎯 IoU-focused mode (default): IoU weight boosted to {current_iou_w:.3f}")
        
        if not args.no_dynamic_loss_weights or not args.balanced_loss:
            print(f"⚙️ Loss weights: BCE={current_bce_w:.3f}, Dice={current_dice_w:.3f}, IoU={current_iou_w:.3f}")
        
        # ---- train ----
        model.train()
        tr_loss=tr_cls=tr_seg=0.0; step=0
        pbar = tqdm(dl_tr, ncols=100, desc=f"Epoch {ep:02d} (img={current_img_size})")
        optim.zero_grad(set_to_none=True)
        for batch in pbar:
            pix = batch["pixel_values"].to(device)
            # Apply channels-last format if enabled (default: enabled)
            if not args.no_channels_last and device.type == 'cuda':
                pix = pix.contiguous(memory_format=torch.channels_last)
            y_binary = batch["y_fake"].to(device)    # Binary for segmentation
            y_class  = batch["y_class"].to(device)   # 3-class for classification
            m   = batch["mask"].to(device)
            
            # Apply Kornia GPU augmentations during training
            if kornia_aug is not None and model.training and torch.rand(1).item() < 0.2:
                with torch.no_grad():  # Don't track gradients for augmentation
                    pix = kornia_aug(pix)
            hm  = batch["has_mask"].to(device)
            with torch.amp.autocast('cuda', enabled=not args.no_amp, dtype=amp_dtype):
                cls, seg = model(pix)
                lc = F.cross_entropy(cls, y_class)  # 3-class CrossEntropy loss
                if hm.any():
                    if args.use_enhanced_loss:
                        ls = combined_segmentation_loss(
                            seg[hm], m[hm], 
                            bce_w=current_bce_w, focal_w=args.focal_w, dice_w=current_dice_w,
                            boundary_w=args.boundary_w, iou_w=current_iou_w, morph_w=args.morph_w
                        )
                    else:
                        ls = bce_dice_loss(seg[hm], m[hm], current_bce_w, current_dice_w)
                else:
                    ls = seg.sum()*0.0
                loss = lc + args.lam_seg*ls
            scaler.scale(loss / current_grad_accum).backward()
            tr_loss += loss.item(); tr_cls += lc.item(); tr_seg += ls.item()
            if (step+1) % current_grad_accum == 0:
                # Apply gradient clipping if enabled
                if args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
                if args.use_plateau_scheduler:
                    pass  # ReduceLROnPlateau steps after validation loss
                else:
                    pass  # CosineAnnealingLR steps once per epoch
                # Dual memory optimization - clear cache periodically
                if (step+1) % (current_grad_accum * 10) == 0:
                    torch.cuda.empty_cache()
            step += 1
            pbar.set_postfix(loss=f"{tr_loss/step:.4f}", cls=f"{tr_cls/step:.4f}", seg=f"{tr_seg/step:.4f}")

        # ---- validate ----
        model.eval()
        val_loss=val_cls=val_seg=0.0
        preds=[]; gts=[]
        preds_3class=[]; gts_3class=[]  # For 3-class confusion matrix
        seg_dice=[]; seg_iou=[]
        # For threshold sweeping
        seg_logits_list=[]; masks_list=[] if not args.no_sweep_mask_thr else None
        pix_auc_buf = PixelAUCBuffer(max_pixels=args.pixel_auc_max, seed=args.seed)

        saved_overlays = 0
        ep_dir = os.path.join(args.overlay_dir, f"epoch_{ep:02d}"); ensure_dir(ep_dir)
        
        # For visualization collage
        collage_images, collage_gts, collage_preds, collage_ids = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(dl_va, ncols=100, desc="Val"):
                pix = batch["pixel_values"].to(device)
                # Apply channels-last format if enabled (default: enabled)
                if not args.no_channels_last and device.type == 'cuda':
                    pix = pix.contiguous(memory_format=torch.channels_last)
                y_binary = batch["y_fake"].to(device)    # Binary for segmentation
                y_class  = batch["y_class"].to(device)   # 3-class for classification
                m   = batch["mask"].to(device)
                hm  = batch["has_mask"].to(device)

                # Use autocast for validation with same precision as training
                with torch.amp.autocast('cuda', enabled=not args.no_amp, dtype=amp_dtype):
                    cls, seg = model(pix)
                    
                # Early exit optimization - skip seg processing if cls confidence is low
                if args.early_exit_thresh > 0:
                    # For 3-class: get probability of "real" class (index 0)
                    cls_softmax = torch.softmax(cls, dim=1)
                    real_prob = cls_softmax[:, 0]  # Probability of "real" class
                    skip_seg_mask = real_prob > (1.0 - args.early_exit_thresh)  # High real probability
                    if skip_seg_mask.any():
                        # Zero out segmentation for images likely to be real
                        seg[skip_seg_mask] = seg[skip_seg_mask] * 0
                        
                lc = F.cross_entropy(cls, y_class)  # 3-class CrossEntropy loss
                val_cls += lc.item()

                if hm.any():
                    # seg loss
                    if args.use_enhanced_loss:
                        ls = combined_segmentation_loss(
                            seg[hm], m[hm], 
                            bce_w=current_bce_w, focal_w=args.focal_w, dice_w=current_dice_w,
                            boundary_w=args.boundary_w, iou_w=current_iou_w, morph_w=args.morph_w
                        )
                    else:
                        ls = bce_dice_loss(seg[hm], m[hm], current_bce_w, current_dice_w)
                    val_seg += ls.item()
                    # Collect logits and masks for threshold sweeping
                    if not args.no_sweep_mask_thr:
                        seg_logits_list.append(seg[hm].detach().cpu())
                        masks_list.append(m[hm].detach().cpu())
                    # Apply morphological post-processing if enabled
                    seg_for_eval = seg[hm]
                    if args.use_morphological_postprocess:
                        seg_prob = torch.sigmoid(seg_for_eval)
                        seg_refined = morphological_postprocess(seg_prob, args.morph_kernel_size)
                        # Convert back to logits for evaluation
                        seg_for_eval = torch.log(seg_refined + 1e-8) - torch.log(1 - seg_refined + 1e-8)
                    
                    # Dice/IoU (using current threshold with optional post-processing)
                    d_list, i_list, pbin = dice_iou_from_logits(seg_for_eval, m[hm], thr=args.mask_thr)
                    seg_dice += d_list; seg_iou += i_list
                    # pixel-AUC sampling (logits vs gt)
                    pix_auc_buf.add_batch(seg[hm], m[hm])
                    # overlays
                    if saved_overlays < args.max_overlays:
                        pr_full = (torch.sigmoid(seg) > args.mask_thr).float().cpu().numpy()
                        for i in range(min(pix.size(0), args.max_overlays - saved_overlays)):
                            if not hm[i]: continue
                            gt = m[i,0].cpu().numpy()
                            pr = pr_full[i,0]
                            H,W = gt.shape
                            pr = (cv2.resize(pr, (W,H), interpolation=cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
                            gt = (gt > 0.5).astype(np.uint8)
                            img_np = batch["orig"][i]
                            pr_up = cv2.resize(pr, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                            gt_up = cv2.resize(gt, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                            ov = color_overlay_fp_fn(img_np, gt_up, pr_up, alpha=0.45)
                            stem = f"{batch['ids'][i]}"
                            # Save with WebP format for faster I/O (enabled by default)
                            if not args.no_webp_overlays:
                                Image.fromarray(img_np).save(os.path.join(ep_dir, f"{stem}_img.webp"), method=6, quality=80)
                                Image.fromarray((gt_up*255).astype(np.uint8)).save(os.path.join(ep_dir, f"{stem}_gt.webp"), method=6)
                                Image.fromarray((pr_up*255).astype(np.uint8)).save(os.path.join(ep_dir, f"{stem}_pred.webp"), method=6)
                                Image.fromarray(ov).save(os.path.join(ep_dir, f"{stem}_overlay.webp"), method=6, quality=80)
                            else:
                                Image.fromarray(img_np).save(os.path.join(ep_dir, f"{stem}_img.jpg"), quality=95)
                                Image.fromarray((gt_up*255).astype(np.uint8)).save(os.path.join(ep_dir, f"{stem}_gt.png"))
                                Image.fromarray((pr_up*255).astype(np.uint8)).save(os.path.join(ep_dir, f"{stem}_pred.png"))
                                Image.fromarray(ov).save(os.path.join(ep_dir, f"{stem}_overlay.png"))
                            saved_overlays += 1
                            
                            # Collect data for collage if requested
                            if not args.no_save_plots and len(collage_images) < args.collage_samples:
                                collage_images.append(img_np)
                                collage_gts.append(gt_up)
                                collage_preds.append(pr_up)
                                collage_ids.append(batch['ids'][i])
                                
                            if saved_overlays >= args.max_overlays: break
                else:
                    ls = seg.sum()*0.0
                val_loss += (lc + args.lam_seg*ls).item()

                # For 3-class classification: get class predictions
                cls_preds = torch.argmax(cls, dim=1).cpu().numpy()
                preds_3class.append(cls_preds)
                gts_3class.append(y_class.cpu().numpy())
                
                # For binary metrics: convert 3-class to binary (Real=0, Synthetic/Tampered=1)
                binary_preds = (cls_preds > 0).astype(np.float32)
                binary_gts = y_binary.cpu().numpy()
                preds.append(binary_preds)
                gts.append(binary_gts)

        preds = np.concatenate(preds); gts = np.concatenate(gts)
        preds_3class = np.concatenate(preds_3class); gts_3class = np.concatenate(gts_3class)
        
        # Detection (paper): Acc, F1 (also compute AUC/AP for reference) - Binary metrics
        f1 = f1_score(gts, (preds>0.5).astype(np.uint8))
        acc = accuracy_score(gts, (preds>0.5).astype(np.uint8))
        try: auc = roc_auc_score(gts, preds)
        except: auc = float("nan")
        ap  = average_precision_score(gts, preds)
        
        # Generate both binary and 3-class confusion matrices
        pred_binary = (preds > 0.5).astype(np.uint8)
        create_confusion_matrix(gts, pred_binary, ep, args.overlay_dir)  # Binary
        create_3class_confusion_matrix(gts_3class, preds_3class, ep, args.overlay_dir)  # 3-class
        # Localization (paper): AUC, F1, IoU (tampered only)
        mean_dice = float(np.mean(seg_dice)) if seg_dice else 0.0
        mean_iou  = float(np.mean(seg_iou))  if seg_iou  else 0.0
        pix_auc   = pix_auc_buf.auc()
        
        # Threshold sweeping for optimal metrics (enabled by default)
        best_thr_metrics = None
        if not args.no_sweep_mask_thr and seg_logits_list:
            print(f"\n🔍 Sweeping {args.thr_steps} thresholds from {args.thr_min:.2f} to {args.thr_max:.2f}...")
            best_thr_metrics = sweep_mask_thresholds(
                seg_logits_list, masks_list, 
                args.thr_min, args.thr_max, args.thr_steps
            )
            print(f"✅ Best F1: {best_thr_metrics['f1']:.4f} @ thr={best_thr_metrics['thr_f1']:.3f}")
            print(f"✅ Best Dice: {best_thr_metrics['dice']:.4f} @ thr={best_thr_metrics['thr_dice']:.3f}")
            print(f"✅ Best IoU: {best_thr_metrics['iou']:.4f} @ thr={best_thr_metrics['thr_iou']:.3f}")
            # Update metrics with best swept values
            mean_dice = best_thr_metrics['dice']
            mean_iou = best_thr_metrics['iou']
            
            # Auto-threshold optimization: update args.mask_thr for next epoch (enabled by default)
            if not args.no_auto_threshold:
                # Use IoU-optimized threshold for next epoch (balanced approach)
                args.mask_thr = best_thr_metrics['thr_iou']
                print(f"🎯 Auto-threshold updated to {args.mask_thr:.3f} for next epoch")

        sweep_info = ""
        if not args.no_sweep_mask_thr and best_thr_metrics:
            sweep_info = f" | SweptDice {best_thr_metrics['dice']:.4f}@{best_thr_metrics['thr_dice']:.2f} | SweptIoU {best_thr_metrics['iou']:.4f}@{best_thr_metrics['thr_iou']:.2f}"
        
        print(f"\nEpoch {ep:02d} | "
              f"ValLoss {val_loss/len(dl_va):.4f} | Cls {val_cls/len(dl_va):.4f} | Seg {val_seg/len(dl_va):.4f} | "
              f"Acc {acc:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | AP {ap:.4f} | "
              f"Dice {mean_dice:.4f} | IoU {mean_iou:.4f} | PixAUC {pix_auc:.4f}{sweep_info}\n")

        with open(args.metrics_csv, "a", newline="") as f:
            row = [
                ep,
                round(val_loss/len(dl_va),6), round(val_cls/len(dl_va),6), round(val_seg/len(dl_va),6),
                round(acc,6), round(f1,6), round(float(auc) if not np.isnan(auc) else -1.0,6),
                round(ap,6), round(mean_dice,6), round(mean_iou,6), round(pix_auc if not np.isnan(pix_auc) else -1.0,6)
            ]
            # Always add threshold sweep columns if sweeping is enabled
            if not args.no_sweep_mask_thr:
                if best_thr_metrics:
                    row.extend([
                        round(best_thr_metrics['f1'],6), round(best_thr_metrics['thr_f1'],3),
                        round(best_thr_metrics['dice'],6), round(best_thr_metrics['thr_dice'],3),
                        round(best_thr_metrics['iou'],6), round(best_thr_metrics['thr_iou'],3)
                    ])
                else:
                    row.extend([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])  # Placeholder values
            csv.writer(f).writerow(row)

        # Store metrics for visualization
        epoch_metrics.append({
            'epoch': ep, 'val_loss': val_loss/len(dl_va), 'val_cls': val_cls/len(dl_va), 
            'val_seg': val_seg/len(dl_va), 'acc': acc, 'f1': f1, 'auc': float(auc), 
            'ap': ap, 'dice': mean_dice, 'iou': mean_iou, 'pix_auc': pix_auc
        })
        
        # Learning rate scheduling
        current_val_loss = val_loss/len(dl_va)
        if args.use_plateau_scheduler:
            sched.step(current_val_loss)
        else:
            sched.step()  # CosineAnnealingLR
            
        # Early stopping logic (disabled by default)
        if args.early_stopping:
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"⏰ Early stopping counter: {early_stop_counter}/{args.patience}")
                
            if early_stop_counter >= args.patience:
                print(f"🛑 Early stopping triggered after {ep} epochs (patience={args.patience})")
                break
            
        # Save best by F1 (det)
        if f1 > getattr(train_one, "_best_f1", -1.0):
            train_one._best_f1 = f1
            ckpt = os.path.join(args.out, "best_siglip2_segformer.pt")
            metrics_dict = {"epoch": ep, "acc": acc, "f1": f1, "auc": float(auc), "ap": ap,
                           "dice": mean_dice, "iou": mean_iou, "pix_auc": pix_auc}
            if not args.no_sweep_mask_thr and best_thr_metrics:
                metrics_dict.update({
                    "best_f1_seg": best_thr_metrics['f1'], "thr_f1": best_thr_metrics['thr_f1'],
                    "best_dice": best_thr_metrics['dice'], "thr_dice": best_thr_metrics['thr_dice'],
                    "best_iou": best_thr_metrics['iou'], "thr_iou": best_thr_metrics['thr_iou']
                })
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "epoch": ep,
                "siglip_ckpt": args.siglip_ckpt,
                "seg_layers": args.seg_layers,
                "embed_dim": args.embed_dim,
                "metrics": metrics_dict
            }, ckpt)
            print(f"✅ Saved best to {ckpt} (F1={f1:.4f})")
            
        # Generate visualizations (enabled by default)
        if not args.no_save_plots:
            # Create validation collage
            if collage_images:
                create_overlay_collage(collage_images, collage_gts, collage_preds, 
                                     collage_ids, ep, args.out, args.collage_samples)
                
                # Create IoU confusion matrix and statistics
                create_iou_confusion_matrix(collage_gts, collage_preds, seg_dice, seg_iou, ep, args.overlay_dir)
            
            # Create training curves every epoch for monitoring
            try:
                plot_training_curves(args.metrics_csv, args.out)
            except Exception as e:
                print(f"⚠️ Training curves failed: {e}")

    # Final visualizations and reports (enabled by default)
    if not args.no_save_plots:
        print("📈 Generating final visualizations...")
        create_results_table(args.metrics_csv, args.out)
        plot_training_curves(args.metrics_csv, args.out)
    
    # Final JSON with enhanced metrics
    final_metrics = {
        "best_f1": getattr(train_one, "_best_f1", -1.0),
        "total_epochs": ep,  # Actual epochs completed (may be less due to early stopping)
        "early_stopped": early_stop_counter >= args.patience if args.early_stopping else False,
        "final_val_loss": current_val_loss,
        "config": {
            "dropout": args.dropout,
            "early_stopping": args.early_stopping,
            "progressive_resize": not args.no_progressive_resize,
            "enhanced_aug": not args.no_enhanced_aug
        }
    }
    
    with open(os.path.join(args.out, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"✅ Training completed! Final metrics saved. (Epochs: {ep}/{args.epochs})")
    
    return final_metrics


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    # I/O
    ap.add_argument("--out", type=str, default="./sid_ckpts")
    ap.add_argument("--metrics_csv", type=str, default="./sid_ckpts/epoch_metrics.csv")
    ap.add_argument("--overlay_dir", type=str, default="./sid_ckpts/val_overlays")
    # Performance optimizations (enabled by default)
    ap.add_argument("--no_channels_last", action="store_true", help="Disable channels-last memory format")
    ap.add_argument("--no_compile_decoder", action="store_true", help="Disable decoder compilation")
    ap.add_argument("--no_bf16", action="store_true", help="Disable BF16 precision (use FP16)")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (0 to disable)")
    ap.add_argument("--early_exit_thresh", type=float, default=0.0, help="Skip segmentation if cls prob < threshold (DISABLED by default for 3-class)")
    ap.add_argument("--no_webp_overlays", action="store_true", help="Use PNG instead of WebP for overlays")
    # Progressive training & advanced optimizations (enabled by default)
    ap.add_argument("--no_progressive_resize", action="store_true", help="Disable progressive resize training")
    ap.add_argument("--prog_start_size", type=int, default=320, help="Starting image size for progressive training")
    ap.add_argument("--prog_epochs", type=int, nargs="+", default=[2, 4], help="Epochs to increase resolution")
    ap.add_argument("--max_img_size", type=int, default=448, help="Maximum image size (to prevent VRAM overflow)")
    ap.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient settings for high-res training", default=True)
    ap.add_argument("--no_memory_efficient", action="store_true", help="Disable memory optimizations")
    ap.add_argument("--no_enhanced_aug", action="store_true", help="Disable enhanced Kornia augmentations")
    ap.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement", default=True)
    ap.add_argument("--no_clahe", action="store_true", help="Disable CLAHE contrast enhancement (deprecated, CLAHE now disabled by default)")
    ap.add_argument("--clahe_clip_limit", type=float, default=2.0, help="CLAHE contrast limiting threshold")
    ap.add_argument("--clahe_tile_size", type=int, default=8, help="CLAHE tile grid size (NxN)")
    ap.add_argument("--no_dynamic_loss_weights", action="store_true", help="Disable dynamic loss weighting")
    ap.add_argument("--no_auto_threshold", action="store_true", help="Disable auto-threshold optimization")
    # Regularization and early stopping (enabled by default)
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (disabled by default for better IoU)")
    ap.add_argument("--early_stopping", action="store_true", help="Enable early stopping (disabled by default)")
    ap.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    ap.add_argument("--use_plateau_scheduler", action="store_true", help="Use ReduceLROnPlateau instead of cosine")
    ap.add_argument("--plateau_patience", type=int, default=3, help="Plateau scheduler patience")
    ap.add_argument("--no_save_plots", action="store_true", help="Disable visualization plots generation")
    ap.add_argument("--collage_samples", type=int, default=8, help="Number of samples for validation collage")
    # Data / model
    ap.add_argument("--siglip_ckpt", type=str, default="google/siglip2-large-patch16-384", help="SigLIP model (large for good performance)")
    ap.add_argument("--use_base_siglip", action="store_true", help="Use base SigLIP model instead of giant patch (saves significant memory)")
    ap.add_argument("--seg_layers", type=int, nargs="+", default=[2,6,10,-1])
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--ultra_large_decoder", action="store_true", help="Use ultra-large decoder for maximum IoU (512 dim, all layers)", default=True)
    ap.add_argument("--large_decoder", action="store_true", help="Use large decoder (384 dim, multi-layer)")
    ap.add_argument("--standard_decoder", action="store_true", help="Use standard decoder (256 dim, faster)")
    ap.add_argument("--img", type=int, default=224, help="Smallest resolution for maximum speed")
    ap.add_argument("--bs", type=int, default=12, help="Stable batch size for reliable training")
    ap.add_argument("--workers", type=int, default=8, help="Optimized for multi-core CPUs")
    ap.add_argument("--subset_train", type=str, default=None, help='e.g. "train[:5%%]"')
    ap.add_argument("--subset_val", type=str, default=None, help='e.g. "validation[:10%%]"')
    # Train
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--no_amp", action="store_true")
    # Loss/metrics
    ap.add_argument("--bce_w", type=float, default=0.2, help="Reduced for IoU focus")
    ap.add_argument("--dice_w", type=float, default=0.3, help="Reduced for IoU focus") 
    ap.add_argument("--focal_w", type=float, default=0.0, help="DISABLED - kills IoU")
    ap.add_argument("--boundary_w", type=float, default=0.0, help="DISABLED - kills IoU")
    ap.add_argument("--iou_w", type=float, default=1.5, help="BOOSTED for IoU breakthrough")
    ap.add_argument("--iou_focused", action="store_true", help="Use IoU-focused loss weights for maximum segmentation performance", default=True)
    ap.add_argument("--balanced_loss", action="store_true", help="Use balanced loss weights instead of IoU-focused")
    ap.add_argument("--morph_w", type=float, default=0.0, help="Morphological weight disabled by default (prevents over-smoothing)")
    ap.add_argument("--use_morphological_postprocess", action="store_true", help="Apply morphological post-processing to predictions")
    ap.add_argument("--no_morphological_postprocess", action="store_true", help="Disable morphological post-processing")
    ap.add_argument("--morph_kernel_size", type=int, default=3, help="Morphological operation kernel size")
    ap.add_argument("--lam_seg", type=float, default=1.0)
    ap.add_argument("--use_enhanced_loss", action="store_true", help="Use enhanced multi-component loss (DISABLED by default - kills IoU)")
    ap.add_argument("--no_enhanced_loss", action="store_true", help="Disable enhanced loss, use BCE+Dice only")
    ap.add_argument("--mask_thr", type=float, default=0.3)
    ap.add_argument("--no_sweep_mask_thr", action="store_true", help="Disable mask threshold sweeping", default=True)
    ap.add_argument("--sweep_mask_thr", action="store_true", help="Enable mask threshold sweeping (disabled by default for speed)")
    ap.add_argument("--thr_min", type=float, default=0.1, help="Minimum threshold for sweep")
    ap.add_argument("--thr_max", type=float, default=0.9, help="Maximum threshold for sweep")
    ap.add_argument("--thr_steps", type=int, default=17, help="Number of threshold steps for sweep")
    ap.add_argument("--pixel_auc_max", type=int, default=400000, help="Max pixels sampled for localization AUC (memory cap)")
    ap.add_argument("--max_overlays", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Resume training from the best checkpoint", default=True)
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Specific checkpoint path to resume from")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_one(args)
