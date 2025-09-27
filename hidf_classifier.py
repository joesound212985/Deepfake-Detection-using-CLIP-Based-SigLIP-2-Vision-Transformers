#!/usr/bin/env python3
"""
HIDF Binary Classifier — Fast Training Edition
Real vs Fake image classification using SigLIP vision transformer.

Key optimizations for faster training:
- Default fast training settings (smaller image size, higher lr, fewer epochs)
- Backbone freezing enabled by default (train head only)
- TF32, cuDNN benchmark, and model compilation with max-autotune
- BF16 mixed precision training by default on CUDA
- Automatic class imbalance handling with pos_weight
- GPU-accelerated preprocessing with Kornia
- Optimized DataLoader settings (16 workers, 12x prefetch)
- Evaluation every 2 epochs for ultra-fast training

Usage (ultra-fast training - 20 epochs in ~30 minutes):
  python hidf_classifier.py --data_dir "/path/to/hidf images"

Usage (slower but more thorough):
  python hidf_classifier.py --data_dir "/path/to/hidf images" \
      --freeze_backbone=False --batch_size 128 --lr 1e-4 \
      --eval_every_n_epochs 1 --compile_mode default
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
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import shutil
import glob
import kornia
import kornia.augmentation as K
import math
import torch.nn.functional as F

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

# Fast image loading optimizations
try:
    from turbojpeg import TurboJPEG
    TURBOJPEG_AVAILABLE = True
    turbo_jpeg = TurboJPEG()
except ImportError:
    TURBOJPEG_AVAILABLE = False
    turbo_jpeg = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("Warning: open_clip not available. Install with: pip install open-clip-torch")

# Advanced augmentation imports
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Install with: pip install albumentations")

# Import torch._dynamo at global scope to avoid local assignment issues
try:
    import torch._dynamo
    DYNAMO_AVAILABLE = True
except ImportError:
    DYNAMO_AVAILABLE = False

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples"""
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss with pos_weight
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingBCELoss(nn.Module):
    """Label smoothing for binary classification"""
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(inputs, targets_smooth, pos_weight=self.pos_weight)
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets_smooth)

class CutMixUp(nn.Module):
    """CutMix and MixUp implementation for better generalization"""
    def __init__(self, cutmix_prob=0.5, mixup_prob=0.3, cutmix_alpha=1.0, mixup_alpha=0.4):
        super().__init__()
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
    
    def cutmix(self, images, labels):
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Random permutation
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # Get bounding box coordinates
        h, w = images.size(2), images.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        images_cutmix = images.clone()
        images_cutmix[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return images_cutmix, labels, labels[rand_index], lam
    
    def mixup(self, images, labels):
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Random permutation
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # Apply mixup
        images_mixup = lam * images + (1 - lam) * images[rand_index]
        
        return images_mixup, labels, labels[rand_index], lam
    
    def forward(self, images, labels):
        if not self.training:
            return images, labels
        
        rand = np.random.random()
        
        if rand < self.cutmix_prob:
            return self.cutmix(images, labels)
        elif rand < self.cutmix_prob + self.mixup_prob:
            return self.mixup(images, labels)
        else:
            return images, labels

def fast_image_load(img_path):
    """Fast image loading using turbojpeg or cv2 as fallback"""
    try:
        # Try TurboJPEG first (fastest)
        if TURBOJPEG_AVAILABLE and img_path.lower().endswith(('.jpg', '.jpeg')):
            with open(img_path, 'rb') as f:
                image_array = turbo_jpeg.decode(f.read())
            return Image.fromarray(image_array)
        
        # Try CV2 next (faster than PIL for many formats)
        elif CV2_AVAILABLE:
            image_array = cv2.imread(img_path)
            if image_array is not None:
                # Convert BGR to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image_array)
        
        # Fallback to PIL
        return Image.open(img_path).convert('RGB')
    
    except Exception:
        # Final fallback
        return Image.open(img_path).convert('RGB')

def create_splits(real_dir, fake_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/val/test splits from Real-img/ and Fake-img/ directories"""
    
    print(f"Creating dataset splits...")
    print(f"Real images directory: {real_dir}")
    print(f"Fake images directory: {fake_dir}")
    
    # Get all image files
    real_files = []
    fake_files = []
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    
    # Search for real images (including subdirectories)
    for ext in extensions:
        real_files.extend(glob.glob(os.path.join(real_dir, ext)))
        real_files.extend(glob.glob(os.path.join(real_dir, ext.upper())))
        real_files.extend(glob.glob(os.path.join(real_dir, '**', ext), recursive=True))
        real_files.extend(glob.glob(os.path.join(real_dir, '**', ext.upper()), recursive=True))
    
    # Search for fake images (including subdirectories)
    for ext in extensions:
        fake_files.extend(glob.glob(os.path.join(fake_dir, ext)))
        fake_files.extend(glob.glob(os.path.join(fake_dir, ext.upper())))
        fake_files.extend(glob.glob(os.path.join(fake_dir, '**', ext), recursive=True))
        fake_files.extend(glob.glob(os.path.join(fake_dir, '**', ext.upper()), recursive=True))
    
    print(f"Found {len(real_files)} real images")
    print(f"Found {len(fake_files)} fake images")
    
    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError("No images found in one or both directories!")
    
    # Split real images
    real_train, real_temp = train_test_split(real_files, test_size=(val_ratio + test_ratio), random_state=42)
    real_val, real_test = train_test_split(real_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
    
    # Split fake images
    fake_train, fake_temp = train_test_split(fake_files, test_size=(val_ratio + test_ratio), random_state=42)
    fake_val, fake_test = train_test_split(fake_temp, test_size=test_ratio/(val_ratio + test_ratio), random_state=42)
    
    print(f"Split sizes:")
    print(f"  Train: {len(real_train)} real, {len(fake_train)} fake")
    print(f"  Val: {len(real_val)} real, {len(fake_val)} fake")
    print(f"  Test: {len(real_test)} real, {len(fake_test)} fake")
    
    # Create directory structure
    splits = {'train': (real_train, fake_train), 
              'val': (real_val, fake_val), 
              'test': (real_test, fake_test)}
    
    for split_name, (real_files_split, fake_files_split) in splits.items():
        # Create directories
        split_real_dir = os.path.join(output_dir, split_name.upper(), 'REAL')
        split_fake_dir = os.path.join(output_dir, split_name.upper(), 'FAKE')
        os.makedirs(split_real_dir, exist_ok=True)
        os.makedirs(split_fake_dir, exist_ok=True)
        
        # Copy real files
        for file_path in real_files_split:
            filename = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(split_real_dir, filename))
        
        # Copy fake files
        for file_path in fake_files_split:
            filename = os.path.basename(file_path)
            shutil.copy2(file_path, os.path.join(split_fake_dir, filename))
    
    # Save split info
    split_info = {
        'train_real': len(real_train),
        'train_fake': len(fake_train),
        'val_real': len(real_val),
        'val_fake': len(fake_val),
        'test_real': len(real_test),
        'test_fake': len(fake_test),
        'total_real': len(real_files),
        'total_fake': len(fake_files)
    }
    
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✅ Dataset splits created in: {output_dir}")
    return output_dir

# Model size configurations (OpenCLIP SigLIP models)
MODEL_CONFIGS = {
    'tiny': {
        'model': 'ViT-B-16-SigLIP-256',
        'resolution': 256,
        'params': '86M',
        'description': 'Small SigLIP model, fast'
    },
    'small': {
        'model': 'ViT-B-16-SigLIP-384', 
        'resolution': 384,
        'params': '86M',
        'description': 'Small SigLIP model, higher resolution'
    },
    'medium': {
        'model': 'ViT-L-16-SigLIP-384', 
        'resolution': 384,
        'params': '307M',
        'description': 'Large SigLIP model - good balance (default)'
    },
    'large': {
        'model': 'ViT-SO400M-16-SigLIP2-512',
        'resolution': 512, 
        'params': '400M',
        'description': 'Best SigLIP-2 model for maximum accuracy'
    }
}

class HIDFDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, gpu_transform=None, cache_tensors=False, use_albumentations=False, progressive_resize=None, image_size=256):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.gpu_transform = gpu_transform
        self.cache_tensors = cache_tensors
        self.tensor_cache = {} if cache_tensors else None
        self.use_albumentations = use_albumentations and ALBUMENTATIONS_AVAILABLE
        self.progressive_resize = progressive_resize
        self.image_size = image_size
        
        # Advanced augmentations for training
        if self.use_albumentations and split == 'train':
            self.album_transform = A.Compose([
                # Geometric transforms - essential for fake detection
                A.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.Transpose(p=0.1),
                
                # Geometric distortions - critical for deepfake detection
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                    A.OpticalDistortion(distort_limit=0.1, p=1.0),
                    A.Perspective(scale=(0.05, 0.1), p=1.0),
                ], p=0.4),
                
                # Blur augmentations - helps with compression artifacts
                A.OneOf([
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                    A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=1.0),
                ], p=0.4),
                
                # Noise augmentations - essential for fake detection
                A.OneOf([
                    A.GaussNoise(std_range=(0.1, 0.3), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.3),
                
                # Compression artifacts simulation
                A.OneOf([
                    A.ImageCompression(p=1.0),
                    A.Downscale(p=1.0),
                ], p=0.3),
                
                # Color and contrast augmentations
                A.OneOf([
                    A.CLAHE(clip_limit=(2, 6), p=1.0),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.RandomToneCurve(scale=0.1, p=1.0),
                ], p=0.7),
                
                # Advanced color manipulations
                A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                    A.ChannelShuffle(p=1.0),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                ], p=0.4),
                
                # Dropout and occlusion augmentations
                A.OneOf([
                    A.CoarseDropout(p=1.0),
                    A.GridDropout(ratio=0.5, p=1.0),
                ], p=0.4),
                
                # Advanced augmentations
                A.OneOf([
                    A.Posterize(num_bits=4, p=1.0),
                    A.Equalize(p=1.0),
                    A.Solarize(p=1.0),
                    A.InvertImg(p=1.0),
                ], p=0.2),
                
                # Additional color effects
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
                ], p=0.3),
                
                # Final normalization
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            self.album_transform = None
        
        # Load from REAL/FAKE folders
        real_dir = os.path.join(data_dir, split.upper(), 'REAL')
        fake_dir = os.path.join(data_dir, split.upper(), 'FAKE')
        
        self.samples = []
        self.labels = []
        
        # Load REAL images (label=0)
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    self.samples.append(os.path.join(real_dir, img_file))
                    self.labels.append(0)
        
        # Load FAKE images (label=1) 
        if os.path.exists(fake_dir):
            for img_file in os.listdir(fake_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    self.samples.append(os.path.join(fake_dir, img_file))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.samples)} {split} samples: {sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Check tensor cache first
        if self.cache_tensors and img_path in self.tensor_cache:
            return self.tensor_cache[img_path], label
        
        try:
            # Fast image loading with minimal CPU preprocessing
            image = fast_image_load(img_path)
            
            # Progressive resizing
            if self.progressive_resize and self.split == 'train':
                size = self.progressive_resize
                image = image.resize((size, size), Image.Resampling.LANCZOS)
            
            if self.use_albumentations and self.album_transform:
                # Convert PIL to numpy for albumentations
                image_np = np.array(image)
                augmented = self.album_transform(image=image_np)
                image = augmented['image']
            elif self.transform:
                image = self.transform(image)  # Just ToTensor on CPU
            
            # Cache the tensor if enabled
            if self.cache_tensors:
                self.tensor_cache[img_path] = image
                
            return image, label
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            print(f"Warning: Error loading image {img_path}: {e}")
            # Return a noise image as fallback to avoid learning artifacts
            fallback_image = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
            if self.transform:
                fallback_image = self.transform(fallback_image)
            return fallback_image, label

class LightweightAttention(nn.Module):
    """Lightweight attention mechanism for smaller models"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class FastBinaryClassifier(nn.Module):
    """Fast, lightweight binary classifier for fake image detection"""
    def __init__(self, model_size='medium', device='cuda', 
                 dropout_rate=0.1, use_lightweight_attention=True):
        super().__init__()
        
        self.model_size = model_size
        self.config = MODEL_CONFIGS[model_size]
        self.resolution = self.config['resolution']
        
        if not OPENCLIP_AVAILABLE:
            raise ImportError("open_clip required. Install with: pip install open-clip-torch")
        
        # Use OpenCLIP SigLIP models
        pretrained_name = 'webli' if 'SigLIP2' in self.config['model'] else 'webli'
        self.backbone, _, _ = open_clip.create_model_and_transforms(
            self.config['model'], 
            pretrained=pretrained_name,
            device=device
        )
        
        # Get feature dimension
        if hasattr(self.backbone, 'embed_dim'):
            self.feature_dim = self.backbone.embed_dim
        else:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.resolution, self.resolution, device=device)
                dummy_features = self.backbone.encode_image(dummy_input)
                self.feature_dim = dummy_features.shape[-1]
        self.use_clip = True
        print(f"✅ Using SigLIP model: {self.config['model']}")
        
        # Attention mechanism based on model size
        if use_lightweight_attention and model_size in ['tiny', 'small']:
            self.attention = LightweightAttention(self.feature_dim, num_heads=4)
            self.use_attention = True
        elif model_size == 'large':
            # Enhanced attention for large model
            num_heads = min(8, self.feature_dim // 64)
            self.attention = nn.MultiheadAttention(
                self.feature_dim, num_heads, dropout=dropout_rate, batch_first=True
            )
            self.use_attention = True
        else:
            self.use_attention = False
        
        # Classifier head scaled by model size
        if model_size == 'tiny':
            # Ultra-minimal head for tiny model
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(self.feature_dim, 1)
            )
        elif model_size == 'small':
            # Single hidden layer
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim // 4, 1)
            )
        else:  # medium, large
            # Standard head for medium/large
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.GELU(), 
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(self.feature_dim // 4, 1)
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"🚀 Fast Model: {model_size.upper()} ({self.config['params']})")
        print(f"📐 Resolution: {self.resolution}px, Features: {self.feature_dim}")
        print(f"💡 Description: {self.config['description']}")
        if self.use_attention:
            print(f"🎯 Lightweight attention: Enabled")
        print(f"⚡ Backend: OpenCLIP SigLIP")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        # Resize input if needed
        if x.shape[-1] != self.resolution:
            x = nn.functional.interpolate(x, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        
        # Extract features using SigLIP
        features = self.backbone.encode_image(x)
        
        # Normalize features
        features = features / features.norm(dim=-1, keepdim=True)
        features = self.layer_norm(features)
        
        # Apply attention if enabled
        if self.use_attention:
            if features.dim() == 2:
                features = features.unsqueeze(1)  # Add sequence dimension
            
            if isinstance(self.attention, LightweightAttention):
                features = self.attention(features)
            else:
                # Full multi-head attention
                features, _ = self.attention(features, features, features)
            
            features = features.squeeze(1)  # Remove sequence dimension
        
        if return_features:
            return features
        
        # Classification
        logits = self.classifier(features)
        return logits.squeeze(-1)

# Backward compatibility
BinaryClassifier = FastBinaryClassifier

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, accumulate_grad_batches=1, gpu_transform=None, amp_dtype=torch.float16, cutmix_mixup=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        
        # Apply GPU transforms for massive speedup
        if gpu_transform is not None:
            images = gpu_transform(images)
        
        # Apply CutMix/MixUp
        mixed_loss = False
        if cutmix_mixup is not None:
            result = cutmix_mixup(images, labels.float())
            if len(result) == 4:  # CutMix/MixUp applied
                images, labels_a, labels_b, lam = result
                mixed_loss = True
            else:  # No mixing applied
                images, labels = result
        
        # Mixed precision forward pass
        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(images)
                if mixed_loss:
                    loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels.float())
                loss = loss / accumulate_grad_batches
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            logits = model(images)
            if mixed_loss:
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels.float())
            loss = loss / accumulate_grad_batches
            loss.backward()
            
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        # Statistics
        total_loss += loss.item() * accumulate_grad_batches
        if not mixed_loss:
            predicted = (torch.sigmoid(logits) > 0.5).long()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        else:
            # For mixed samples, use original labels for accuracy calculation
            predicted = (torch.sigmoid(logits) > 0.5).long()
            correct += (lam * (predicted == labels_a.long()).float() + (1 - lam) * (predicted == labels_b.long()).float()).sum().item()
            total += labels_a.size(0)
        
        pbar.set_postfix({
            'Loss': f'{loss.item() * accumulate_grad_batches:.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # Clear cache periodically to prevent OOM
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device, gpu_transform=None, amp_dtype=torch.float16, use_tta=False):
    """Evaluation with optional Test-Time Augmentation"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            
            # Apply GPU transforms
            if gpu_transform is not None:
                images = gpu_transform(images)
            
            if use_tta:
                # Test-Time Augmentation with multiple predictions
                tta_probs = []
                
                # Original image
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(images)
                else:
                    logits = model(images)
                tta_probs.append(torch.sigmoid(logits))
                
                # Horizontal flip
                images_flip = torch.flip(images, dims=[3])
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits_flip = model(images_flip)
                else:
                    logits_flip = model(images_flip)
                tta_probs.append(torch.sigmoid(logits_flip))
                
                # Vertical flip
                images_vflip = torch.flip(images, dims=[2])
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits_vflip = model(images_vflip)
                else:
                    logits_vflip = model(images_vflip)
                tta_probs.append(torch.sigmoid(logits_vflip))
                
                # 90 degree rotation
                images_rot = torch.rot90(images, k=1, dims=[2, 3])
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits_rot = model(images_rot)
                else:
                    logits_rot = model(images_rot)
                tta_probs.append(torch.sigmoid(logits_rot))
                
                # Average all TTA predictions
                probs = torch.stack(tta_probs).mean(dim=0)
                
                # Calculate loss with original prediction for consistency
                loss = criterion(logits, labels.float())
            else:
                # Standard evaluation
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        logits = model(images)
                        loss = criterion(logits, labels.float())
                else:
                    logits = model(images)
                    loss = criterion(logits, labels.float())
                
                probs = torch.sigmoid(logits)
            
            total_loss += loss.item()
            
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            
            # Clear cache occasionally during evaluation
            if len(all_labels) % 10 == 0:
                torch.cuda.empty_cache()
    
    # Calculate metrics - convert BF16 to float32 for numpy compatibility
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).float().numpy()
    all_preds = (all_probs > 0.5).astype(int)
    
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)  # Average Precision
    mcc = matthews_corrcoef(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, balanced_acc, precision, recall, f1, auc, ap, mcc, cm, all_labels, all_probs

def save_plots(labels, probs, save_dir, epoch=None):
    """Save separate publication-quality plots"""
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    
    # Set publication style
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
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Real', 'Fake'])
    plt.yticks([0.5, 1.5], ['Real', 'Fake'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix - Raw
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                cbar_kws={'label': 'Count'})
    plt.title('Raw Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0.5, 1.5], ['Real', 'Fake'])
    plt.yticks([0.5, 1.5], ['Real', 'Fake'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, linewidth=3, label=f'PR Curve (AP = {ap:.3f})')
    plt.axhline(y=labels.mean(), color='k', linestyle='--', linewidth=2, 
                label=f'Random (AP = {labels.mean():.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Score Distribution
    plt.figure(figsize=(10, 6))
    real_scores = probs[labels == 0]
    fake_scores = probs[labels == 1]
    plt.hist(real_scores, bins=50, alpha=0.7, label=f'Real (n={len(real_scores)})', 
             color='blue', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.7, label=f'Fake (n={len(fake_scores)})', 
             color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Classification Score')
    plt.ylabel('Density')
    plt.title('Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}score_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Performance Metrics Bar Chart
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
    plt.title('Performance Metrics')
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
    
    print(f"Saved separate plots to {save_dir}/ with prefix '{prefix}'")

def save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, save_dir):
    """Save separate training curves"""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12
    })
    
    epochs = range(1, len(train_losses) + 1)
    
    # 1. Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', linewidth=3, label='Training Loss', marker='o', markersize=6)
    plt.plot(epochs, val_losses, 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy Curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc/100 for acc in train_accs], 'b-', linewidth=3, label='Training Accuracy', marker='o', markersize=6)
    plt.plot(epochs, [acc*100 for acc in val_accs], 'r-', linewidth=3, label='Validation Accuracy', marker='s', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. F1 Score Curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1s, 'g-', linewidth=3, label='Validation F1 Score', marker='^', markersize=6)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved separate training curves to {save_dir}/")

def main():
    parser = argparse.ArgumentParser(description='HIDF Binary Classifier')
    parser.add_argument('--real_dir', type=str, default='/mnt/c/Users/admin/Desktop/Real-img/', help='Path to real images directory')
    parser.add_argument('--fake_dir', type=str, default='/mnt/c/Users/admin/Desktop/Fake-img/', help='Path to fake images directory')
    parser.add_argument('--data_dir', type=str, default='/mnt/c/Users/admin/Desktop/hidf images/', help='Output directory for processed dataset')
    parser.add_argument('--model_size', type=str, default='medium', 
                       choices=['tiny', 'small', 'medium', 'large'], 
                       help='Model size: tiny (B-16-256), small (B-16-384), medium (L-16-384, default), large (SO400M SigLIP-2)')
    parser.add_argument('--use_albumentations', action='store_true', default=True, help='Use advanced albumentations augmentation')
    parser.add_argument('--progressive_resize', action='store_true', default=True, help='Use progressive resizing')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (minimal for maximum gradient accumulation)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./hidf_checkpoints', help='Save directory')
    parser.add_argument('--skip_split', action='store_true', help='Skip data splitting (use existing splits)')
    parser.add_argument('--split_only', action='store_true', help='Only create data splits, don\'t train')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    parser.add_argument('--accumulate_grad_batches', type=int, default=16, help='Gradient accumulation steps (maximum for ultra-stable gradients)')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs for learning rate')
    parser.add_argument('--eval_every_n_epochs', type=int, default=2, help='Evaluate every N epochs for speed')
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help='Train full model for better accuracy (default)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead', choices=['default','reduce-overhead','max-autotune'], help='torch.compile mode')
    # Image size is now determined by model_size
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['fp16','bf16'], help='AMP precision on CUDA')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers (reduced for memory)')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor (reduced for memory)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set AMP dtype
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' and device.type == 'cuda' else torch.float16
    print(f"Using AMP dtype: {amp_dtype}")
    
    # Print speed optimization status
    optimizations = []
    if device.type == 'cuda':
        optimizations.append("✅ GPU-accelerated preprocessing with kornia")
        optimizations.append(f"✅ {args.amp_dtype.upper()} mixed precision training")
        optimizations.append("✅ TF32 enabled for faster matmul")
    if TURBOJPEG_AVAILABLE:
        optimizations.append("✅ TurboJPEG fast image decoding")
    elif CV2_AVAILABLE:
        optimizations.append("✅ OpenCV fast image decoding")
    
    if optimizations:
        print("🚀 Speed optimizations enabled:")
        for opt in optimizations:
            print(f"   {opt}")
    else:
        print("⚠️  No speed optimizations available")
    
    # Use existing splits
    print(f"Using existing splits in: {args.data_dir}")
    
    # Check if splits exist
    if not os.path.exists(os.path.join(args.data_dir, 'TRAIN')):
        print(f"⚠️ No splits found in {args.data_dir}")
        print(f"Run: python create_splits.py")
        return
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model_size]
    target_resolution = config['resolution']
    
    print(f"\n📊 Model Configuration:")
    print(f"   Model: {config['model']}")
    print(f"   Size: {args.model_size.upper()} ({config['params']})")
    print(f"   Resolution: {config['resolution']}px")
    print(f"   Backend: OpenCLIP SigLIP")
    print(f"   Description: {config['description']}")
    
    # GPU-accelerated transforms for faster preprocessing
    if device.type == 'cuda':
        # GPU transforms using kornia for maximum speed
        gpu_transform = nn.Sequential(
            K.Resize(target_resolution, antialias=True),
            K.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ).to(device)
        
        # CPU transforms (resize on CPU to ensure consistent tensor sizes)
        cpu_transform = transforms.Compose([
            transforms.Resize((target_resolution, target_resolution), antialias=True),
            transforms.ToTensor()
        ])
        
        print(f"🚀 Using GPU-accelerated preprocessing with kornia @ {target_resolution}px")
    else:
        # Fallback CPU transforms
        cpu_transform = transforms.Compose([
            transforms.Resize((target_resolution, target_resolution), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        gpu_transform = None
        print(f"Using CPU preprocessing @ {target_resolution}px (GPU not available)")
    
    # Datasets with GPU transforms and caching (disable caching for larger batches)
    use_cache = args.batch_size <= 64  # Only cache for smaller batch sizes to avoid OOM
    # Progressive resize function
    def progressive_resize_schedule(epoch, max_epochs, model_size='medium'):
        config = MODEL_CONFIGS[model_size]
        target_res = config['resolution']
        
        if model_size == 'tiny':
            if epoch < max_epochs * 0.5:
                return max(128, target_res - 96)
            else:
                return target_res
        elif model_size == 'small':
            if epoch < max_epochs * 0.3:
                return max(160, target_res - 128)
            elif epoch < max_epochs * 0.6:
                return max(224, target_res - 64)
            else:
                return target_res
        else:
            if epoch < max_epochs * 0.3:
                return 224
            elif epoch < max_epochs * 0.6:
                return max(288, target_res - 96)
            else:
                return target_res
    
    train_dataset = HIDFDataset(
        args.data_dir, 'train', cpu_transform, gpu_transform, 
        cache_tensors=False,
        use_albumentations=args.use_albumentations,
        progressive_resize=progressive_resize_schedule(0, args.epochs, args.model_size) if args.progressive_resize else None,
        image_size=target_resolution
    )
    val_dataset = HIDFDataset(args.data_dir, 'val', cpu_transform, gpu_transform, cache_tensors=False, image_size=target_resolution) 
    test_dataset = HIDFDataset(args.data_dir, 'test', cpu_transform, gpu_transform, cache_tensors=False, image_size=target_resolution)
    
    # Memory-optimized data loaders
    effective_batch_size = args.batch_size
    if args.model_size == 'large':
        # Reduce batch size for large model
        effective_batch_size = max(args.batch_size // 2, 1)
        print(f"⚠️ Large model detected: reducing batch size to {effective_batch_size}")
    
    print(f"Effective batch size: {effective_batch_size} (with {args.accumulate_grad_batches}x accumulation = {effective_batch_size * args.accumulate_grad_batches} total)")
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True, 
                              prefetch_factor=args.prefetch_factor, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True, 
                            prefetch_factor=args.prefetch_factor)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=True, 
                             prefetch_factor=args.prefetch_factor)
    
    # SigLIP-based classifier
    model = FastBinaryClassifier(
        model_size=args.model_size,
        device=device,
        dropout_rate=args.dropout_rate,
        use_lightweight_attention=args.model_size in ['tiny', 'small']
    ).to(device)
    
    # Freeze backbone for faster training if enabled
    if args.freeze_backbone:
        for param in model.backbone.parameters():
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
            # Add fallback to suppress compilation errors if needed
            if DYNAMO_AVAILABLE:
                torch._dynamo.config.suppress_errors = True
    
    # Convert model to channels_last for faster training
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    
    # Advanced loss with automatic pos_weight for class imbalance
    neg_count = sum(1 for label in train_dataset.labels if label == 0)
    pos_count = sum(1 for label in train_dataset.labels if label == 1)
    pos_weight_val = neg_count / max(1, pos_count) if pos_count > 0 else 1.0
    pos_weight = torch.tensor(pos_weight_val, device=device)
    
    # Use Focal Loss for better handling of hard examples and class imbalance
    criterion = FocalLoss(alpha=0.75, gamma=2.0, pos_weight=pos_weight)
    print(f"📊 Using Focal Loss with pos_weight = {pos_weight_val:.4f} (neg={neg_count}, pos={pos_count})")
    
    # Add CutMix/MixUp for better generalization
    cutmix_mixup = CutMixUp(cutmix_prob=0.5, mixup_prob=0.3, cutmix_alpha=1.0, mixup_alpha=0.4).to(device)
    print(f"🎯 CutMix/MixUp enabled for better generalization")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Advanced learning rate scheduler with cosine annealing and restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.1, last_epoch=-1
    )
    print(f"📈 Using Cosine Annealing with Warm Restarts scheduler")
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not args.evaluate_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    if args.evaluate_only:
        # Evaluate only
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
        save_plots(test_labels, test_probs, args.save_dir)
        return
    
    # Training loop
    best_f1 = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        # Update progressive resize for current epoch
        if args.progressive_resize:
            new_size = progressive_resize_schedule(epoch, args.epochs, args.model_size)
            train_dataset.progressive_resize = new_size
            print(f"Progressive resize: {new_size}px")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, args.accumulate_grad_batches, gpu_transform, amp_dtype, cutmix_mixup)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate on validation set (every N epochs for speed, plus first/last epochs)
        should_eval = (epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % args.eval_every_n_epochs == 0)
        
        if should_eval:
            # Use TTA for final test evaluation only to save time during training
            use_tta_eval = (epoch == args.epochs - 1)
            val_loss, val_acc, val_balanced_acc, val_precision, val_recall, val_f1, val_auc, val_ap, val_mcc, val_cm, val_labels, val_probs = evaluate(model, val_loader, criterion, device, gpu_transform, amp_dtype, use_tta_eval)
        else:
            # Skip expensive validation, use previous values
            val_loss = val_losses[-1] if val_losses else 0.5
            val_acc = val_accs[-1] if val_accs else 0.5
            val_f1 = val_f1s[-1] if val_f1s else 0.5
            val_balanced_acc = val_precision = val_recall = val_auc = val_ap = val_mcc = 0.0
            
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        scheduler.step()
        
        # Clear memory after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                torch.save(checkpoint, os.path.join(args.save_dir, 'best_hidf_binary.pth'))
                print(f"🎯 New best F1: {val_f1:.4f} - model saved")
                save_plots(val_labels, val_probs, args.save_dir, epoch)
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs (patience: {args.early_stopping_patience})")
                    break
        else:
            print(f"Val (cached): Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            patience_counter += 1  # Increment patience when skipping eval
    
    # Final test evaluation with TTA for maximum accuracy
    print(f"\n🧪 Final Test Evaluation with Test-Time Augmentation:")
    test_loss, test_acc, test_balanced_acc, test_precision, test_recall, test_f1, test_auc, test_ap, test_mcc, test_cm, test_labels, test_probs = evaluate(model, test_loader, criterion, device, gpu_transform, amp_dtype, use_tta=True)
    print(f"Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
    
    # Save final training curves and test plots
    save_training_curves(train_losses, train_accs, val_losses, val_accs, val_f1s, args.save_dir)
    save_plots(test_labels, test_probs, args.save_dir, "final_test")

if __name__ == '__main__':
    main()