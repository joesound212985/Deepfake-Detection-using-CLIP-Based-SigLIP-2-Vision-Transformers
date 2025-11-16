#!/usr/bin/env python3
"""
Train the *new* upgraded FreqMLP on 24-dim FFT+SRM features.

- Uses the same type of handcrafted features you already use
- Upgraded FreqMLP architecture:
    - FeatureNormalizer
    - ContrastScaler
    - BandGating (4 bands)
    - 2x ResidualMLPBlock
    - TemperatureScaler
- Saves freq_mlp.safetensors (for use in your v5 app.py)

Usage example:

python train_freq_mlp_v5.py \
  --real-dir "/mnt/c/Users/admin/Desktop/Real-img/Real-img" \
  --fake-dir "/mnt/c/Users/admin/Desktop/Fake-img/Image" \
  --limit 8000 \
  --epochs 80 \
  --batch-size 8 \
  --lr 1e-3 \
  --freq-out "freq_mlp.safetensors"
"""

import argparse
import math
import os
import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFile
from safetensors.torch import save_file
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# CONSTANTS / DEFAULT PATHS
# ---------------------------------------------------------------------------

IMG_EXTS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif", ".jfif", ".heic", ".heif",
)
EPS = 1e-8

DEVICE = os.environ.get("FREQ_MLP_DEVICE", "cpu").lower()
if DEVICE not in {"cpu", "cuda"}:
    DEVICE = "cpu"

DEFAULT_REAL_DIR = "/mnt/c/Users/admin/Desktop/Real-img/Real-img"
DEFAULT_FAKE_DIR = "/mnt/c/Users/admin/Desktop/Fake-img/Image"
DEFAULT_SAMPLE   = 1000  # samples per class (0 = use all)

# ---------------------------------------------------------------------------
# FFT + SRM HANDCRAFTED FEATURES (24-D)
# ---------------------------------------------------------------------------

SRM_K = [
    torch.tensor(
        [
            [0,0,0,0,0],
            [0,-1,2,-1,0],
            [0, 2,-4, 2,0],
            [0,-1,2,-1,0],
            [0,0,0,0,0],
        ], dtype=torch.float32
    ),
    torch.tensor([[-1,2,-1],[ 2,-4,2],[-1,2,-1]], dtype=torch.float32),
    torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32),
]

def _pil_to_gray256_clahe(pil: Image.Image) -> torch.Tensor:
    """Grayscale + CLAHE + resize to 256x256 (float32 [0,1])."""
    g = ImageOps.exif_transpose(pil).convert("L")
    arr = np.array(g, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    arr = clahe.apply(arr)
    g = Image.fromarray(arr).resize((256,256), Image.BICUBIC)
    return torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)

def fft_features(pil: Image.Image) -> List[float]:
    x = _pil_to_gray256_clahe(pil)
    F = torch.fft.fftshift(torch.fft.fft2(x))
    F_mag = torch.abs(F)
    F_phase = torch.angle(F)

    h, w = F_mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = float(r.max())

    # radial bands
    r1, r2 = 0.15 * rmax, 0.45 * rmax
    Et = float(F_mag.sum().item()) + EPS
    El = float(F_mag[r <= r1].sum().item())
    Em = float(F_mag[(r > r1) & (r <= r2)].sum().item())
    Eh = float(F_mag[r > r2].sum().item())

    # log-spectrum slope
    rb = torch.logspace(math.log10(1.0), math.log10(rmax + 1.0), 40)
    ridx = torch.bucketize(r.flatten() + 1.0, rb) - 1

    mu = []
    flatF = F_mag.flatten()
    for i in range(len(rb) - 1):
        mask = (ridx == i)
        if mask.any():
            mu.append(float(torch.log(flatF[mask] + 1e-6).mean().item()))
        else:
            mu.append(0.0)

    xs = np.arange(len(mu))
    ys = np.nan_to_num(mu)
    slope = float(np.polyfit(xs, ys, 1)[0])

    # phase entropy
    phase_hist = torch.histc(F_phase.flatten(), bins=50, min=-math.pi, max=math.pi)
    phase_prob = phase_hist / (phase_hist.sum() + EPS)
    phase_entropy = float(-(phase_prob * torch.log(phase_prob + EPS)).sum().item())

    # directional anisotropy
    ang = torch.atan2(yy - cy, xx - cx)
    sect_means = []
    for a0 in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        mask = (ang >= a0) & (ang < a0 + math.pi/4)
        if mask.any():
            sect_means.append(float(F_mag[mask].mean().item()))
        else:
            sect_means.append(0.0)
    anis = float(np.var(sect_means))

    # wavelets (db1, 2 levels)
    cA1, (cH1, cV1, cD1) = pywt.dwt2(_pil_to_gray256_clahe(pil).numpy(), "db1")
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, "db1")
    wave = [
        np.mean(np.abs(c)**2)
        for c in [cA1, cH1, cV1, cD1, cA2, cH2, cV2, cD2]
    ]

    feats = [
        El / Et,
        Em / Et,
        Eh / Et,
        (Eh + EPS) / (El + EPS),
        slope,
        anis,
        phase_entropy,
    ] + wave
    return feats

def srm_features(pil: Image.Image) -> List[float]:
    x = _pil_to_gray256_clahe(pil)[None, None, ...]
    feats: List[float] = []
    for k2d in SRM_K:
        k = (k2d / (k2d.abs().sum() + EPS)).view(1, 1, *k2d.shape)
        y = nn.functional.conv2d(x, k, padding=k2d.shape[-1] // 2)
        arr = y.flatten().numpy()
        mean = float(arr.mean())
        var  = float(arr.var())
        kurt = float(((arr - mean)**4).mean() / ((var + EPS)**2))
        feats += [mean, var, kurt]
    return feats

def extract_freq_vector(pil: Image.Image) -> torch.Tensor:
    feats = fft_features(pil) + srm_features(pil)
    return torch.tensor(feats, dtype=torch.float32)  # [24]

# ---------------------------------------------------------------------------
# DATASET UTILITIES
# ---------------------------------------------------------------------------

def list_images(folder: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, name))
    return sorted(out)

def _limit_paths(paths: List[str], limit: int, rng: random.Random) -> List[str]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    return rng.sample(paths, limit)

def prepare_paths(real_dir: str, fake_dir: str, limit: int, seed: int) -> Tuple[List[str], List[str]]:
    if not real_dir or not fake_dir:
        raise SystemExit("Please provide both --real-dir and --fake-dir (or set REAL_DIR / FAKE_DIR).")
    real_paths = list_images(real_dir)
    fake_paths = list_images(fake_dir)
    if not real_paths or not fake_paths:
        raise SystemExit("No images found under the provided directories.")
    rng = random.Random(seed)
    real_paths = _limit_paths(real_paths, limit, rng)
    fake_paths = _limit_paths(fake_paths, limit, rng)
    print(f"[data] using {len(real_paths)} real and {len(fake_paths)} fake samples")
    return real_paths, fake_paths

def extract_freq_matrix(paths: Sequence[str]) -> torch.Tensor:
    feats: List[List[float]] = []
    for p in tqdm(paths, desc="freq", leave=False):
        with Image.open(p) as pil:
            feats.append(extract_freq_vector(pil.convert("RGB")).tolist())
    return torch.tensor(feats, dtype=torch.float32)

# ---------------------------------------------------------------------------
# NEW FreqMLP ARCHITECTURE
# ---------------------------------------------------------------------------

class FeatureNormalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std",  torch.ones(dim))
    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0)
        self.std  = x.std(dim=0) + 1e-6
    def forward(self, x: torch.Tensor):
        return (x - self.mean) / (self.std + 1e-6)

class ContrastScaler(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
    def forward(self, x: torch.Tensor):
        return torch.tanh(self.alpha * x + self.beta)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(1.0))
    def forward(self, logits: torch.Tensor):
        return logits / (self.T + 1e-6)

class BandGating(nn.Module):
    def __init__(self, dim: int, num_bands: int = 4):
        super().__init__()
        assert dim % num_bands == 0
        self.band_dim  = dim // num_bands
        self.num_bands = num_bands
        self.gates     = nn.Parameter(torch.zeros(num_bands))
    def forward(self, x: torch.Tensor):
        chunks = torch.split(x, self.band_dim, dim=-1)
        gates  = torch.sigmoid(self.gates)
        return torch.cat([c * gates[i] for i, c in enumerate(chunks)], dim=-1)

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor):
        r = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + r

class FreqMLP(nn.Module):
    """
    Upgraded frequency head:
      - FeatureNormalizer
      - ContrastScaler
      - BandGating
      - 2x ResidualMLPBlock
      - Linear head + TemperatureScaler
    """
    def __init__(self, dim: int = 24, hidden: int = 64, num_bands: int = 4):
        super().__init__()
        self.normer   = FeatureNormalizer(dim)
        self.contrast = ContrastScaler(dim)
        self.band     = BandGating(dim, num_bands)
        self.blocks   = nn.ModuleList([
            ResidualMLPBlock(dim, hidden),
            ResidualMLPBlock(dim, hidden),
        ])
        self.head = nn.Linear(dim, 1)
        self.temp = TemperatureScaler()

    def fit_normalization(self, feats: torch.Tensor):
        self.normer.fit(feats)

    def forward(self, x: torch.Tensor):
        x = self.normer(x)
        x = self.contrast(x)
        x = self.band(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x).squeeze(-1)
        return self.temp(logits)

# ---------------------------------------------------------------------------
# TRAINING HELPERS
# ---------------------------------------------------------------------------

def _safe_auc(target: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(target, scores))
    except ValueError:
        return float("nan")

def _make_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------------------------------------------------------------------
# MAIN TRAIN LOOP (FreqMLP ONLY)
# ---------------------------------------------------------------------------

def train_freq_mlp(
    real_paths: Sequence[str],
    fake_paths: Sequence[str],
    epochs: int,
    batch_size: int,
    lr: float,
    save_path: str,
) -> None:
    print(f"[freq] extracting features for {len(real_paths)} real / {len(fake_paths)} fake images")
    real_feats = extract_freq_matrix(real_paths)
    fake_feats = extract_freq_matrix(fake_paths)

    features = torch.cat([real_feats, fake_feats], dim=0)
    labels = torch.cat(
        [
            torch.zeros(len(real_feats), dtype=torch.float32),
            torch.ones(len(fake_feats), dtype=torch.float32),
        ],
        dim=0,
    )

    model = FreqMLP().to(DEVICE)
    # Important: fit normalization on all features (or just train split if you add a split)
    model.fit_normalization(features)

    loader = _make_loader(features, labels, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            logits = model(features.to(DEVICE)).cpu()
            probs = torch.sigmoid(logits).numpy()
            auc = _safe_auc(labels.numpy(), probs)
            acc = float(((probs >= 0.5) == labels.numpy()).mean())

        print(
            f"[freq] epoch {epoch:03d}/{epochs} "
            f"loss={np.mean(epoch_losses):.4f} acc={acc:.3f} auc={auc:.3f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None and save_path:
        save_file(best_state, save_path)
        print(f"[freq] saved best model to {save_path} (AUC={best_auc:.3f})")
    else:
        print("[freq] WARNING: no best state saved")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train upgraded FreqMLP (24-d FFT+SRM) for Deepfake Detection v5"
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=os.environ.get("REAL_DIR", DEFAULT_REAL_DIR),
        help="Folder that contains REAL images.",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        default=os.environ.get("FAKE_DIR", DEFAULT_FAKE_DIR),
        help="Folder that contains FAKE images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("SAMPLE", DEFAULT_SAMPLE)),
        help="Samples per class (0 = use all).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--freq-out",
        type=str,
        default="freq_mlp.safetensors",
        help="Output path for the freq MLP weights.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    real_paths, fake_paths = prepare_paths(args.real_dir, args.fake_dir, args.limit, args.seed)

    train_freq_mlp(
        real_paths=real_paths,
        fake_paths=fake_paths,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.freq_out,
    )
    print(f"Done!  {args.freq_out}")

if __name__ == "__main__":
    main()
