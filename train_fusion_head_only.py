#!/usr/bin/env python3
"""
Train ONLY the AdaptiveFusionHead for v5.0 Deepfake Detector.

Requires:
  - best_model.safetensors  (SigLIP + classifier)
  - freq_mlp.safetensors    (trained FreqMLP, v5 architecture)

Outputs:
  - fusion_head.safetensors (trained AdaptiveFusionHead)

Usage (example):

python train_fusion_head_only.py ^
  --real-dir "C:\\Users\\admin\\Desktop\\Real-img\\Real-img" ^
  --fake-dir "C:\\Users\\admin\\Desktop\\Fake-img\\Image" ^
  --best-model "C:\\Users\\admin\\Desktop\\best_model.safetensors" ^
  --freq-mlp "C:\\Users\\admin\\Desktop\\freq_mlp.safetensors" ^
  --fusion-out "C:\\Users\\admin\\Desktop\\fusion_head.safetensors"
"""

import argparse
import math
import os
from typing import List, Sequence

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFile
from safetensors.torch import load_file, save_file
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torchvision import transforms
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)

IMG_EXTS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif", ".jfif", ".heic", ".heif"
)

IMG_SIZE   = 384
SIGLIP_DIM = 1024
EPS        = 1e-8

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
FREQ_DEVICE  = os.environ.get("FREQ_MLP_DEVICE", "cpu").lower()
if FREQ_DEVICE not in {"cpu", "cuda"}:
    FREQ_DEVICE = "cpu"

# -------------------- Preprocess (matches v5 app) -------------------- #

def apply_clahe(pil_img: Image.Image) -> Image.Image:
    arr = np.array(pil_img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for chan in range(3):
        arr[:, :, chan] = clahe.apply(arr[:, :, chan])
    return Image.fromarray(arr)

preprocess = transforms.Compose(
    [
        transforms.Lambda(apply_clahe),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

# -------------------- SigLIP Binary Classifier (v5 app-style) -------------------- #

class BinaryClassifier(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.backbone, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP-384", pretrained="webli", device=device
        )
        self.se = nn.Sequential(
            nn.Linear(SIGLIP_DIM, SIGLIP_DIM // 16),
            nn.ReLU(),
            nn.Linear(SIGLIP_DIM // 16, SIGLIP_DIM),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(SIGLIP_DIM),
            nn.Dropout(0.3),
            nn.Linear(SIGLIP_DIM, SIGLIP_DIM // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(SIGLIP_DIM // 2, SIGLIP_DIM // 4),
            nn.GELU(),
            nn.Linear(SIGLIP_DIM // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if x.shape[-1] != IMG_SIZE:
                x = nn.functional.interpolate(x, size=(IMG_SIZE, IMG_SIZE))
            f = self.backbone.encode_image(x)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        se = self.se(f)
        z  = self.classifier(f * se).squeeze(-1)
        return z

def _filter_state_for_model(state, model):
    msd = model.state_dict()
    return {
        k: v for k, v in state.items()
        if (not k.startswith("backbone.text.")) and (k in msd) and (v.shape == msd[k].shape)
    }

def load_siglip_from_best(best_path: str) -> BinaryClassifier:
    model = BinaryClassifier(DEVICE).to(DEVICE).eval()
    state = load_file(best_path)
    filt  = _filter_state_for_model(state, model)
    model.load_state_dict(filt, strict=False)
    print(f"[siglip] Loaded weights from: {best_path}")
    return model

# -------------------- Frequency feature extractor (v5) -------------------- #

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
    gray = ImageOps.exif_transpose(pil).convert("L")
    arr  = np.array(gray, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    gray = Image.fromarray(arr).resize((256, 256), Image.BICUBIC)
    return torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)

def fft_features(pil: Image.Image):
    x = _pil_to_gray256_clahe(pil)
    fft_c = torch.fft.fft2(x)
    f_mag = torch.abs(torch.fft.fftshift(fft_c))
    f_phase = torch.angle(torch.fft.fftshift(fft_c))

    h, w = f_mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = float(r.max())
    r1, r2 = 0.15 * rmax, 0.45 * rmax

    etotal = f_mag.sum().item() + EPS
    elow   = f_mag[r <= r1].sum().item()
    emid   = f_mag[(r > r1) & (r <= r2)].sum().item()
    ehigh  = f_mag[r > r2].sum().item()

    rb   = torch.logspace(math.log10(1.0), math.log10(rmax + 1.0), 40)
    ridx = torch.bucketize(r.flatten() + 1.0, rb) - 1
    mu   = []
    flat_mag = f_mag.flatten()
    for i in range(len(rb) - 1):
        mask = ridx == i
        mu.append(torch.log(flat_mag[mask] + 1e-6).mean().item() if mask.any() else 0.0)
    xs = np.arange(len(mu))
    ys = np.nan_to_num(mu)
    slope = float(np.polyfit(xs, ys, 1)[0])

    phase_flat = f_phase.flatten()
    phase_hist = torch.histc(phase_flat, bins=50, min=-math.pi, max=math.pi)
    phase_prob = phase_hist / (phase_hist.sum() + EPS)
    phase_entropy = -torch.sum(phase_prob * torch.log(phase_prob + EPS)).item()

    coeffs1 = pywt.dwt2(x.numpy(), "db1")
    cA1, (cH1, cV1, cD1) = coeffs1
    coeffs2 = pywt.dwt2(cA1, "db1")
    cA2, (cH2, cV2, cD2) = coeffs2
    wavelet_energies = [
        np.mean(np.abs(c) ** 2)
        for c in [cA1, cH1, cV1, cD1, cA2, cH2, cV2, cD2]
    ]

    ang = torch.atan2(yy - cy, xx - cx)
    sect_means = []
    for a0 in np.linspace(-math.pi, math.pi, 8, endpoint=False):
        mask = (ang >= a0) & (ang < a0 + math.pi / 4)
        sect_means.append(f_mag[mask].mean().item() if mask.any() else 0.0)
    anis = float(np.var(sect_means))

    feats = [
        elow / etotal,
        emid / etotal,
        ehigh / etotal,
        (ehigh + EPS) / (elow + EPS),
        slope,
        anis,
        phase_entropy,
    ] + wavelet_energies
    return feats

def srm_features(pil: Image.Image):
    x = _pil_to_gray256_clahe(pil)[None, None, ...]
    feats: List[float] = []
    for k2d in SRM_K:
        k = (k2d / (k2d.abs().sum() + EPS)).view(1, 1, *k2d.shape)
        y = nn.functional.conv2d(x, k, padding="same")
        arr = y.flatten().numpy()
        mean = float(arr.mean())
        var  = float(arr.var())
        kurt = float(((arr - mean) ** 4).mean() / ((var + EPS) ** 2))
        feats += [mean, var, kurt]
    return feats

def extract_freq_vector(pil: Image.Image) -> torch.Tensor:
    feats = fft_features(pil) + srm_features(pil)
    return torch.tensor(feats, dtype=torch.float32)

# -------------------- v5 FreqMLP + AdaptiveFusionHead -------------------- #

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
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, dim)
    def forward(self, x: torch.Tensor):
        r = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x + r

class FreqMLP(nn.Module):
    def __init__(self, dim: int = 24, hidden: int = 64, num_bands: int = 4):
        super().__init__()
        self.normer  = FeatureNormalizer(dim)
        self.contrast = ContrastScaler(dim)
        self.band    = BandGating(dim, num_bands)
        self.blocks  = nn.ModuleList([
            ResidualMLPBlock(dim, hidden),
            ResidualMLPBlock(dim, hidden),
        ])
        self.head = nn.Linear(dim, 1)
        self.temp = TemperatureScaler()
    def forward(self, x: torch.Tensor):
        x = self.normer(x)
        x = self.contrast(x)
        x = self.band(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x).squeeze(-1)
        return self.temp(logits)

class AdaptiveFusionHead(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.temp = TemperatureScaler()
    def forward(self, z_freq: torch.Tensor, z_sig: torch.Tensor):
        diff = torch.abs(z_freq - z_sig)
        x    = torch.stack([z_freq, z_sig, diff], dim=-1)  # [B, 3]
        w    = F.softmax(self.mlp(x), dim=-1)              # [B, 2]
        z    = w[..., 0] * z_freq + w[..., 1] * z_sig
        return self.temp(z)

# -------------------- Helpers -------------------- #

def list_images(folder: str) -> List[str]:
    out: List[str] = []
    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, name))
    return sorted(out)

def extract_freq_logits(freq_model: FreqMLP, paths: Sequence[str]) -> torch.Tensor:
    feats: List[List[float]] = []
    for p in tqdm(paths, desc="freq", leave=False):
        with Image.open(p) as pil:
            feats.append(extract_freq_vector(pil.convert("RGB")).tolist())
    feats_t = torch.tensor(feats, dtype=torch.float32).to(FREQ_DEVICE)
    with torch.no_grad():
        logits = freq_model(feats_t).cpu()
    return logits  # [N]

@torch.no_grad()
def extract_siglip_logits(siglip: BinaryClassifier, paths: Sequence[str]) -> torch.Tensor:
    out: List[float] = []
    for p in tqdm(paths, desc="siglip", leave=False):
        with Image.open(p) as pil:
            x = preprocess(pil.convert("RGB")).unsqueeze(0).to(DEVICE)
        z = siglip(x).item()
        out.append(z)
    return torch.tensor(out, dtype=torch.float32)

def _make_loader(x: torch.Tensor, y: torch.Tensor, bs: int) -> DataLoader:
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")

# -------------------- Fusion training -------------------- #

def train_fusion_head(
    real_dir: str,
    fake_dir: str,
    best_model_path: str,
    freq_mlp_path: str,
    fusion_out: str,
    batch_size: int,
    epochs: int,
):
    real_paths = list_images(real_dir)
    fake_paths = list_images(fake_dir)
    if not real_paths or not fake_paths:
        raise SystemExit("No images found under real/fake dirs.")

    all_paths = real_paths + fake_paths
    labels = torch.cat(
        [
            torch.zeros(len(real_paths), dtype=torch.float32),
            torch.ones(len(fake_paths), dtype=torch.float32),
        ],
        dim=0,
    )

    print(f"[data] {len(real_paths)} real, {len(fake_paths)} fake, total {len(all_paths)}")

    # Load SigLIP (with your best_model.safetensors)
    siglip = load_siglip_from_best(best_model_path)

    # Load FreqMLP
    freq_model = FreqMLP().to(FREQ_DEVICE)
    freq_state = load_file(freq_mlp_path)
    freq_model.load_state_dict(freq_state, strict=True)
    freq_model.eval()
    print(f"[freq] Loaded FreqMLP: {freq_mlp_path}")

    # Extract logits
    print("[stage] Extracting frequency logits...")
    z_freq = extract_freq_logits(freq_model, all_paths)  # [N]
    print("[stage] Extracting SigLIP logits...")
    z_sig  = extract_siglip_logits(siglip, all_paths)    # [N]

    fusion_inputs = torch.stack([z_freq, z_sig], dim=1)  # [N, 2]

    loader = _make_loader(fusion_inputs, labels, batch_size)

    fusion_head = AdaptiveFusionHead().to(DEVICE)
    optim = torch.optim.AdamW(fusion_head.parameters(), lr=5e-4)
    crit  = nn.BCEWithLogitsLoss()

    best_auc = -float("inf")
    best_state = None

    print("[stage] Training AdaptiveFusionHead...")
    for ep in range(1, epochs + 1):
        fusion_head.train()
        losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            zf = xb[:, 0]
            zs = xb[:, 1]
            optim.zero_grad()
            logits = fusion_head(zf, zs)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(fusion_head.parameters(), max_norm=5.0)
            optim.step()
            losses.append(loss.item())

        fusion_head.eval()
        with torch.no_grad():
            logits_all = fusion_head(
                z_freq.to(DEVICE),
                z_sig.to(DEVICE),
            ).cpu()
            probs = torch.sigmoid(logits_all).numpy()
            auc = _safe_auc(labels.numpy(), probs)
            acc = float(((probs >= 0.5) == labels.numpy()).mean())

        print(
            f"[fusion] epoch {ep:03d}/{epochs} "
            f"loss={np.mean(losses):.4f} acc={acc:.3f} auc={auc:.3f}"
        )

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in fusion_head.state_dict().items()}

    if best_state is None:
        print("[fusion] WARNING: no best_state, not saving.")
        return

    save_file(best_state, fusion_out)
    print(f"[fusion] Saved trained fusion head to: {fusion_out}")
    print(f"[fusion] Best AUC on training set: {best_auc:.3f}")

# -------------------- CLI -------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description="Train ONLY fusion_head.safetensors")
    ap.add_argument("--real-dir", type=str, required=True, help="Folder with REAL images.")
    ap.add_argument("--fake-dir", type=str, required=True, help="Folder with FAKE images.")
    ap.add_argument("--best-model", type=str, required=True, help="Path to best_model.safetensors.")
    ap.add_argument("--freq-mlp", type=str, required=True, help="Path to freq_mlp.safetensors (v5 FreqMLP).")
    ap.add_argument("--fusion-out", type=str, required=True, help="Output path for fusion_head.safetensors.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    ap.add_argument("--epochs", type=int, default=5, help="Fusion training epochs.")
    return ap.parse_args()

def main():
    args = parse_args()
    train_fusion_head(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        best_model_path=args.best_model,
        freq_mlp_path=args.freq_mlp,
        fusion_out=args.fusion_out,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

if __name__ == "__main__":
    main()
