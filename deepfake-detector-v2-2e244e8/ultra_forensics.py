#!/usr/bin/env python3
# ultra_forensics.py — Full ULTRA MODE Forensic Diagnostic Module
#
# Generates:
#   PRNU strength map
#   PRNU FFT consistency map
#   CFA anomaly map
#   JPEG residual map
#   Patch anomaly map
#   Frequency anomaly map
#   Combined anomaly locator heatmap

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================
# 1. ----------  PRNU SENSOR NOISE EXTRACTION  ---------------
# ============================================================

def extract_prnu(image, sigma=3):
    """Extract sensor PRNU (Photo-Response Non-Uniformity)."""
    if image.dtype != np.float32:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()

    if img.ndim == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = img

    smooth = gaussian_filter(gray, sigma)
    noise = gray - smooth
    noise -= noise.mean()
    noise /= (noise.std() + 1e-8)
    return noise.astype(np.float32)


def prnu_strength_map(noise, block=64):
    """Block-wise PRNU amplitude map."""
    h, w = noise.shape
    H, W = h // block, w // block
    out = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            tile = noise[i * block : (i + 1) * block, j * block : (j + 1) * block]
            out[i, j] = float(np.mean(np.abs(tile)))
    return out


def prnu_fft_consistency_map(noise, block=64):
    """Detect fake-PRNU injection or region inconsistencies."""
    h, w = noise.shape
    H, W = h // block, w // block
    out = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            tile = noise[i * block : (i + 1) * block, j * block : (j + 1) * block]
            fft = np.fft.fft2(tile)
            mag = np.abs(fft)
            radial = np.mean(mag, axis=0)
            smooth = gaussian_filter(radial, 3)
            out[i, j] = float(np.mean(np.abs(radial - smooth)))
    return out


# ============================================================
# 2. ---------------- CFA ANOMALY MAP ------------------------
# ============================================================

def cfa_anomaly_map(img, block=32):
    """
    CFA interpolation consistency per block.
    High values = suspicious (inpainting, splice, diffusion fill).
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    H, W = h // block, w // block
    M = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            t = gray[i * block : (i + 1) * block, j * block : (j + 1) * block]
            fx = cv2.Scharr(t, cv2.CV_32F, 1, 0)
            fy = cv2.Scharr(t, cv2.CV_32F, 0, 1)
            energy = np.mean(np.abs(fx) + np.abs(fy))
            M[i, j] = float(energy)

    # Normalize CFA anomaly (deviation from mean)
    M = np.abs(M - np.mean(M))
    return M


# ============================================================
# 3. --------- JPEG RESIDUAL BLOCK CONSISTENCY ---------------
# ============================================================

def jpeg_residual_map(img, block=8):
    """Recompress and compute JPEG error block map."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ok, enc = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image.")
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rec is None:
        raise RuntimeError("Failed to JPEG-decode image.")

    diff = cv2.absdiff(img, rec).astype(np.float32) / 255.0
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    H, W = h // block, w // block
    out = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            tile = gray[i * block : (i + 1) * block, j * block : (j + 1) * block]
            out[i, j] = float(np.mean(tile))
    return out


# ============================================================
# 4. ------------- PATCH ANOMALY (GRID CONSISTENCY) ----------
# ============================================================

def patch_anomaly_map(img, block=64):
    """Compute block variance differences between patches."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    H, W = h // block, w // block

    M = np.zeros((H, W), np.float32)
    for i in range(H):
        for j in range(W):
            tile = gray[i * block : (i + 1) * block, j * block : (j + 1) * block]
            M[i, j] = float(np.var(tile))

    M = np.abs(M - np.mean(M))
    return M


# ============================================================
# 5. --------- MULTISCALE FFT ANOMALY MAP --------------------
# ============================================================

def multiscale_fft_map(img, block=32):
    """Detect unnatural global or local frequency suppression."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    h, w = gray.shape
    H, W = h // block, w // block
    out = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            tile = gray[i * block : (i + 1) * block, j * block : (j + 1) * block]
            fft = np.fft.fft2(tile)
            mag = np.abs(fft)
            out[i, j] = float(np.mean(mag))

    return np.abs(out - np.mean(out))


# ============================================================
# 6. --------- PERLIN (DIFFUSION ARTIFACT) MAP ---------------
# ============================================================

def perlin_like_noise_map(img, block=32):
    """Detect smooth coherent AI noise fields common in diffusion."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = gray - gaussian_filter(gray, 3)

    h, w = noise.shape
    H, W = h // block, w // block
    out = np.zeros((H, W), np.float32)

    for i in range(H):
        for j in range(W):
            tile = noise[i * block : (i + 1) * block, j * block : (j + 1) * block]
            out[i, j] = float(np.var(tile))

    return np.abs(out - np.mean(out))


# ============================================================
# 7. ----------- COMBINED ANOMALY LOCATOR MAP ---------------
# ============================================================

def combined_anomaly_map(*maps):
    """Merge multiple anomaly maps into a unified suspiciousness map."""
    M = np.zeros_like(maps[0])
    for m in maps:
        m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
        M += m_norm
    M /= len(maps)
    return M


# ============================================================
# 8. -------------------- VISUALIZATION ----------------------
# ============================================================

def save_heatmap(mat, path):
    """Normalize a 2D map to [0,255] and save as a JET heatmap."""
    m = mat.astype(np.float32).copy()
    m -= float(m.min())
    m /= (float(m.max()) + 1e-8)
    m = (m * 255).astype(np.uint8)
    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), m)


# ============================================================
# 9. ---------------------- MAIN ENTRY ------------------------
# ============================================================

def ultra_forensics(image_path, out_dir="ultra_out"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    # PRNU noise
    noise = extract_prnu(img)

    # Generate all maps
    maps = {
        "prnu_strength": prnu_strength_map(noise, 64),
        "prnu_fft": prnu_fft_consistency_map(noise, 64),
        "cfa_anomaly": cfa_anomaly_map(img, 32),
        "jpeg_residual": jpeg_residual_map(img, 8),
        "patch_anomaly": patch_anomaly_map(img, 64),
        "fft_anomaly": multiscale_fft_map(img, 32),
        "perlin_anomaly": perlin_like_noise_map(img, 32),
    }

    # Combined locator map
    combined = combined_anomaly_map(*maps.values())
    maps["combined"] = combined

    # Save all maps
    for name, m in maps.items():
        save_heatmap(m, out / f"{name}.png")

    print("\n[ULTRA MODE] Diagnostics saved to:", out)
    for name in maps.keys():
        print(f"  → {name}.png")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ultra_forensics.py <input_image> [out_dir]")
        raise SystemExit(1)
    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "ultra_out"
    ultra_forensics(img_path, out_dir)

