#!/usr/bin/env python3
# ultra_diagnostics.py — high-level forensic diagnostics

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def extract_prnu(image, sigma=3):
    """
    Extract PRNU sensor noise from a BGR or RGB image.
    Returns a 2D float32 map.
    """
    if image.dtype != np.float32:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()

    # Convert to grayscale (use BGR→GRAY for cv2.imread input)
    if img.ndim == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        ) / 255.0
    else:
        gray = img

    smooth = gaussian_filter(gray, sigma)
    noise = gray - smooth
    noise -= noise.mean()
    noise /= (noise.std() + 1e-8)
    return noise.astype(np.float32)


def prnu_strength_map(noise, block=64):
    """
    Block-wise PRNU strength map (mean |noise| per block).
    """
    h, w = noise.shape
    H, W = h // block, w // block
    m = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            tile = noise[i * block : (i + 1) * block, j * block : (j + 1) * block]
            m[i, j] = float(np.mean(np.abs(tile)))
    return m


def prnu_fft_consistency_map(noise, block=64):
    """
    Block-wise PRNU FFT consistency map:
    lower values → more structured (real sensor),
    higher values → more random (synthetic PRNU).
    """
    h, w = noise.shape
    H, W = h // block, w // block
    m = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            tile = noise[i * block : (i + 1) * block, j * block : (j + 1) * block]
            fft = np.fft.fft2(tile)
            mag = np.abs(fft)
            radial = np.mean(mag, axis=0)
            sm = gaussian_filter(radial, 3)
            m[i, j] = float(np.mean(np.abs(radial - sm)))
    return m


def jpeg_residual_map(image, quality=95, block=8):
    """
    JPEG residual block map: recompress image and measure per-block residual energy.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, enc = cv2.imencode(".jpg", image, encode_param)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image for residual map.")
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rec is None:
        raise RuntimeError("Failed to JPEG-decode image for residual map.")

    diff = cv2.absdiff(image, rec).astype(np.float32) / 255.0
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    H, W = h // block, w // block
    m = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            tile = gray[i * block : (i + 1) * block, j * block : (j + 1) * block]
            m[i, j] = float(np.mean(tile))
    return m


def save_norm_heatmap(mat, out_path):
    """
    Normalize a 2D matrix to [0,255] and save as a JET heatmap PNG.
    """
    m = mat.astype(np.float32).copy()
    m -= float(m.min())
    if m.max() > 0:
        m /= float(m.max())
    m = (m * 255).astype(np.uint8)
    m = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_path), m)


def run_ultra(image_path, out_dir="ultra_out"):
    """
    Run high-level forensic diagnostics on a single image path, saving:
      - prnu_strength_map.png
      - prnu_fft_consistency_map.png
      - jpeg_residual_map.png
    into the specified output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1) PRNU maps
    noise = extract_prnu(img)
    prnu_map = prnu_strength_map(noise, block=64)
    prnu_fft_map = prnu_fft_consistency_map(noise, block=64)

    save_norm_heatmap(prnu_map, out_dir / "prnu_strength_map.png")
    save_norm_heatmap(prnu_fft_map, out_dir / "prnu_fft_consistency_map.png")

    # 2) JPEG residual map
    jpeg_map = jpeg_residual_map(img, quality=95, block=8)
    save_norm_heatmap(jpeg_map, out_dir / "jpeg_residual_map.png")

    print("[ULTRA] Diagnostics saved to:", out_dir)
    print("[ULTRA] PRNU map: prnu_strength_map.png")
    print("[ULTRA] PRNU-FFT consistency map: prnu_fft_consistency_map.png")
    print("[ULTRA] JPEG residual map: jpeg_residual_map.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ultra_diagnostics.py <image_path> [out_dir]")
        raise SystemExit(1)
    image_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "ultra_out"
    run_ultra(image_path, out_dir)

