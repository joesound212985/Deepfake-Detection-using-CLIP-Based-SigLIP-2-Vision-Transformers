#!/usr/bin/env python3
"""
Fit CORAL calibration for v5 Detector:
- Takes fused logits (z_fused) from SigLIP + FreqMLP + FusionHead
- Learns ordinal cutpoints for 5 risk levels:
      REAL, LEAN_REAL, BORDERLINE, LEAN_FAKE, FAKE
- Saves:
      coral_cutpoints.json
      coral_temp.json
      coral_bins.npy

Usage:

python fit_coral_v5.py ^
  --real-dir "C:\Users\Admin\Desktop\Real-img\Real-img" ^
  --fake-dir "C:\Users\Admin\Desktop\Fake-img\Image" ^
  --best-model "C:\Users\Admin\Desktop\best_model.safetensors" ^
  --freq-mlp  "C:\Users\Admin\Desktop\freq_mlp.safetensors" ^
  --fusion-head "C:\Users\Admin\Desktop\fusion_head.safetensors"
"""

import argparse
import json
import math
import os
from typing import List

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFile
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from torchvision import transforms
import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)

IMG_SIZE   = 384
SIGLIP_DIM = 1024
EPS        = 1e-8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
FREQ_DEVICE= "cpu"

IMG_EXTS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff", ".gif", ".jfif", ".heic", ".heif"
)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def list_images(folder):
    out=[]
    for root,_,files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root,f))
    return sorted(out)

def apply_clahe(pil):
    arr=np.array(pil,dtype=np.uint8)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    for c in range(3):
        arr[:,:,c]=clahe.apply(arr[:,:,c])
    return Image.fromarray(arr)

preprocess = transforms.Compose([
    transforms.Lambda(apply_clahe),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3),
])

# ------------------------------------------------------------
# SigLIP binary classifier (same as v5 app)
# ------------------------------------------------------------

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone,_,_ = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP-384", pretrained="webli", device=DEVICE
        )
        self.se = nn.Sequential(
            nn.Linear(SIGLIP_DIM, SIGLIP_DIM//16),
            nn.ReLU(),
            nn.Linear(SIGLIP_DIM//16, SIGLIP_DIM),
            nn.Sigmoid()
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(SIGLIP_DIM),
            nn.Dropout(0.3),
            nn.Linear(SIGLIP_DIM, SIGLIP_DIM//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(SIGLIP_DIM//2, SIGLIP_DIM//4),
            nn.GELU(),
            nn.Linear(SIGLIP_DIM//4, 1),
        )

    def forward(self,x):
        with torch.no_grad():
            if x.shape[-1]!=IMG_SIZE:
                x=nn.functional.interpolate(x,size=(IMG_SIZE,IMG_SIZE))
            f=self.backbone.encode_image(x)
            f=f/(f.norm(dim=-1,keepdim=True)+1e-6)
        f=f*self.se(f)
        return self.cls(f).squeeze(-1)


def load_siglip(best_path):
    model = BinaryClassifier().to(DEVICE).eval()
    state = load_file(best_path)
    msd   = model.state_dict()
    filt  = {k:v for k,v in state.items() if k in msd and v.shape==msd[k].shape}
    model.load_state_dict(filt, strict=False)
    print("[siglip] loaded best_model")
    return model

# ------------------------------------------------------------
# Frequency extractor + FreqMLP (v5)
# ------------------------------------------------------------

SRM_K = [
    torch.tensor(
        [
            [0,0,0,0,0],
            [0,-1,2,-1,0],
            [0, 2,-4, 2,0],
            [0,-1,2,-1,0],
            [0,0,0,0,0],
        ],dtype=torch.float32),
    torch.tensor([[-1,2,-1],[2,-4,2],[-1,2,-1]],dtype=torch.float32),
    torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=torch.float32),
]

def gray256_clahe(pil):
    g=ImageOps.exif_transpose(pil).convert("L")
    arr=np.array(g,dtype=np.uint8)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    arr=clahe.apply(arr)
    g=Image.fromarray(arr).resize((256,256),Image.BICUBIC)
    return torch.from_numpy(np.asarray(g,dtype=np.float32)/255.0)

def fft_feats(pil):
    x=gray256_clahe(pil)
    F=torch.fft.fftshift(torch.fft.fft2(x))
    Fm=torch.abs(F)
    h,w=Fm.shape
    cy,cx=h//2,w//2
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing="ij")
    r=torch.sqrt((yy-cy)**2+(xx-cx)**2)
    rmax=float(r.max())
    r1,r2=0.15*rmax,0.45*rmax
    Et=float(Fm.sum().item())+EPS
    El=float(Fm[r<=r1].sum().item())
    Em=float(Fm[(r>r1)&(r<=r2)].sum().item())
    Eh=float(Fm[r>r2].sum().item())
    rb=torch.logspace(math.log10(1.0),math.log10(rmax+1.0),40)
    ridx=torch.bucketize(r.flatten()+1.0,rb)-1
    flat=Fm.flatten()
    mu=[]
    for i in range(len(rb)-1):
        m=(ridx==i)
        mu.append(torch.log(flat[m]+1e-6).mean().item() if m.any() else 0.0)
    xs=np.arange(len(mu))
    slope=float(np.polyfit(xs,np.nan_to_num(mu),1)[0])
    # phase entropy
    phase=torch.angle(F)
    ph=phase.flatten()
    hist=torch.histc(ph,bins=50,min=-math.pi,max=math.pi)
    prob=hist/(hist.sum()+EPS)
    pent=float(-(prob*torch.log(prob+EPS)).sum().item())
    # anisotropy
    ang=torch.atan2(yy-cy,xx-cx)
    sect=[]
    for a0 in np.linspace(-math.pi,math.pi,8,endpoint=False):
        m=(ang>=a0)&(ang<a0+math.pi/4)
        sect.append(Fm[m].mean().item() if m.any() else 0.0)
    anis=float(np.var(sect))
    # wavelets
    cA1,(cH1,cV1,cD1)=pywt.dwt2(gray256_clahe(pil).numpy(),"db1")
    cA2,(cH2,cV2,cD2)=pywt.dwt2(cA1,"db1")
    wave=[np.mean(np.abs(c)**2) for c in [cA1,cH1,cV1,cD1,cA2,cH2,cV2,cD2]]
    return [El/Et,Em/Et,Eh/Et,(Eh+EPS)/(El+EPS),slope,anis,pent]+wave

def srm_feats(pil):
    x=gray256_clahe(pil)[None,None,...]
    feats=[]
    for k2d in SRM_K:
        k=(k2d/(k2d.abs().sum()+EPS)).view(1,1,*k2d.shape)
        y=nn.functional.conv2d(x,k,padding="same")
        arr=y.flatten().numpy()
        m=float(arr.mean()); v=float(arr.var())
        kurt=float(((arr-m)**4).mean()/((v+EPS)**2))
        feats+=[m,v,kurt]
    return feats

def extract_freq_vector(pil):
    return torch.tensor(fft_feats(pil)+srm_feats(pil),dtype=torch.float32)

# --- new FreqMLP layers ---

class FeatureNormalizer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.register_buffer("mean",torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
    def forward(self,x):
        return (x-self.mean)/(self.std+1e-6)

class ContrastScaler(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.alpha=nn.Parameter(torch.ones(dim))
        self.beta =nn.Parameter(torch.zeros(dim))
    def forward(self,x):
        return torch.tanh(self.alpha*x+self.beta)

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T=nn.Parameter(torch.tensor(1.0))
    def forward(self,x):
        return x/(self.T+1e-6)

class BandGating(nn.Module):
    def __init__(self,dim,num_bands=4):
        super().__init__()
        self.band_dim=dim//num_bands
        self.gates=nn.Parameter(torch.zeros(num_bands))
        self.num_bands=num_bands
    def forward(self,x):
        chunks=torch.split(x,self.band_dim,dim=-1)
        g=torch.sigmoid(self.gates)
        return torch.cat([c*g[i] for i,c in enumerate(chunks)],dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self,dim,hid):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.fc1=nn.Linear(dim,hid)
        self.fc2=nn.Linear(hid,dim)
    def forward(self,x):
        r=x
        x=self.norm(x)
        x=F.gelu(self.fc1(x))
        x=self.fc2(x)
        return x+r

class FreqMLP(nn.Module):
    def __init__(self,dim=24,hid=64,bands=4):
        super().__init__()
        self.norm = FeatureNormalizer(dim)
        self.contrast=ContrastScaler(dim)
        self.band=BandGating(dim,bands)
        self.b1=ResidualBlock(dim,hid)
        self.b2=ResidualBlock(dim,hid)
        self.head=nn.Linear(dim,1)
        self.temp=TemperatureScaler()
    def forward(self,x):
        x=self.norm(x)
        x=self.contrast(x)
        x=self.band(x)
        x=self.b1(x)
        x=self.b2(x)
        return self.temp(self.head(x).squeeze(-1))


# ------------------------------------------------------------
# Adaptive Fusion Head (v5)
# ------------------------------------------------------------

class AdaptiveFusionHead(nn.Module):
    def __init__(self,hidden=32):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(3,hidden),nn.GELU(),
            nn.Linear(hidden,2),
        )
        self.temp=TemperatureScaler()
    def forward(self,zf,zs):
        d=torch.abs(zf-zs)
        x=torch.stack([zf,zs,d],dim=-1)
        w=F.softmax(self.mlp(x),dim=-1)
        z=w[...,0]*zf + w[...,1]*zs
        return self.temp(z)


# ------------------------------------------------------------
# CORAL fitting
# ------------------------------------------------------------

def fit_coral_cutpoints(logits, labels, num_classes=5):
    """
    Fit CORAL cutpoints by scanning quantiles.
    Bins roughly correspond to:
        REAL → LEAN_REAL → BORDERLINE → LEAN_FAKE → FAKE
    """
    logits = logits.numpy()
    labels = labels.numpy()

    # Sort logits from lowest (real) to highest (fake)
    order = np.argsort(logits)
    logits_sorted = logits[order]
    labels_sorted = labels[order]

    # We want ~5 bins — e.g., percentiles: 15%, 35%, 55%, 75%
    cuts = []
    for q in [0.15, 0.35, 0.55, 0.75]:
        idx = int(q * len(logits_sorted))
        val = logits_sorted[idx]
        # Convert prob→logit-like
        cuts.append(float(val))

    return cuts


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--best-model", required=True)
    ap.add_argument("--freq-mlp", required=True)
    ap.add_argument("--fusion-head", required=True)
    ap.add_argument("--out-prefix", default="coral")
    args=ap.parse_args()

    # Load images
    real_paths=list_images(args.real_dir)
    fake_paths=list_images(args.fake_dir)
    all_paths=real_paths+fake_paths
    labels = torch.cat([
        torch.zeros(len(real_paths),dtype=torch.float32),
        torch.ones(len(fake_paths),dtype=torch.float32)
    ])

    print(f"[load] {len(real_paths)} real, {len(fake_paths)} fake")

    # Load models
    siglip = load_siglip(args.best_model)
    freq   = FreqMLP().to(FREQ_DEVICE).eval()
    freq.load_state_dict(load_file(args.freq_mlp), strict=True)
    fusion = AdaptiveFusionHead().to(FREQ_DEVICE).eval()
    fusion.load_state_dict(load_file(args.fusion_head), strict=True)

    # Gather fused logits
    fused_logits = []
    for p in tqdm(all_paths,desc="fused"):
        with Image.open(p) as pil:
            pil=pil.convert("RGB")
            # SigLIP
            x = preprocess(pil).unsqueeze(0).to(DEVICE)
            z_sig = siglip(x).cpu()[0]
            # freq
            fvec = extract_freq_vector(pil).unsqueeze(0).to(FREQ_DEVICE)
            z_freq = freq(fvec).cpu()[0]
            # fused
            z_fused = fusion(
                torch.tensor([z_freq],device=FREQ_DEVICE),
                torch.tensor([z_sig], device=FREQ_DEVICE)
            )[0].cpu()
            fused_logits.append(float(z_fused))

    fused_logits = torch.tensor(fused_logits, dtype=torch.float32)

    # Fit CORAL cutpoints
    cutpoints = fit_coral_cutpoints(fused_logits, labels)
    print("[coral] cutpoints:", cutpoints)

    # Save cutpoints
    cp_path = args.out_prefix + "_cutpoints.json"
    with open(cp_path,"w") as f:
        json.dump(cutpoints,f,indent=2)
    print("[save]", cp_path)

    # Temperature (optional: calibrate)
    temp = 1.0
    temp_path = args.out_prefix + "_temp.json"
    with open(temp_path,"w") as f:
        json.dump({"temperature":temp},f,indent=2)
    print("[save]", temp_path)

    # Save bins (for visualization)
    bins = np.histogram(fused_logits.numpy(), bins=50)[0]
    np.save(args.out_prefix + "_bins.npy", bins)
    print("[save]", args.out_prefix + "_bins.npy")


if __name__=="__main__":
    main()
