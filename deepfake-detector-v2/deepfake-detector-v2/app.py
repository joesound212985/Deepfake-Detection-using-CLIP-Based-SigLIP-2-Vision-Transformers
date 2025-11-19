import os, io, math, numpy as np, torch, torch.nn as nn
import pywt
import cv2
from PIL import Image, ImageOps, ImageFile
from torchvision import transforms
try:
    from huggingface_hub import hf_hub_download, login, InferenceClient
except ImportError:
    from huggingface_hub import hf_hub_download, login
    InferenceClient = None
try:
    import requests
except Exception:
    requests = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
from safetensors.torch import load_file
import matplotlib
import matplotlib.pyplot as plt
import gradio as gr
import spaces
try:
    import gradio_client.utils as grc_utils
except Exception:
    grc_utils = None

try:
    import insightface
    _face_providers = (
        ['CUDAExecutionProvider'] if torch.cuda.is_available()
        else ['CPUExecutionProvider']
    )
    FACE_MODEL = insightface.app.FaceAnalysis(providers=_face_providers)
    FACE_MODEL.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    HAS_FACE = True
except Exception as _e:
    FACE_MODEL = None
    HAS_FACE = False
    print(f"[warn] insightface not available → face-specific boost disabled ({_e.__class__.__name__})")
    # Helpful hint for HF GPU Spaces: wrong ONNXRuntime build
    try:
        import onnxruntime as _ort
        if torch.cuda.is_available() and getattr(_ort, "get_device", lambda: "CPU")() == "CPU":
            print(
                "[hint] CUDA is available but onnxruntime reports CPU-only. "
                "On Hugging Face GPU Spaces you should swap "
                "onnxruntime==1.17.1 → onnxruntime-gpu==1.17.1 in requirements.txt."
            )
    except Exception:
        pass

# Stable plotting backend
matplotlib.use("Agg", force=True)

torch.set_grad_enabled(False)
torch.set_num_threads(2)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
#       PATCH GRADIO CLIENT JSON-SCHEMA BUG (optional)
# ============================================================
# Some gradio/gradio_client versions raise a TypeError when
# json_schema_to_python_type encounters {"additionalProperties": true}.
# We defensively wrap it so /info does not crash the Space.
if grc_utils is not None and hasattr(grc_utils, "json_schema_to_python_type"):
    _orig_json_schema_to_python_type = grc_utils.json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        try:
            return _orig_json_schema_to_python_type(schema, defs)
        except TypeError:
            # Fallback: treat unexpected schemas as "Any"-like"
            return "Any"

    grc_utils.json_schema_to_python_type = _safe_json_schema_to_python_type

# ============================================================
#          HUGGING FACE MODEL REPO + CONFIG
# ============================================================

MODEL_REPO = "joesound212985/siglip"   # Must contain best_model.safetensors + freq_mlp.safetensors + CORAL files
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/hf-cache")
os.makedirs(CACHE_DIR, exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional runtime toggles (no re-training required)
DETECT_USE_CLAHE = os.getenv("DETECT_USE_CLAHE", "0").strip() in {"1","true","True"}
DETECT_USE_FUSION = os.getenv("DETECT_USE_FUSION", "1").strip() in {"1","true","True"}
DETECT_USE_STABILIZER = os.getenv("DETECT_USE_STABILIZER", "1").strip() in {"1","true","True"}
DETECT_USE_FORENSICS = os.getenv("DETECT_USE_FORENSICS", "1").strip() in {"1","true","True"}
DETECT_EXTRA_TTA = os.getenv("DETECT_EXTRA_TTA", "0").strip() in {"1","true","True"}

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("[debug] HF_TOKEN detected: True (login ok)")
    except Exception as e:
        print(f"[debug] HF_TOKEN detected but login failed: {e.__class__.__name__}")
else:
    print("[debug] HF_TOKEN detected: False")

# ============================================================
#           OPTIONAL LLM EXPLANATION (DeepSeek R1)
# ============================================================

HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

if InferenceClient is not None:
    try:
        _deepseek_client = InferenceClient(model=HF_LLM_MODEL, token=HF_TOKEN)
        print(f"[llm] DeepSeek client initialized for model: {HF_LLM_MODEL}")
    except Exception as _e:
        _deepseek_client = None
        print(f"[llm] Failed to init DeepSeek client: {_e.__class__.__name__}")
else:
    _deepseek_client = None


def explain_with_deepseek(metrics: dict) -> str:
    """
    Use DeepSeek-R1-Distill-Qwen-7B (or compatible HF LLM)
    to generate a short, human-readable explanation of the
    detector metrics.
    """
    if _deepseek_client is None:
        return (
            "Automatic explanation disabled or unavailable. "
            "The numeric forensic scores above are still valid."
        )

    prompt = f"""
You are a deepfake forensics assistant.
You receive numeric outputs from a deepfake detector (SigLIP + frequency + forensic heads).
Your job: explain in under 150 words whether the image is likely REAL or FAKE and why,
using clear language that a non-expert can understand.
Be concise, avoid math formulas, and reference only the most important signals.
Here are the metrics (keys are feature names):
{metrics}
"""

    # Prefer InferenceClient.conversational if available and compatible
    try:
        if hasattr(_deepseek_client, "conversational"):
            raw = _deepseek_client.conversational(prompt)
        else:
            # Fallback: call HF Inference API directly for conversational task
            if requests is None or HF_TOKEN is None:
                raise RuntimeError("requests or HF_TOKEN not available for LLM call")

            api_url = f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}"
            payload = {
                "inputs": {
                    "past_user_inputs": [],
                    "generated_responses": [],
                    "text": prompt,
                }
            }
            headers = {"Authorization": f"Bearer {HF_TOKEN}"}
            resp = requests.post(api_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
    except Exception as e:
        print(f"[llm] DeepSeek explanation error: {e}")
        return (
            f"Automatic explanation unavailable (LLM error: {e.__class__.__name__}). "
            "Please rely on the numeric scores above."
        )

    # Try to extract a string response from the conversational payload
    text = ""
    try:
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, dict):
            if "generated_text" in raw:
                text = str(raw["generated_text"])
            elif "conversation" in raw and isinstance(raw["conversation"], dict):
                conv = raw["conversation"]
                if isinstance(conv.get("generated_responses"), list) and conv["generated_responses"]:
                    text = str(conv["generated_responses"][-1])
                elif "generated_text" in conv:
                    text = str(conv["generated_text"])
                else:
                    text = _json.dumps(raw)
            else:
                text = _json.dumps(raw)
        elif isinstance(raw, list) and raw:
            item = raw[-1]
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict) and "generated_text" in item:
                text = str(item["generated_text"])
            else:
                text = _json.dumps(raw)
        else:
            text = str(raw)
    except Exception:
        text = str(raw)

    # Strip DeepSeek-R1-style hidden reasoning, if present
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    return text

# ============================================================
#              LOAD CALIBRATED CORAL FILES (SIMPLE)
# ============================================================

import json as _json

def load_coral():
    print("[coral] downloading…")
    try:
        coral_cutpoints_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="coral_cutpoints.json",
            cache_dir=CACHE_DIR,
            token=HF_TOKEN,
        )
        coral_temp_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="coral_temp.json",
            cache_dir=CACHE_DIR,
            token=HF_TOKEN,
        )
        coral_bins_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="coral_bins.npy",
            cache_dir=CACHE_DIR,
            token=HF_TOKEN,
        )
        # cutpoints can be dict or list/values
        with open(coral_cutpoints_path) as f:
            coral_cuts = _json.load(f)
        # temp can be float or {"temp": float} or {"temperature": float}
        with open(coral_temp_path) as f:
            raw_temp = _json.load(f)
            if isinstance(raw_temp, dict):
                coral_temp = float(raw_temp.get("temp", raw_temp.get("temperature", 1.0)))
            else:
                coral_temp = float(raw_temp)
        coral_bins = np.load(coral_bins_path)
        print("[coral] Loaded CORAL calibration successfully.")
        print("[debug] CORAL_CUTS =", coral_cuts)
        print("[debug] CORAL_TEMP =", coral_temp)
        print("[debug] CORAL_BINS loaded? ", coral_bins is not None)
        return coral_cuts, coral_temp, coral_bins
    except Exception as e:
        print("[coral] FAILED:", e)
        print("[coral] Fallback → CORAL disabled: temp=1.0, no cutpoints.")
        return None, 1.0, None

CORAL_CUTS, CORAL_TEMP, CORAL_BINS = load_coral()

# ============================================================
#                 DEVICE + CONSTANTS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FREQ_DEVICE = "cpu"   # keeps frequency MLP cheap

IMG_SIZE = 384
EPS = 1e-8

MIN_SIDE = 64      # Reject tiny images
MAX_SIDE = 2048    # Auto-downscale very large images

# ============================================================
#            BALANCED DETECTION CONFIG (IMPORTANT)
# ============================================================

# TEMPERATURE is now replaced by CORAL_TEMP from calibration
STATIC_THRESHOLD_GLOBAL    = 0.55   # was 0.45 previously (less aggressive)
STATIC_THRESHOLD_PATCH     = 0.80   # was 0.65 previously (much less aggressive)

SIGLIP_WEIGHT              = 0.40
FREQ_WEIGHT                = 0.60
FREQ_TEMP                  = 1.25   # Softens freq MLP spikes

EDGE_NUDGE                 = 0.03   # Only applied in ambiguous band

PATCH_GRID_ROWS = 4
PATCH_GRID_COLS = 4

# Final decision threshold for p_enhanced (simple rule)
FINAL_THRESHOLD = float(os.getenv("FINAL_THRESHOLD", "0.52"))

# Optional: cosine anomaly uses a real-image mean embedding
REAL_REF_DIR   = os.getenv("REAL_REF_DIR")  # Optional folder with real photos
MEAN_EMBEDDING = None
MEAN_EMBED_CACHE = os.path.join(CACHE_DIR, "mean_real_embedding.npy")

# ============================================================
#               PREPROCESSING FOR SIGLIP
# ============================================================

def apply_clahe(pil):
    """Apply per-channel CLAHE to improve low-contrast inputs (optional)."""
    try:
        import cv2
        arr = np.array(pil, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for ch in range(3):
            arr[..., ch] = clahe.apply(arr[..., ch])
        return Image.fromarray(arr)
    except Exception:
        return pil

if DETECT_USE_CLAHE:
    preprocess = transforms.Compose([
        transforms.Lambda(apply_clahe),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
else:
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

# ============================================================
#        SIGLIP (OpenCLIP WebLI) + CLASSIFICATION HEAD
# ============================================================

class BinaryClassifier(nn.Module):
    def __init__(self, model_size="large", device="cpu"):
        super().__init__()
        import open_clip  # from pip: open-clip-torch

        cfg = {
            "small": ("ViT-B-16-SigLIP-384", 768),
            "large": ("ViT-L-16-SigLIP-384", 1024),
        }
        name, dim = cfg[model_size]

        # Load SigLIP webli backbone
        self.backbone, _, _ = open_clip.create_model_and_transforms(
            name, pretrained="webli", device=device
        )

        # Squeeze-and-Excitation (makes embeddings sharper)
        self.se = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid()
        )

        # 3-layer MLP head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.3),
            nn.Linear(dim, dim//2), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim//2, dim//4), nn.GELU(),
            nn.Linear(dim//4, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            if x.shape[-1] != IMG_SIZE:
                x = nn.functional.interpolate(x, size=(IMG_SIZE, IMG_SIZE))
            f = self.backbone.encode_image(x)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)

        se = self.se(f)
        z = self.classifier(f * se).squeeze(-1)
        return z


def _filter_state_for_model(state, model):
    """Keeps only matching keys/shapes (SigLIP checkpoints are large)."""
    msd = model.state_dict()
    return {
        k: v for k, v in state.items()
        if (not k.startswith("backbone.text.")) and (k in msd) and (v.shape == msd[k].shape)
    }


@torch.no_grad()
def get_siglip_model():
    """Loads SigLIP + MLP head from HF repo."""
    desired_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(get_siglip_model, "_cache"):
        best_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="best_model.safetensors",
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
        )

        state = load_file(best_path)
        model = BinaryClassifier("large", desired_device).to(desired_device).eval()
        model.load_state_dict(_filter_state_for_model(state, model), strict=False)
        
        get_siglip_model._cache = model
        print(f"[init] SigLIP loaded on {desired_device} from {best_path}")
    else:
        # Move cached model if device availability changed
        m = get_siglip_model._cache
        current = next(m.parameters()).device.type
        if current != desired_device:
            m = m.to(desired_device).eval()
            get_siglip_model._cache = m
            print(f"[init] SigLIP moved to {desired_device}")
    return get_siglip_model._cache


def _list_images_recursive(root):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dirpath, f))
    return paths


# === REAL-IMAGE EMBEDDING CALIBRATION ===
@torch.no_grad()
def compute_mean_embedding():
    """Populate MEAN_EMBEDDING from REAL_REF_DIR, local cache, or HF fallback."""
    global MEAN_EMBEDDING
    if MEAN_EMBEDDING is not None:
        return

    # 1) Try local cached embedding in CACHE_DIR
    if os.path.isfile(MEAN_EMBED_CACHE):
        try:
            MEAN_EMBEDDING = np.load(MEAN_EMBED_CACHE)
            print(f"[calib] Loaded mean_real_embedding.npy from local cache: {MEAN_EMBED_CACHE}")
            return
        except Exception as e:
            print(f"[calib] Failed to load local mean_real_embedding.npy: {e}")

    # 2) Prefer local REAL_REF_DIR if provided
    if REAL_REF_DIR and os.path.isdir(REAL_REF_DIR):
        print(f"[calib] Computing mean embedding from REAL_REF_DIR={REAL_REF_DIR}…")
        siglip = get_siglip_model()
        all_paths = _list_images_recursive(REAL_REF_DIR)
        if len(all_paths) < 10:
            print(f"[calib] Too few reference images in REAL_REF_DIR ({len(all_paths)}); skipping.")
        else:
            # Sample up to 200 diverse images
            paths = all_paths[:200]
            embeds = []
            for p in paths:
                try:
                    pil = Image.open(p).convert("RGB")
                    if min(pil.size) < 128:
                        continue
                    x = preprocess(pil).unsqueeze(0).to(DEVICE)
                    feat = siglip.backbone.encode_image(x)
                    feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-6)
                    embeds.append(feat.cpu().numpy())
                except Exception as e:
                    print(f"[calib] Skipping {p}: {e}")
                    continue
            if embeds:
                arr = np.concatenate(embeds, axis=0)
                MEAN_EMBEDDING = arr.mean(axis=0)
                # Cache to disk for future runs
                try:
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    np.save(MEAN_EMBED_CACHE, MEAN_EMBEDDING)
                    print(f"[calib] Saved mean_real_embedding.npy to {MEAN_EMBED_CACHE}")
                except Exception as e:
                    print(f"[calib] Failed to save mean_real_embedding.npy: {e}")
                print(f"[calib] Mean embedding computed from {len(arr)} real crops")
                return

    # 3) Fallback: try to load a precomputed mean embedding from HF
    if MEAN_EMBEDDING is None:
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename="mean_real_embedding.npy",
                token=HF_TOKEN,
                cache_dir=CACHE_DIR,
            )
            MEAN_EMBEDDING = np.load(path)
            print("[calib] Loaded mean_real_embedding.npy from HF repo")
        except Exception:
            print("[calib] No REAL_REF_DIR / HF mean embedding available; cosine anomaly disabled")

# ============================================================
#        SIMPLE PATH LOADER FOR MODEL WEIGHTS (OPTIONAL)
# ============================================================

def load_siglip_models():
    """Download model artifacts to CACHE_DIR and return their paths.
    Includes best_model.safetensors, freq_mlp.safetensors, and optional fusion head.
    OpenCLIP head file is optional; backbone loads via open-clip-torch.
    """
    print("[init] downloading model weights…")
    best_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.safetensors",
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
    )
    # Optional: custom OpenCLIP head checkpoint
    openclip_path = None
    try:
        openclip_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="open_clip_model.safetensors",
            cache_dir=CACHE_DIR,
            token=HF_TOKEN,
        )
    except Exception:
        pass
    freq_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="freq_mlp.safetensors",
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
    )
    fusion_head = None
    try:
        fusion_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="fusion_head.safetensors",
            cache_dir=CACHE_DIR,
            token=HF_TOKEN,
        )
        fusion_head = fusion_path
        print("[fusion] fusion head loaded")
    except Exception:
        print("[fusion] fusion head missing → MoE fallback")
    return best_path, openclip_path, freq_path, fusion_head

# ============================================================
#           OPTIONAL XGBOOST FUSION HEAD (v6 STYLE)
# ============================================================

_XGB_MODEL = None
_PLATT = None


def load_xgb_fusion():
    """
    Load XGBoost fusion model + Platt scaler from the HF repo.
    Requires xgb_fusion.json and platt.json to be present in MODEL_REPO.
    Returns (model, platt_dict) or (None, None) on failure.
    """
    global _XGB_MODEL, _PLATT

    if xgb is None:
        print("[xgb] xgboost not installed; XGB fusion disabled.")
        return None, None

    if _XGB_MODEL is not None and _PLATT is not None:
        return _XGB_MODEL, _PLATT

    try:
        xgb_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="xgb_fusion.json",
            cache_dir=CACHE_DIR,
            token=HF_TOKEN,
        )
        platt_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="platt.json",
            cache_dir=CACHE_DIR,
            token=HF_TOKEN,
        )
    except Exception as e:
        print(f"[xgb] Could not download fusion files from HF: {e}")
        return None, None

    try:
        model = xgb.Booster()
        model.load_model(xgb_path)
    except Exception as e:
        print(f"[xgb] Failed to load XGBoost model: {e}")
        return None, None

    try:
        with open(platt_path, "r") as f:
            platt = _json.load(f)
    except Exception as e:
        print(f"[xgb] Failed to load Platt scaler: {e}")
        return None, None

    _XGB_MODEL = model
    _PLATT = platt
    print(f"[xgb] Loaded fusion model from: {xgb_path}")
    print(f"[xgb] Loaded Platt scaler from: {platt_path}")
    return _XGB_MODEL, _PLATT

# ============================================================
#                  FREQUENCY MLP (24-D)
# ============================================================

class SafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias  = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias


class FreqMLP(nn.Module):
    def __init__(self, in_dim=24, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            SafeLayerNorm(in_dim),
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.Linear(hid, 1),
        )

    def forward(self, x):
        # v4.3.1 smoothing improvement - prevents jitter spikes
        if not self.training:
            x = x + 0.001 * torch.randn_like(x)
        return self.net(x).squeeze(-1)  # logit output


@torch.no_grad()
def get_freq_mlp():
    """Load frequency MLP from HF repo."""
    if not hasattr(get_freq_mlp, "_cache"):
        freq_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="freq_mlp.safetensors",
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
        )
        model = FreqMLP().to(FREQ_DEVICE)
        model.load_state_dict(load_file(freq_path))
        model.eval()
        get_freq_mlp._cache = model
        print(f"[init] FreqMLP loaded on {FREQ_DEVICE} from {freq_path}")

    return get_freq_mlp._cache

# ============================================================
#                     V5 ADAPTIVE FUSION HEAD (OPTIONAL)
# ============================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(1.0))
    def forward(self, logits: torch.Tensor):
        return logits / (self.T + 1e-6)


class AdaptiveFusionHeadV5(nn.Module):
    """
    Matches the v5 fusion head: inputs are z_freq and z_sig; features use
    [z_freq, z_sig, |diff|] -> MLP -> softmax weights -> weighted sum -> temp scale.
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.temp = TemperatureScaler()

    def forward(self, z_freq: torch.Tensor, z_sig: torch.Tensor):
        diff = torch.abs(z_freq - z_sig)
        x    = torch.stack([z_freq, z_sig, diff], dim=-1)
        w    = torch.softmax(self.mlp(x), dim=-1)
        z    = w[..., 0] * z_freq + w[..., 1] * z_sig
        return self.temp(z)


@torch.no_grad()
def get_fusion_head():
    """Load simple 2-input fusion head (required)."""
    if not DETECT_USE_FUSION:
        raise SystemExit("Fusion head required but DETECT_USE_FUSION=0")
    if hasattr(get_fusion_head, "_cache"):
        return get_fusion_head._cache

    class FusionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 1)
        def forward(self, x):
            return self.fc(x)

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="fusion_head.safetensors",
        token=HF_TOKEN,
        cache_dir=CACHE_DIR,
    )
    state = load_file(path)
    head = FusionHead().to(DEVICE).eval()
    head.load_state_dict(state, strict=True)
    get_fusion_head._cache = head
    print(f"[fusion] Loaded fusion_head.safetensors on {DEVICE} from {path} (2->1 linear)")
    return head

# ============================================================
#              FFT + SRM FORENSIC FEATURES (24-D)
# ============================================================

SRM_K = [
    torch.tensor(
        [[0,0,0,0,0],
         [0,-1,2,-1,0],
         [0,2,-4,2,0],
         [0,-1,2,-1,0],
         [0,0,0,0,0]], dtype=torch.float32
    ),
    torch.tensor(
        [[-1,2,-1],
         [2,-4,2],
         [-1,2,-1]], dtype=torch.float32
    ),
    torch.tensor(
        [[0,-1,0],
         [-1,4,-1],
         [0,-1,0]], dtype=torch.float32
    ),
]


def _pil_to_gray256(pil):
    g = ImageOps.exif_transpose(pil).convert("L")
    if DETECT_USE_CLAHE:
        try:
            import cv2
            arr8 = np.array(g, dtype=np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            arr8 = clahe.apply(arr8)
            g = Image.fromarray(arr8)
        except Exception:
            pass
    g = g.resize((256,256), Image.BICUBIC)
    arr = np.asarray(g, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def fft_features(pil):
    x = _pil_to_gray256(pil)
    F = torch.fft.fftshift(torch.fft.fft2(x))
    F_mag = torch.abs(F)
    F_phase = torch.angle(F)

    h, w = F_mag.shape
    cy, cx = h // 2, w // 2

    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    rmax = float(r.max())

    # radial bands
    r1, r2 = 0.15*rmax, 0.45*rmax
    Et = float(F_mag.sum().item()) + EPS
    El = float(F_mag[r <= r1].sum().item())
    Em = float(F_mag[(r > r1) & (r <= r2)].sum().item())
    Eh = float(F_mag[r > r2].sum().item())

    # log-spectrum slope
    rb = torch.logspace(math.log10(1.0), math.log10(rmax + 1.0), 40)
    ridx = torch.bucketize(r.flatten() + 1.0, rb) - 1

    mu = []
    flatF = F_mag.flatten()
    for i in range(len(rb)-1):
        mask = (ridx == i)
        if mask.any():
            mu.append(float(torch.log(flatF[mask] + 1e-6).mean().item()))
        else:
            mu.append(np.nan)

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
    cA1, (cH1, cV1, cD1) = pywt.dwt2(_pil_to_gray256(pil).numpy(), "db1")
    cA2, (cH2, cV2, cD2) = pywt.dwt2(cA1, "db1")
    wave = [
        np.mean(np.abs(c)**2)
        for c in [cA1, cH1, cV1, cD1, cA2, cH2, cV2, cD2]
    ]

    feats = [
        El/Et,
        Em/Et,
        Eh/Et,
        (Eh+EPS)/(El+EPS),
        slope,
        anis,
        phase_entropy,
    ] + wave

    return feats, F_mag.numpy()


def srm_features(pil):
    x = _pil_to_gray256(pil)[None, None, ...]   # (1,1,256,256)
    feats = []
    for k2d in SRM_K:
        k = (k2d / (k2d.abs().sum() + EPS)).view(1,1,*k2d.shape)
        y = nn.functional.conv2d(x, k, padding=k2d.shape[-1]//2)
        arr = y.flatten().numpy()
        m = float(arr.mean())
        v = float(arr.var())
        kurt = float(((arr - m)**4).mean() / ((v+EPS)**2))
        feats += [m,v,kurt]
    return feats


def extract_freq_vector(pil):
    f, _ = fft_features(pil)
    s = srm_features(pil)
    v = torch.tensor(f + s, dtype=torch.float32)
    if v.std() < 1e-6:
        return v*0.0
    return (v - v.mean()) / (v.std() + 1e-6)

# ============================================================
#            ADDITIONAL FORENSIC DETECTORS (OPTIONAL)
# ============================================================

def wavelet_inconsistency_score(img_np: np.ndarray) -> float:
    gray = np.mean(img_np, axis=2).astype(np.float32)
    try:
        LL, (LH, HL, HH) = pywt.dwt2(gray, 'bior4.4')
    except Exception:
        LL, (LH, HL, HH) = pywt.dwt2(gray, 'db1')
    def _norm(a):
        a = np.abs(a)
        return a / (a.mean() + 1e-6)
    LHn = _norm(LH); HLn = _norm(HL); HHn = _norm(HH)
    var_lh = float(np.var(LHn))
    var_hl = float(np.var(HLn))
    var_hh = float(np.var(HHn))
    score = abs(var_lh - var_hl) + abs(var_hh - var_lh)
    return float(score)

def benford_distance(data: np.ndarray) -> float:
    x = np.abs(data).flatten()
    x = x[x > 1]
    if x.size == 0:
        return 0.0
    # Leading digits 1..9
    mags = np.floor(np.log10(x) + 1e-9)
    leading = (x // (10 ** mags)).astype(np.int64)
    leading = leading[(leading >= 1) & (leading <= 9)]
    if leading.size == 0:
        return 0.0
    counts = np.bincount(leading, minlength=10)[1:10].astype(np.float64)
    counts = counts / (counts.sum() + 1e-8)
    benford = np.array([np.log10(1 + 1/d) for d in range(1,10)], dtype=np.float64)
    return float(np.sum(np.abs(counts - benford)))

def benford_wavelet_score(img_np: np.ndarray) -> float:
    gray = np.mean(img_np, axis=2).astype(np.float32)
    try:
        LL, (LH, HL, HH) = pywt.dwt2(gray, 'bior4.4')
    except Exception:
        LL, (LH, HL, HH) = pywt.dwt2(gray, 'db1')
    d_LH = benford_distance(LH)
    d_HL = benford_distance(HL)
    d_HH = benford_distance(HH)
    return float((d_LH + d_HL + d_HH) / 3.0)

def extract_prnu(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    denoised_u8 = cv2.fastNlMeansDenoising(gray.astype(np.uint8), None, 10, 7, 21)
    denoised = denoised_u8.astype(np.float32)
    residual = gray - denoised
    hp = residual - cv2.GaussianBlur(residual, ksize=(0,0), sigmaX=3)
    hp = hp / (np.std(hp) + 1e-6)
    return hp

def prnu_consistency_score(img_np: np.ndarray) -> float:
    prnu = extract_prnu(img_np)
    return float(np.var(prnu.flatten()))


def noiseprint_score(img_np: np.ndarray) -> float:
    """Approximate camera noiseprint consistency: higher = more fake."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    denoised_u8 = cv2.fastNlMeansDenoising(
        gray.astype(np.uint8), None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    residual = gray - denoised_u8.astype(np.float32)
    patches = []
    H, W = residual.shape
    for y in range(0, H - 64, 32):
        for x in range(0, W - 64, 32):
            p = residual[y:y+64, x:x+64]
            patches.append(np.var(p))
    if not patches:
        return 0.0
    patches = np.array(patches, dtype=np.float32)
    mean_v = float(patches.mean()) + 1e-6
    std_v = float(patches.std())
    consistency = 1.0 - min(std_v / mean_v, 1.0)
    fake_score = 1.0 - consistency  # higher = more inconsistent noise → more fake
    return float(np.clip(fake_score, 0.0, 1.0))

def emd_mode_mixing_score(img_np: np.ndarray) -> float:
    try:
        from PyEMD import EMD  # optional
    except Exception:
        return 0.0
    gray = np.mean(img_np, axis=2).astype(np.float32)
    signal = gray.mean(axis=0)
    try:
        IMFs = EMD().emd(signal)
    except Exception:
        return 0.0
    if IMFs is None or len(IMFs) == 0:
        return 0.0
    energies = np.array([np.sum(imf**2) for imf in IMFs], dtype=np.float64)
    energies = energies / (energies.sum() + 1e-8)
    entropy = float(-(energies * np.log(energies + 1e-12)).sum())
    return entropy

def forensic_score(img_np: np.ndarray) -> float:
    try:
        wv = wavelet_inconsistency_score(img_np)
        bf = benford_wavelet_score(img_np)
        em = emd_mode_mixing_score(img_np)
        pr = prnu_consistency_score(img_np)
        # Normalize to 0-1
        wv_n = min(wv / 1.0, 1.0)
        bf_n = min(bf / 0.20, 1.0)
        em_n = min(em / 1.5, 1.0)
        pr_n = 1.0 - min(pr / 3.5, 1.0)
        fusion = (0.25*wv_n + 0.25*bf_n + 0.20*em_n + 0.30*pr_n)
        return float(np.clip(fusion, 0.0, 1.0))
    except Exception:
        return 0.0

# ============================================================
#  FORENSIC V2: Classic + Diffusion Deepfake Predictors
# ============================================================

def perlin_residual_score(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hp = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    f = np.fft.fft2(hp)
    fshift = np.fft.fftshift(f)
    psd = np.abs(fshift) ** 2

    H, W = psd.shape
    cy, cx = H // 2, W // 2
    ys, xs = np.indices(psd.shape)
    r_float = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    r_int = r_float.astype(np.int32)
    r_max = int(r_float.max())

    # Focus on mid-high frequency band (avoid DC / extreme corners)
    r_min_band = max(1, int(0.2 * r_max))
    r_max_band = max(r_min_band + 1, int(0.8 * r_max))

    radial_mean = []
    for rad in range(r_min_band, r_max_band):
        mask = (r_int == rad)
        if mask.sum() > 0:
            radial_mean.append(psd[mask].mean())

    if len(radial_mean) == 0:
        return 0.0

    radial_mean = np.array(radial_mean, dtype=np.float32) + 1e-9
    radial_norm = radial_mean / radial_mean.max()

    flatness = 1.0 - float(np.var(radial_norm))
    return float(np.clip(flatness, 0, 1))


def diffusion_perlin_residual(img_np):
    """
    Returns a fake score in [0,1].
    Higher = more likely diffusion-generated.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)

    residual = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    F = np.fft.fft2(residual)
    Fshift = np.fft.fftshift(F)
    PSD = np.abs(Fshift) ** 2

    H, W = PSD.shape
    cy, cx = H // 2, W // 2
    ys, xs = np.indices(PSD.shape)

    R_float = np.sqrt((ys - cy)**2 + (xs - cx)**2)
    R = R_float.astype(np.int32)
    rmax = int(R_float.max())

    # Focus on mid-high frequency band only
    r_min_band = max(2, int(0.2 * rmax))
    r_max_band = max(r_min_band + 1, int(0.8 * rmax))

    radial_power = []
    for r in range(r_min_band, r_max_band):
        mask = (R == r)
        if mask.sum() > 0:
            radial_power.append(PSD[mask].mean())

    if not radial_power:
        return 0.0

    radial_power = np.array(radial_power, dtype=np.float32) + 1e-8
    radial_norm = radial_power / radial_power.max()

    f = np.arange(len(radial_norm), dtype=np.float32)
    log_f = np.log(f + 1e-6)
    log_p = np.log(radial_norm + 1e-6)

    A = np.vstack([log_f, np.ones_like(log_f)]).T
    slope, _ = np.linalg.lstsq(A, log_p, rcond=None)[0]
    slope = float(slope)

    diffusion_score = (slope + 1.0) / 1.0
    diffusion_score = float(np.clip(diffusion_score, 0.0, 1.0))
    return diffusion_score


def vov_score(img_np, patch_size=32):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    H, W = gray.shape

    vars_ = []
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            vars_.append(np.var(patch))

    vars_ = np.array(vars_, dtype=np.float32)
    if len(vars_) < 4:
        return 0.0

    v = float(np.var(vars_))
    norm_v = v / (v + 0.05)
    score = 1.0 - norm_v
    return float(np.clip(score, 0, 1))


def self_similarity_anomaly_score(img_np, patch=16, stride=8, max_patches=200):
    small = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_AREA)
    H, W, _ = small.shape
    patches = []
    coords = []

    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            p = small[y:y+patch, x:x+patch].astype(np.float32) / 255.0
            patches.append(p.reshape(-1))
            coords.append((y, x))

    patches = np.stack(patches, axis=0)
    coords = np.array(coords)
    N = patches.shape[0]

    if N > max_patches:
        idx = np.random.choice(N, max_patches, replace=False)
        patches = patches[idx]
        coords = coords[idx]
        N = max_patches

    norms = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-9
    patches_n = patches / norms

    sims = []
    for i in range(N):
        for j in range(i+1, N):
            if abs(coords[i,0] - coords[j,0]) < patch*2 and abs(coords[i,1] - coords[j,1]) < patch*2:
                continue
            s = float(np.dot(patches_n[i], patches_n[j]))
            sims.append(s)

    if not sims:
        return 0.0

    sims = np.array(sims)
    high = np.mean(sims > 0.90)
    return float(np.clip(high, 0, 1))


def diffusion_score(img_np):
    s1 = perlin_residual_score(img_np)
    s2 = vov_score(img_np)
    s3 = self_similarity_anomaly_score(img_np)
    return float(np.clip(0.4*s1 + 0.3*s2 + 0.3*s3, 0, 1))


def forensic_v2(img_np):
    """Returns: forensic_score_v2 ∈ [0,1], diffusion_score ∈ [0,1]. Higher = fake."""
    forensic_classic = forensic_score(img_np)
    diff_score = diffusion_score(img_np)
    perlin_score = diffusion_perlin_residual(img_np)
    texture_noise = texture_noise_score(img_np)
    noiseprint = noiseprint_score(img_np)
    # Prioritize Perlin, texture/noise, and noiseprint for diffusion-based fakes
    forensic_v3 = np.clip(
        0.30 * forensic_classic
        + 0.30 * perlin_score
        + 0.20 * texture_noise
        + 0.20 * noiseprint,
        0,
        1,
    )
    score = float(np.clip(0.4 * forensic_v3 + 0.6 * diff_score, 0, 1))
    return score, diff_score


def texture_noise_score(img_np):
    """Returns a fake score [0,1] based on texture uniformity and high-frequency noise."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Texture uniformity (over-smoothing check)
    patch_size = 32
    vars_ = []
    for y in range(0, gray.shape[0] - patch_size + 1, patch_size):
        for x in range(0, gray.shape[1] - patch_size + 1, patch_size):
            patch = gray[y:y+patch_size, x:x+patch_size]
            vars_.append(np.var(patch))
    texture_var = np.var(vars_) if vars_ else 0.0
    texture_score = 1.0 - min(texture_var / 0.05, 1.0)  # High score for uniform texture
    # High-frequency noise (subtle manipulation check)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    hf_noise = np.var(lap) / (np.mean(np.abs(lap)) + 1e-6)
    noise_score = min(hf_noise / 5.0, 1.0)  # High score for anomalous noise
    return float(np.clip(0.5 * texture_score + 0.5 * noise_score, 0, 1))

# ============================================================
# CLUE 1: Spectral Flatness (Diffusion "Flat Spectrum" Signature)
# ============================================================
def spectral_flatness_score(img_np):
    """High flatness = diffusion-generated (flat PSD in mid-high freq)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    PSD = np.abs(Fshift) ** 2
    PSD += 1e-8

    H, W = PSD.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # Mid-high frequency band (typical diffusion artifact zone)
    rmin = 0.2 * min(H, W)
    rmax = 0.6 * min(H, W)
    mask = (r > rmin) & (r < rmax)
    band = PSD[mask]
    if band.size == 0:
        return 0.0

    # Geometric mean / arithmetic mean = flatness
    gm = np.exp(np.mean(np.log(band)))
    am = np.mean(band)
    flatness = gm / (am + 1e-8)
    score = float(np.clip(1.0 - flatness * 10.0, 0.0, 1.0))
    return score

# ============================================================
# CLUE 2: Edge Continuity Anomaly (AI edges are too clean)
# ============================================================
def edge_continuity_score(img_np):
    """AI edges are unnaturally smooth; real photos have micro-breaks."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    edges = cv2.Canny(gray, 50, 150)
    if edges.sum() == 0:
        return 0.0

    # Dilate to connect nearby edge pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    connected = dilated - edges  # newly connected gaps

    gap_ratio = float(connected.sum()) / (float(edges.sum()) + 1e-8)
    score = float(np.clip(gap_ratio * 3.0, 0.0, 1.0))  # AI tends to fill more gaps
    return score

# ============================================================
# CLUE 3: Color Correlation Drift (AI colors decorrelated)
# ============================================================
def color_correlation_score(img_np):
    """AI often breaks natural RGB channel correlations."""
    img = img_np.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    h, w = r.shape
    patch_size = 32
    corrs = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            pr = r[y:y+patch_size, x:x+patch_size].reshape(-1)
            pg = g[y:y+patch_size, x:x+patch_size].reshape(-1)
            pb = b[y:y+patch_size, x:x+patch_size].reshape(-1)

            if pr.size < 4:
                continue

            corr_rg = np.corrcoef(pr, pg)[0, 1]
            corr_rb = np.corrcoef(pr, pb)[0, 1]
            corr_gb = np.corrcoef(pg, pb)[0, 1]

            if not (np.isnan(corr_rg) or np.isnan(corr_rb) or np.isnan(corr_gb)):
                corrs.append((corr_rg + corr_rb + corr_gb) / 3.0)

    if not corrs:
        return 0.0
    mean_corr = float(np.mean(corrs))
    score = float(np.clip(1.0 - (mean_corr - 0.3) * 2.0, 0.0, 1.0))
    return score

# ============================================================
#             CFA / Bayer Pattern Fake Detector
# ============================================================

def cfa_bayer_score(img_np):
    """
    Returns a FAKE score from 0 to 1.
    Higher = more fake (no CFA pattern detected).
    """
    y = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = y[2:, 2:] - y[:-2, :-2]
    ad = np.abs(diff)
    periodicity = np.mean(ad)
    score = (periodicity - 5) / 15.0
    score = float(np.clip(score, 0.0, 1.0))
    return score

# ============================================================
#                    CORAL ORDINAL CALIBRATION (NEW)
# ============================================================

def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p/(1-p))

class CoralCalibrator:
    def __init__(self):
        if CORAL_CUTS:
            cuts = [
                _logit(CORAL_CUTS["q25"]),
                _logit(CORAL_CUTS["q50"]),
                _logit(CORAL_CUTS["q75"]),
                _logit(CORAL_CUTS["max"]),
            ]
        else:
            cuts = [_logit(0.32), _logit(0.47), _logit(0.61), _logit(0.75)]
        self.c = torch.tensor(cuts, dtype=torch.float32)

    @torch.no_grad()
    def probs(self, z_scaled):
        g = torch.sigmoid(z_scaled - self.c)
        K = g.numel() + 1
        p = torch.zeros(K, dtype=torch.float32)
        p[0] = 1.0 - g[0]
        for k in range(1, K-1):
            p[k] = g[k-1] - g[k]
        p[K-1] = g[-1]
        return p / (p.sum() + 1e-8)

    @torch.no_grad()
    def predict(self, z_scaled):
        p = self.probs(z_scaled)
        idx = int(torch.argmax(p).item())
        return idx, p

CORAL = CoralCalibrator()
RISK_NAMES = ["REAL","LEAN_REAL","BORDERLINE","LEAN_FAKE","FAKE"]

# ============================================================
#     FALSE-POSITIVE PATCH — SOFT CORAL + PATCH NORMALIZER
# ============================================================

def stabilized_fusion(raw, coral, v, f, max_patch, patch_mean):
    """False-Positive Shield: safer fusion for REAL images"""
    spread = max_patch - patch_mean
    # Wavelet / PRNU / lighting safety
    if f < 0.55 and v < 0.55 and spread < 0.18:
        coral *= 0.40
        raw *= 0.80
    # Patch spike soften
    if max_patch > 0.90 and f < 0.55:
        max_patch *= 0.75
        raw *= 0.90
        coral *= 0.60
    # safer blend
    final = (0.55 * raw) + (0.45 * coral)
    # hard REAL guard
    if f < 0.45 and v < 0.50:
        final *= 0.65
    return float(np.clip(final, 0.0, 1.0))

# ============================================================
#            CORE SIGNAL FUSION (SigLIP + Freq + CORAL)
# ============================================================

def detect_core(pil, siglip, freq_mlp, multicrop=True):
    """
    Run SigLIP + frequency MLP + CORAL on a single PIL image,
    and return a dict of logits/probs/fusion signals.
    This is the core detection used by both global and patch-level analysis.
    """
    # ---------- SigLIP + freq logits ----------
    if multicrop:
        crops, weights = make_multicrops(pil)
        x_batch = torch.stack([preprocess(c) for c in crops], dim=0).to(DEVICE)
        z_sigs = siglip(x_batch).detach().cpu()

        f_batch = torch.stack(
            [extract_freq_vector(c) for c in crops], dim=0
        ).to(FREQ_DEVICE)
        z_freqs = freq_mlp(f_batch).detach().cpu()

        z_sig  = float((z_sigs * weights).sum().item())
        z_freq = float((z_freqs * weights).sum().item())
    else:
        x = preprocess(pil).unsqueeze(0).to(DEVICE)
        z_sig = float(siglip(x).item())
        fvec = extract_freq_vector(pil).unsqueeze(0).to(FREQ_DEVICE)
        z_freq = float(freq_mlp(fvec).item())

    # Head-wise probabilities for diagnostics
    p_sig = float(torch.sigmoid(torch.tensor(z_sig)).item())
    p_freq = float(torch.sigmoid(torch.tensor(z_freq / FREQ_TEMP)).item())

    # ---------- Fusion: required simple 2->1 head on probabilities ----------
    head = get_fusion_head()
    x_fuse = torch.tensor([[p_sig, p_freq]], dtype=torch.float32, device=DEVICE)
    z_tensor = head(x_fuse).detach().cpu().squeeze(0)
    z = float(z_tensor.item())

    # ---------- Raw logit-based probability ----------
    z_scaled = float(z) / max(CORAL_TEMP, 1e-3)   # <<< use calibrated temperature
    p_fake_raw = float(torch.sigmoid(torch.tensor(z_scaled)).item())

    # ---------- CORAL ordinal head ----------
    risk_idx, risk_probs = CORAL.predict(torch.tensor(z_scaled))

    # Turn CORAL distribution into a smoother fake probability
    risk_vec = torch.arange(5, dtype=torch.float32)
    mu  = float((risk_vec * risk_probs).sum().item())          # mean in [0,4]
    var = float((risk_probs * (risk_vec - mu) ** 2).sum().item())
    p_coral_gauss = max(0.0, min(1.0, (mu / 4.0) + 0.5 * var))

    # CORAL entropy (peaked vs uncertain)
    entropy = float(-(risk_probs * torch.log(risk_probs + 1e-8)).sum().item())

    # Probability-space fusion (MoE) for diagnostics (or when no head)
    if head is None:
        # MoE fallback: still allow CORAL, but never let it dominate
        p_or = 1.0 - (1.0 - p_sig) * (1.0 - p_freq)
        alpha = p_sig  * (1.0 - p_freq)
        beta  = p_freq * (1.0 - p_sig)
        alpha = max(0.05, min(0.95, alpha))
        beta  = max(0.05, min(0.95, beta))
        p_moe = (alpha * p_sig + beta * p_freq) / (alpha + beta + 1e-6)
        # CORAL is only 25% of the blend now
        p_blend = 0.4 * p_or + 0.35 * p_moe + 0.25 * p_coral_gauss
    else:
        # Conservative: raw model is primary, CORAL is a gentle correction
        # (Real-bias: CORAL cannot push a borderline image to extreme FAKE alone)
        p_blend = float(0.70 * p_fake_raw + 0.30 * p_coral_gauss)

    p_blend = max(0.0, min(1.0, p_blend))

    return {
        "z_sig": z_sig,
        "z_freq": z_freq,
        "z_scaled": z_scaled,
        "p_fake_raw": p_fake_raw,
        "p_fake_coral": p_coral_gauss,
        "p_blend": p_blend,           # main base probability
        "visual_prob": float(torch.sigmoid(torch.tensor(z_sig)).item()),
        "freq_prob": float(torch.sigmoid(torch.tensor(z_freq / FREQ_TEMP)).item()),
        "p_or": None,
        "p_moe": None,
        "risk_idx": risk_idx,
        "risk_probs": risk_probs,
        "entropy": entropy,
    }

# ============================================================
#          MULTICROP + FLIP TTA + JITTER PREVIEW
# ============================================================

def make_multicrops(pil):
    w,h = pil.size
    w2,h2 = max(1,w//2), max(1,h//2)
    crops = [
        pil,
        pil.resize((IMG_SIZE,IMG_SIZE), Image.BICUBIC),
        pil.crop((0,0,w2,h2)),
        pil.crop((w2,0,w,h2)),
        pil.crop((0,h2,w2,h)),
        pil.crop((w2,h2,w,h)),
    ]
    weights = torch.tensor([0.4,0.4,0.05,0.05,0.05,0.05])
    return crops, weights


def jitter_augment(pil, n=4):
    w,h = pil.size
    out = []
    for _ in range(n):
        angle = np.random.uniform(-2,2)
        tx = np.random.uniform(-0.02,0.02)*w
        ty = np.random.uniform(-0.02,0.02)*h
        j = pil.rotate(angle, translate=(tx,ty), resample=Image.BICUBIC)
        out.append(j)
    return out


def make_jitter_collage(pil,n=4,cols=2):
    imgs = jitter_augment(pil,n)
    if not imgs:
        return None
    tile_w,tile_h = IMG_SIZE//2, IMG_SIZE//2
    rows = (len(imgs)+cols-1)//cols
    canvas = Image.new("RGB",(cols*tile_w,rows*tile_h),(0,0,0))
    for i,img in enumerate(imgs):
        r,c = divmod(i,cols)
        canvas.paste(img.resize((tile_w,tile_h)), (c*tile_w,r*tile_h))
    return canvas

# ============================================================
#    PATCH-GRID HEATMAP + REGION NAMING (INTERPRETABILITY)
# ============================================================

def compute_patch_grid(pil, siglip, freq_mlp, rows=PATCH_GRID_ROWS, cols=PATCH_GRID_COLS):
    w,h = pil.size
    if w < MIN_SIDE or h < MIN_SIDE:
        return None, []

    pw,ph = max(8,w//cols), max(8,h//rows)
    grid = np.zeros((rows,cols), np.float32)
    all_scores = []

    for r in range(rows):
        for c in range(cols):
            x0,y0 = c*pw, r*ph
            x1 = w if c==cols-1 else min(w,x0+pw)
            y1 = h if r==rows-1 else min(h,y0+ph)
            if x1<=x0 or y1<=y0:
                s=0.0
            else:
                patch = pil.crop((x0,y0,x1,y1))
                res = detect_core(patch, siglip, freq_mlp, multicrop=False)
                # Use CORAL-free fused probability for heatmap
                s = float(res["p_fake_raw"])
            grid[r,c] = s
            all_scores.append(s)

    return grid, all_scores


def normalize_for_heatmap(values):
    values = np.array(values, dtype=np.float32)
    vmin = float(values.min())
    vmax = float(values.max())

    # If all values identical → zero map (prevents all-red)
    if abs(vmax - vmin) < 1e-6:
        return np.zeros_like(values)

    # Linear normalization
    norm = (values - vmin) / (vmax - vmin)

    # Slight contrast boost for better colors
    norm = norm ** 0.7
    return norm


def make_heatmap_overlay(pil, grid):
    if grid is None:
        return None
    w,h = pil.size
    arr = np.asarray(grid, dtype=np.float32)

    # Detect flat patch maps and fade them visually
    if float(arr.std()) < 0.01:
        patches_norm = np.zeros_like(arr)
    else:
        patches_norm = normalize_for_heatmap(arr)

    cmap = plt.get_cmap("jet")
    colored = cmap(patches_norm)[...,:3]
    img = (colored*255).astype(np.uint8)
    heat = Image.fromarray(img).resize((w,h), Image.BILINEAR)
    return Image.blend(pil.convert("RGBA"), heat.convert("RGBA"), alpha=0.45)


def region_name(r,c,rows,cols):
    V = ["top", "upper", "middle", "lower", "bottom"]
    H = ["left", "left-center", "center", "right-center", "right"]
    vr = int((r+0.5)/rows * (len(V)-1))
    hr = int((c+0.5)/cols * (len(H)-1))
    return f"{V[vr]} {H[hr]}"

# ============================================================
#                  JPEG RESIDUAL FORENSICS
# ============================================================

def jpeg_residual_score(pil):
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    comp = Image.open(buf).convert("RGB")

    o = np.asarray(pil.resize((256,256)),dtype=np.float32)/255.0
    c = np.asarray(comp.resize((256,256)),dtype=np.float32)/255.0

    resid = o - c
    resid_t = torch.from_numpy(resid).permute(2,0,1).unsqueeze(0)

    k = torch.tensor([[0,-1,0], [-1,4,-1], [0,-1,0]],dtype=torch.float32)
    k = k.view(1,1,3,3)/(k.abs().sum()+1e-6)

    hf=[]
    for ch in range(3):
        hf.append(nn.functional.conv2d(resid_t[:,ch:ch+1], k, padding=1))
    hf = torch.cat(hf,1)

    return float(hf.abs().mean().item())

# ============================================================
#         EMBEDDING ANOMALY (L2 + Optional Cosine)
# ============================================================

def embedding_anomaly_score(pil, siglip):
    x = preprocess(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = siglip.backbone.encode_image(x)
        f = f / (f.norm(dim=-1,keepdim=True)+1e-6)

    vec = f.cpu().numpy()[0]
    L2 = float(np.linalg.norm(vec))
    L2n = min(1.0, L2/28.0)

    # Ensure we have a calibrated mean embedding if possible
    if MEAN_EMBEDDING is None:
        compute_mean_embedding()

    cos_dev = 0.0
    if MEAN_EMBEDDING is not None:
        denom = (np.linalg.norm(vec) * np.linalg.norm(MEAN_EMBEDDING) + 1e-6)
        cos = float(np.dot(vec, MEAN_EMBEDDING) / denom)
        cos_dev = (1.0 - cos) * 3.0
        cos_dev = min(1.0, max(0.0, cos_dev))

    score = 0.6 * L2n + 0.4 * cos_dev
    return score, L2, cos_dev

# ============================================================
#                      TRAFFIC-LIGHT LOGIC
# ============================================================

BAND_COLORS = {
    "GREEN": "#6ef3a5",
    "YELLOW": "#ffd666",
    "RED": "#ff6b6b",
}

def band_and_risk(label, p_final, forensic_score):
    if label == "FAKE":
        if forensic_score >= 0.75 or p_final >= 0.65:
            return "RED", "HIGH_FAKE"
        else:
            return "YELLOW", "LEAN_FAKE"
    else:
        if p_final <= 0.35 and forensic_score <= 0.55:
            return "GREEN", "LOW_REAL"
        else:
            return "YELLOW", "LEAN_REAL"


def traffic_light_label(label, p_final, forensic_score):
    band, risk = band_and_risk(label, p_final, forensic_score)
    color = BAND_COLORS[band]

    if band == "GREEN":
        text = "GREEN - low real"
    elif band == "YELLOW" and risk == "LEAN_REAL":
        text = "YELLOW - lean real"
    elif band == "YELLOW" and risk == "LEAN_FAKE":
        text = "YELLOW - lean fake"
    else:
        text = "RED - high fake"

    return text, color, band, risk


def is_uncertain(p,risk,p_patch,head_delta):
    return (0.45 <= p <= 0.55) and risk <= 2 and p_patch < 0.6 and head_delta >= 0.25


def is_inconclusive(p, pg, pp, risk, entropy, head_delta):
    return (
        0.40 <= p <= 0.60 and
        0.40 <= pg <= 0.60 and
        pp < 0.75 and
        risk in (1, 2) and
        entropy > 1.0 and
        head_delta >= 0.15
    )

# ============================================================
#            BAYESIAN HELPERS (LIKELIHOOD-BASED FUSION)
# ============================================================

def _clamp01(p, eps=1e-6):
    return float(np.clip(p, eps, 1.0 - eps))


def _odds(p):
    p = _clamp01(p)
    return p / (1.0 - p)


def _from_odds(o):
    return float(o / (1.0 + o))


def bayes_combine(probs, weights, prior=0.5):
    """
    Combine multiple probability signals with a Bayesian product of
    likelihood ratios. Each probability p_i is converted to
        LR_i = (p_i / (1 - p_i)) ** w_i
    and multiplied into the prior odds.
    """
    prior = _clamp01(prior)
    odds_total = prior / (1.0 - prior)

    for p, w in zip(probs, weights):
        if p is None:
            continue
        p = _clamp01(p)
        lr = (p / (1.0 - p)) ** float(w)
        odds_total *= lr

    return _from_odds(odds_total)

# ============================================================
#              CLEAN FINAL DECISION ENGINE (Bayesian)
# ============================================================

def final_decision(
    visual_prob,
    freq_prob,
    fusion_prob,
    coral_prob,
    forensic_score,
    diff_score,
    max_patch,
    patch_mean,
    head_delta,
    spectral_score=0.0,
    edge_score=0.0,
    color_score=0.0,
    face_boost=0.0,
):
    """
    Fully Bayesian hierarchical model:
      Level 1: Core model fake probability (SigLIP + freq + CORAL).
      Level 2: Generator type posterior P(diffusion | evidence).
      Level 3: Mode-specific fake posteriors:
               - P(fake | camera, evidence)
               - P(fake | diffusion, evidence)
      Final:   P(fake | all) = P(diff)*P(fake|diff) + (1-P(diff))*P(fake|cam)
    Everything is done via weighted likelihood ratios rather than ad-hoc
    additive bumps.
    """

    # ---------------------------
    # Normalize core probabilities
    # ---------------------------
    p_vis      = _clamp01(visual_prob)
    p_freq     = _clamp01(freq_prob)
    p_coral    = _clamp01(coral_prob)
    p_forensic = _clamp01(forensic_score)
    p_diff_raw = _clamp01(diff_score)
    p_spec     = _clamp01(spectral_score)
    p_edge     = _clamp01(edge_score)
    p_color    = _clamp01(color_score)

    p_patch_mean = _clamp01(patch_mean if patch_mean is not None else 0.5)
    p_patch_max  = _clamp01(max_patch   if max_patch   is not None else 0.5)

    # ---------------------------
    # Level 1: Core fake signal
    # ---------------------------
    # Prior: in the wild, fraction of fakes is not 50%, be conservative
    prior_core_fake = 0.30

    # SigLIP is strongest; freq good; CORAL is calibration only.
    p_core_fake = bayes_combine(
        probs   = [p_vis, p_freq, p_coral],
        weights = [1.20, 1.00, 0.40],
        prior   = prior_core_fake,
    )

    # ---------------------------
    # Level 2: P(diffusion | evidence)
    # ---------------------------
    # Use diffusion_score + spectral/edge/color as evidence for generator type.
    # Prior: most real-world images are still camera, so diffusion prior is modest.
    prior_diff = 0.30

    p_gen_diff = bayes_combine(
        probs   = [p_diff_raw, p_spec, p_edge, p_color],
        weights = [1.30, 0.80, 0.80, 0.80],
        prior   = prior_diff,
    )

    # Slightly let strong global patch anomalies hint toward synthetic
    p_gen_diff = 0.9 * p_gen_diff + 0.1 * p_patch_mean
    p_gen_diff = _clamp01(p_gen_diff)

    # ---------------------------
    # Level 3A: Camera-mode fake
    # ---------------------------
    # Camera pipeline: trust core + classical forensic + patch mean.
    # Prior fake rate for camera images: quite low.
    prior_cam_fake = 0.20

    p_fake_cam = bayes_combine(
        probs   = [p_core_fake, p_forensic, p_patch_mean],
        weights = [1.00, 0.70, 0.50],
        prior   = prior_cam_fake,
    )

    # ---------------------------
    # Level 3B: Diffusion-mode fake
    # ---------------------------
    # Diffusion pipeline: core + diffusion evidence + patch hotspot.
    # Prior fake rate for diffusion-like images: higher (if it's diffusion-ish,
    # it's more likely to be a fake / synthetic asset).
    prior_diff_fake = 0.60

    p_fake_diff = bayes_combine(
        probs   = [p_core_fake, p_diff_raw, p_spec, p_edge, p_color, p_patch_max],
        weights = [1.00, 1.00, 0.70, 0.70, 0.70, 0.60],
        prior   = prior_diff_fake,
    )

    # ---------------------------
    # Level 4: Mixture over generator type
    # ---------------------------
    p_final = p_gen_diff * p_fake_diff + (1.0 - p_gen_diff) * p_fake_cam

    # ---------------------------
    # Bayesian face-specific tweak
    # ---------------------------
    # We treat strong face-based diffusion evidence as a small odds multiplier.
    if face_boost > 0.0:
        odds = _odds(p_final)
        odds *= (1.0 + min(face_boost, 0.10))  # at most +10% odds bump
        p_final = _from_odds(odds)

    # ---------------------------
    # Head disagreement Real-bias
    # ---------------------------
    # If SigLIP says fake-ish but freq says real and they disagree a lot,
    # we damp the odds a bit toward REAL.
    if head_delta >= 0.35 and freq_prob < 0.40 <= visual_prob:
        odds = _odds(p_final)
        odds *= 0.80  # reduce odds of fake by 20%
        p_final = _from_odds(odds)

    p_final = float(np.clip(p_final, 0.0, 1.0))

    # ---------------------------
    # Threshold for final label
    # ---------------------------
    # REAL-safe: require reasonably high posterior to call FAKE.
    THRESH_FAKE = 0.60

    if p_final >= THRESH_FAKE:
        label = "FAKE"
    else:
        label = "REAL"

    return p_final, label

# ============================================================
#                         SANITY CHECK
# ============================================================

def is_near_constant(pil, tol=5/255):
    arr = np.asarray(pil.resize((64,64)),dtype=np.float32)/255.0
    return arr.std() < tol

# ============================================================
#                     CORE PREDICT FUNCTION
# ============================================================

@spaces.GPU
def predict(image):
    # Ensure device is up-to-date in ZeroGPU contexts
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if image is None:
        return "No image uploaded.", None, None, None

    # --------------------------------------------------------
    #                Decode + basic validations
    # --------------------------------------------------------
    try:
        pil = Image.open(image).convert("RGB").copy()
    except Exception as e:
        return f"Decode error: {e}", None, None, None

    # Reject blank images
    if is_near_constant(pil):
        return (
            "Image appears nearly blank/uniform - insufficient signal.",
            None,
            None,
            None,
        )

    w, h = pil.size
    if min(w, h) < MIN_SIDE:
        return (
            f"Image too small (min side {min(w,h)} px, need >= {MIN_SIDE}).",
            None,
            None,
            None,
        )

    # Downscale very large images
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        w, h = pil.size

    # Resolution / aspect ratio trust scaling
    aspect = max(w, h) / max(1, min(w, h))
    res_factor = 1.0
    if min(w, h) < 128:
        res_factor *= 0.8
    if aspect > 3.0:
        res_factor *= 0.85

    with torch.inference_mode():
        siglip   = get_siglip_model()
        freq_mlp = get_freq_mlp()

        # --------------------------------------------------------
        #               MULTICROP GLOBAL DETECTION
        # --------------------------------------------------------
        base = detect_core(pil, siglip, freq_mlp, multicrop=True)

        # --------------------------------------------------------
        #                   FLIP / EXTRA TTA
        # --------------------------------------------------------
        tta_results = [base]

        # Horizontal flip (always on)
        pil_flip = pil.transpose(Image.FLIP_LEFT_RIGHT)
        base_flip = detect_core(pil_flip, siglip, freq_mlp, multicrop=True)
        tta_results.append(base_flip)

        # Optional extra TTA: vertical flip + 90° rotation
        if DETECT_EXTRA_TTA:
            try:
                pil_vflip = pil.transpose(Image.FLIP_TOP_BOTTOM)
                tta_results.append(detect_core(pil_vflip, siglip, freq_mlp, multicrop=True))
            except Exception:
                pass
            try:
                pil_rot = pil.rotate(90, expand=True)
                tta_results.append(detect_core(pil_rot, siglip, freq_mlp, multicrop=True))
            except Exception:
                pass

        fusion_prob = float(np.mean([r["p_fake_raw"] for r in tta_results]))
        coral_prob = float(np.mean([r["p_fake_coral"] for r in tta_results]))
        # use fused head probability as the global baseline (may be overridden by XGB)
        p_global = fusion_prob

        # --------------------------------------------------------
        #                 PATCH-GRID LOCAL ANALYSIS
        # --------------------------------------------------------
        grid_scores, patch_scores = compute_patch_grid(pil, siglip, freq_mlp)
        p_patch_max   = max(patch_scores) if patch_scores else p_global
        p_patch_mean  = float(np.mean(patch_scores)) if patch_scores else p_global
        p_patch_spread = p_patch_max - p_patch_mean  # signal of localized anomalies

        # --------------------------------------------------------
        #              JPEG Residual (compression mismatch)
        # --------------------------------------------------------
        jpeg_score = jpeg_residual_score(pil)
        jpeg_norm  = min(1.0, jpeg_score / 0.05)
        jpeg_boost = 0.025 * jpeg_norm  # half strength

        # --------------------------------------------------------
        #        Embedding anomaly (L2 + optional cos-dev)
        # --------------------------------------------------------
        embed_score, embed_l2, embed_cos = embedding_anomaly_score(pil, siglip)

        # --------------------------------------------------------
        #            Head disagreement between SigLIP/freq
        # --------------------------------------------------------
        visual_prob = base["visual_prob"]
        freq_prob   = base["freq_prob"]
        head_delta  = abs(visual_prob - freq_prob)

        disagreement_boost = 0.0
        if head_delta >= 0.5:
            disagreement_boost = 0.03
        elif head_delta >= 0.3:
            disagreement_boost = 0.015

        # --------------------------------------------------------
        #     Background-foreground texture inconsistency
        # --------------------------------------------------------
        texture_boost = 0.0
        bg = pil.crop((0, 0, w//3, h//3))
        fg = pil.crop((w//3, h//3, 2*w//3, 2*h//3))
        _, F_bg = fft_features(bg)
        _, F_fg = fft_features(fg)
        bg_var = float(np.var(F_bg))
        fg_var = float(np.var(F_fg)) + 1e-6

        if abs(bg_var - fg_var) > 1.0 * fg_var:
            texture_boost = 0.02

        # --------------------------------------------------------
        #              Sharpness inconsistency (reduced)
        # --------------------------------------------------------
        gray = np.array(pil.convert("L"))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())

        sharp_boost = 0.0
        if sharpness < 30:
            sharp_boost += 0.02
        elif sharpness > 5000:
            sharp_boost += 0.02

        # --------------------------------------------------------
        #          Forensic score (0-1, optional)
        # --------------------------------------------------------
        forensic_val = 0.5          # final fused forensic score
        diff_score = 0.0            # legacy diffusion_score from forensic_v2
        cfa_fake_score = None
        perlin_score = None
        forensic_v2_score = None
        texture_noise = None
        face_boost = 0.0
        spectral_score = 0.0
        edge_score = 0.0
        color_score = 0.0
        if DETECT_USE_FORENSICS:
            try:
                img_np = np.asarray(pil.convert("RGB"))
                forensic_v2_score, diff_score = forensic_v2(img_np)
                perlin_score = diffusion_perlin_residual(img_np)
                cfa_fake_score = cfa_bayer_score(img_np)
                texture_noise = texture_noise_score(img_np)

                # New spectral / edge / color clues
                try:
                    spectral_score = spectral_flatness_score(img_np)
                    edge_score = edge_continuity_score(img_np)
                    color_score = color_correlation_score(img_np)
                except Exception as e:
                    print(f"[clues] error: {e}")
                    spectral_score = edge_score = color_score = 0.0

                # Optional face-specific boost for strong diffusion evidence on faces
                if HAS_FACE:
                    try:
                        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                        faces = FACE_MODEL.get(img_bgr)
                        if len(faces) >= 1:
                            face = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])
                            x0, y0, x1, y1 = [int(v) for v in face.bbox]
                            x0 = max(0, x0); y0 = max(0, y0)
                            x1 = min(img_np.shape[1], x1); y1 = min(img_np.shape[0], y1)
                            if x1 > x0 and y1 > y0:
                                face_crop = img_np[y0:y1, x0:x1]
                                perlin_face = diffusion_perlin_residual(face_crop)
                                if perlin_face > 0.85:
                                    face_boost = 0.12
                                elif perlin_face > 0.70:
                                    face_boost = 0.08
                    except Exception:
                        face_boost = 0.0

                # v3 forensic fusion: classic + diffusion/perlin/texture/noiseprint (forensic_v2)
                # plus CFA Bayer consistency
                forensic_val = forensic_v2_score
                if cfa_fake_score is not None:
                    forensic_val = float(
                        np.clip(
                            0.4 * forensic_v2_score + 0.6 * cfa_fake_score,
                            0.0,
                            1.0,
                        )
                    )
            except Exception:
                forensic_val = 0.5
                diff_score = 0.0
                cfa_fake_score = None
                perlin_score = None
                forensic_v2_score = None
                texture_noise = None
                face_boost = 0.0
                spectral_score = 0.0
                edge_score = 0.0
                color_score = 0.0

        # Fuse new clues into forensic_val as an additional soft boost
        extra_clues = 0.33 * spectral_score + 0.33 * edge_score + 0.34 * color_score
        forensic_val = float(np.clip(forensic_val + 0.3 * extra_clues, 0.0, 1.0))

        # --------------------------------------------------------
        #          Optional XGBoost fusion (v6 features)
        # --------------------------------------------------------
        xgb_prob = None
        try:
            xgb_model, platt = load_xgb_fusion()
            if xgb_model is not None and platt is not None:
                # Core logits and probabilities
                z_sig_val = float(base.get("z_sig", 0.0))
                z_freq_val = float(base.get("z_freq", 0.0))
                z_diff_val = abs(z_sig_val - z_freq_val)
                visual_p = float(visual_prob)
                freq_p = float(freq_prob)

                # Forensic-style score used during training: 0.4*diff_score + 0.6*CFA
                if cfa_fake_score is not None:
                    forensic_train = float(
                        np.clip(0.4 * diff_score + 0.6 * cfa_fake_score, 0.0, 1.0)
                    )
                else:
                    forensic_train = float(np.clip(diff_score, 0.0, 1.0))

                # JPEG residual + embedding anomaly
                jpeg_resid = float(jpeg_score)
                embed_anom = float(embed_score)

                # Patch-grid stats
                pmax = float(p_patch_max)
                pmean = float(p_patch_mean)
                pspread = float(p_patch_spread)

                # Texture / Perlin / CFA and head disagreement
                cfa_val = float(cfa_fake_score) if cfa_fake_score is not None else 0.5
                tex_val = float(texture_noise) if texture_noise is not None else 0.5
                perlin_val = float(perlin_score) if perlin_score is not None else 0.5
                head_d = float(head_delta)

                feat_vec = np.array(
                    [[
                        z_sig_val,           # 1
                        z_freq_val,          # 2
                        z_diff_val,          # 3
                        visual_p,            # 4
                        freq_p,              # 5
                        forensic_train,      # 6
                        float(diff_score),   # 7
                        float(spectral_score),  # 8
                        float(edge_score),      # 9
                        float(color_score),     # 10
                        jpeg_resid,          # 11
                        embed_anom,          # 12
                        pmax,                # 13
                        pmean,               # 14
                        pspread,             # 15
                        cfa_val,             # 16
                        tex_val,             # 17
                        perlin_val,          # 18
                        head_d,              # 19
                    ]],
                    dtype=np.float32,
                )

                dmat = xgb.DMatrix(feat_vec)
                raw_logit = float(xgb_model.predict(dmat)[0])
                a = float(platt.get("a", 1.0))
                b = float(platt.get("b", 0.0))
                xgb_prob = 1.0 / (1.0 + math.exp(-(a * raw_logit + b)))
        except Exception as e:
            print(f"[xgb] Inference error: {e}")

        # Use XGBoost fusion probability as the core fusion_prob if available
        # and keep p_global in sync so UI/JSON reflect the active fusion.
        if xgb_prob is not None:
            fusion_prob = float(xgb_prob)
            p_global = fusion_prob

        # --------------------------------------------------------
        #          Clean final decision (Option D)
        # --------------------------------------------------------
        p_enhanced, label = final_decision(
            visual_prob=visual_prob,
            freq_prob=freq_prob,
            fusion_prob=fusion_prob,
            coral_prob=coral_prob,
            forensic_score=forensic_val,
            diff_score=diff_score,
            max_patch=p_patch_max,
            patch_mean=p_patch_mean,
            head_delta=head_delta,
            spectral_score=spectral_score,
            edge_score=edge_score,
            color_score=color_score,
            face_boost=face_boost,
        )

        p_moe = base["p_moe"] if base["p_moe"] is not None else None

        # --------------------------------------------------------
        #                Uncertain / Inconclusive
        # --------------------------------------------------------
        uncertain = is_uncertain(
            p_enhanced, base["risk_idx"], p_patch_max, head_delta
        )

        inconclusive = is_inconclusive(
            p_enhanced,
            p_global,
            p_patch_max,
            base["risk_idx"],
            base["entropy"],
            head_delta,
        )

        band_text, band_color, band, risk_level = traffic_light_label(
            label, p_enhanced, forensic_val
        )

        if inconclusive:
            label = "INCONCLUSIVE"
            band_text = "INCONCLUSIVE - borderline evidence"
            band_color = "#cccccc"
        elif uncertain:
            band_text = "UNCERTAIN - low confidence"
            band_color = "#cccccc"

        # CFA-driven REAL override: strong Bayer pattern → trust camera
        if cfa_fake_score is not None and cfa_fake_score < 0.15:
            label = "REAL"
            band = "GREEN"
            risk_level = "LOW_REAL"
            band_color = BAND_COLORS[band]
            band_text = "GREEN - low real"

        # Conservative CFA-driven FAKE override:
        # only trigger when CFA is VERY suspicious AND model & forensics agree.
        if (
            cfa_fake_score is not None
            and cfa_fake_score >= 0.85
            and p_enhanced >= 0.70
            and forensic_val >= 0.60
        ):
            label = "FAKE"
            band_text, band_color, band, risk_level = traffic_light_label(
                label, p_enhanced, forensic_val
            )

        certainty = abs(p_enhanced - 0.5) * 2.0

        # --------------------------------------------------------
        #              Frequency-spectrum preview
        # --------------------------------------------------------
        _, F = fft_features(pil)
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(np.log(F + 1e-3), cmap="inferno")
        ax.axis("off")
        sbuf = io.BytesIO()
        plt.tight_layout(pad=0)
        plt.savefig(sbuf, format="png")
        plt.close(fig)
        sbuf.seek(0)
        spectrum_img = Image.open(sbuf)

        # --------------------------------------------------------
        #                    Heatmap overlay
        # --------------------------------------------------------
        heatmap_img = (
            make_heatmap_overlay(pil, grid_scores)
            if grid_scores is not None else spectrum_img
        )

        # --------------------------------------------------------
        #                  Jitter collage preview
        # --------------------------------------------------------
        jitter_img = make_jitter_collage(pil, n=4, cols=2)

        # --------------------------------------------------------
        #            Extract top suspicious regions
        # --------------------------------------------------------
        suspicious_text = ""
        if grid_scores is not None:
            flat = grid_scores.reshape(-1)
            rows, cols = grid_scores.shape
            idxs = np.argsort(-flat)[:3]
            regions = []
            for idx in idxs:
                r, c = divmod(int(idx), cols)
                v = float(flat[idx])
                if v < 0.55:
                    continue
                region = region_name(r, c, rows, cols)
                color = "#ff6b6b" if v > 0.80 else "#ffd666"
                regions.append(
                    f"<span style='color:{color};font-weight:bold'>{region} ({v:.1%})</span>"
                )
            if regions:
                suspicious_text = "<br><b>Suspicious regions:</b> " + ", ".join(regions)

        # --------------------------------------------------------
        #                  HTML SUMMARY OUTPUT
        # --------------------------------------------------------
        bar_color = "#ff6b6b" if label=="FAKE" else "#6ef3a5"
        moe_text = f"{p_moe:.1%}" if p_moe is not None else "n/a"
        forensic_text = f"{forensic_val:.2f}" if forensic_val is not None else "n/a"
        cfa_text = f"{cfa_fake_score:.2f}" if cfa_fake_score is not None else "n/a"
        perlin_text = f"{perlin_score:.2f}" if perlin_score is not None else "n/a"
        texture_noise_text = f"{texture_noise:.2f}" if texture_noise is not None else "n/a"
        spectral_text = f"{spectral_score:.2f}" if spectral_score is not None else "n/a"
        edge_text = f"{edge_score:.2f}" if edge_score is not None else "n/a"
        color_text = f"{color_score:.2f}" if color_score is not None else "n/a"
        certainty_warning = (
            "<br><b>WARNING: Low certainty (<20%) - manual review recommended.</b>"
            if certainty < 0.20
            else ""
        )
        html = (
            f"<span style='color:{bar_color};font-weight:bold'>Prediction: {label}</span>"
            f" &nbsp;|&nbsp; Band: <span style='color:{band_color};font-weight:bold'>{band_text}</span>"
            f"<br>Risk level: {risk_level}"
            f"<br>Global prob: {p_global:.1%} &nbsp;|&nbsp; Final blended: {p_enhanced:.1%}"
            f"<br>Max patch: {p_patch_max:.1%} (mean {p_patch_mean:.1%}, spread {p_patch_spread:.1%})"
            f"<br>Visual head: {visual_prob:.1%} &nbsp;|&nbsp; Freq head: {freq_prob:.1%}"
            f"<br>MoE fusion: {moe_text}"
            f"<br>Forensic score: {forensic_text} (0=real, 1=fake)"
            f"<br>CFA fake score: {cfa_text} (0=real, 1=fake)"
            f"<br>Perlin diffusion score: {perlin_text} (0=real, 1=fake)"
            f"<br>Texture/noise score: {texture_noise_text} (0=real, 1=fake)"
            f"<br>Spectral flatness: {spectral_text} (0=real, 1=fake)"
            f"<br>Edge continuity: {edge_text} (0=real, 1=fake)"
            f"<br>Color correlation: {color_text} (0=real, 1=fake)"
            f"<br>JPEG residual: {jpeg_score:.4f} &nbsp;|&nbsp; Embedding anomaly: {embed_score:.3f}"
            f"<br>Sharpness: {sharpness:.1f} &nbsp;|&nbsp; Head Delta: {head_delta:.3f}"
            f"<br>Certainty: {certainty:.1%}"
            "<br><div style='width:240px;background:#444;height:10px;border-radius:4px;margin-top:4px;'>"
            f"<div style='width:{int(p_enhanced*240)}px;background:{bar_color};height:10px;border-radius:4px;'></div></div>"
            f"{suspicious_text}"
            f"{certainty_warning}"
            "<br><small>Note: This is a forensic risk estimate only - use context & provenance.</small>"
        )

        # --------------------------------------------------------
        #                   JSON SUMMARY REPORT
        # --------------------------------------------------------
        report = {
            "prediction": label,
            "band": band,
            "risk_level": risk_level,
            "final_prob": float(p_enhanced),
            "global_prob": float(p_global),
            "certainty": float(certainty),
            "forensic_score": float(forensic_val) if forensic_val is not None else None,
            "perlin_score": float(perlin_score) if perlin_score is not None else None,
            "texture_noise_score": float(texture_noise) if texture_noise is not None else None,
            "cfa_fake_score": float(cfa_fake_score) if cfa_fake_score is not None else None,
            "spectral_flatness_score": float(spectral_score),
            "edge_continuity_score": float(edge_score),
            "color_correlation_score": float(color_score),
            "patch_max": float(p_patch_max),
            "patch_mean": float(p_patch_mean),
            "visual_head": float(visual_prob),
            "freq_head": float(freq_prob),
            "jpeg_residual": float(jpeg_score),
            "embedding_anomaly": float(embed_score),
            "sharpness": float(sharpness),
            "head_delta": float(head_delta),
            "xgb_fusion_prob": float(xgb_prob) if xgb_prob is not None else None,
        }

        # LLM metrics: align with core fusion + forensic features
        z_sig_val = float(base.get("z_sig", 0.0))
        z_freq_val = float(base.get("z_freq", 0.0))
        z_diff_val = abs(z_sig_val - z_freq_val)

        llm_metrics = {
            "prediction": label,
            "final_prob": float(p_enhanced),
            "global_prob": float(p_global),
            "fusion_prob": float(fusion_prob),
            "coral_prob": float(coral_prob),
            "z_sig": z_sig_val,
            "z_freq": z_freq_val,
            "z_diff": z_diff_val,
            "visual_head": float(visual_prob),
            "freq_head": float(freq_prob),
            "forensic_score": float(forensic_val) if forensic_val is not None else None,
            "diffusion_score": float(diff_score),
            "perlin_score": float(perlin_score) if perlin_score is not None else None,
            "texture_noise_score": float(texture_noise) if texture_noise is not None else None,
            "cfa_fake_score": float(cfa_fake_score) if cfa_fake_score is not None else None,
            "spectral_flatness_score": float(spectral_score),
            "edge_continuity_score": float(edge_score),
            "color_correlation_score": float(color_score),
            "patch_max": float(p_patch_max),
            "patch_mean": float(p_patch_mean),
            "patch_spread": float(p_patch_spread),
            "jpeg_residual": float(jpeg_score),
            "embedding_anomaly": float(embed_score),
            "sharpness": float(sharpness),
            "head_delta": float(head_delta),
            "certainty": float(certainty),
            "xgb_fusion_prob": float(xgb_prob) if xgb_prob is not None else None,
        }

        # LLM explanation (optional, may fall back to a static message)
        explanation = explain_with_deepseek(llm_metrics)

        report_str = _json.dumps(report, indent=2)

        return html, heatmap_img, jitter_img, report_str, explanation

# ============================================================
#                        GRADIO UI
# ============================================================

def clear_all():
    return None, "", None, None, "", ""

with gr.Blocks(css=".output-image {transition:opacity 0.3s;}") as demo:
    gr.Markdown("### Deepfake Detector v4.3.1 - Balanced")
    gr.Markdown(
        "Improved SigLIP + FFT/SRM fusion with balanced sensitivity. "
        "Includes TTA, patch-grid heatmaps, JPEG residual, embedding anomaly, "
        "texture/sharpness checks, and uncertainty states."
    )

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="filepath", label="Upload Image")
            with gr.Row():
                btn_run = gr.Button("Analyze", variant="primary")
                btn_clear = gr.Button("Clear")

        with gr.Column():
            result_out = gr.HTML(label="Prediction")
            explanation_out = gr.Markdown(label="AI Explanation (DeepSeek)")
            with gr.Row():
                heatmap_out = gr.Image(label="Suspicious Region Map")
                jitter_out = gr.Image(label="Jitter Preview")
            with gr.Accordion("JSON Report (detailed)", open=False):
                json_out = gr.Textbox(label=None, lines=8)

    btn_run.click(
        predict,
        inputs=image_in,
        outputs=[result_out, heatmap_out, jitter_out, json_out, explanation_out],
    )
    btn_clear.click(
        fn=clear_all,
        inputs=None,
        outputs=[image_in, result_out, heatmap_out, jitter_out, json_out, explanation_out],
    )

# Optional: guard get_api_info to avoid older gradio_client schema bugs
if hasattr(demo, "get_api_info"):
    _orig_get_api_info = demo.get_api_info

    def _safe_get_api_info(*args, **kwargs):
        try:
            return _orig_get_api_info(*args, **kwargs)
        except TypeError:
            # Fallback: minimal schema to keep /info alive
            return {}

    demo.get_api_info = _safe_get_api_info

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT",7860)))
# paste EVERYTHING from the code block here