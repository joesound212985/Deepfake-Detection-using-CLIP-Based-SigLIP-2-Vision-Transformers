import os, io, math, numpy as np, torch, torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pywt
import cv2
from scipy.ndimage import gaussian_filter
from scipy.special import digamma, gammaln
from scipy.fftpack import dct
from matplotlib.colors import LogNorm
try:
    from skimage.feature import greycomatrix, greycoprops
except Exception:
    greycomatrix = None
    greycoprops = None
from PIL import Image, ImageOps, ImageFile
try:
    # Register AVIF support if available
    import pillow_avif  # type: ignore
    _HAS_AVIF = True
except Exception:
    try:
        import pillow_avif_plugin  # type: ignore
        _HAS_AVIF = True
    except Exception:
        _HAS_AVIF = False
try:
    import imageio.v3 as iio
except Exception:
    iio = None
from torchvision import transforms
try:
    from huggingface_hub import hf_hub_download, login, InferenceClient
except ImportError:
    from huggingface_hub import hf_hub_download, login
    InferenceClient = None
try:
    # OpenAI-compatible client for Hugging Face router
    from openai import OpenAI as _OpenAIClient
except Exception:
    _OpenAIClient = None
try:
    import requests
except Exception:
    requests = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:
    IsotonicRegression = None
    LogisticRegression = None
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
# Allow safely handling very large images (up to ~300M pixels)
Image.MAX_IMAGE_PIXELS = 300_000_000

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
DETECT_MAX_VIDEO_FRAMES = int(os.getenv("DETECT_MAX_VIDEO_FRAMES", "12"))
VIDEO_SCENE_DETECT = os.getenv("VIDEO_SCENE_DETECT", "1").strip() in {"1","true","True"}
VIDEO_ADAPTIVE_SAMPLE = os.getenv("VIDEO_ADAPTIVE_SAMPLE", "1").strip() in {"1","true","True"}
SCENE_DETECT_STRIDE = int(os.getenv("SCENE_DETECT_STRIDE", "4"))
SCENE_DETECT_MAX_SAMPLES = int(os.getenv("SCENE_DETECT_MAX_SAMPLES", "600"))
SCENE_CUT_THRESH = float(os.getenv("SCENE_CUT_THRESH", "0.45"))
ADAPTIVE_SAMPLE_RATIO = float(os.getenv("ADAPTIVE_SAMPLE_RATIO", "0.50"))
DISABLE_TAMPERED = os.getenv("DISABLE_TAMPERED", "0").strip() in {"1","true","True"}
DISABLE_INCONCLUSIVE = os.getenv("DISABLE_INCONCLUSIVE", "0").strip() in {"1","true","True"}
try:
    FINAL_REAL_THRESH = float(os.getenv("FINAL_REAL_THRESH", "0.45"))
except Exception:
    FINAL_REAL_THRESH = 0.45
try:
    FINAL_FAKE_THRESH = float(os.getenv("FINAL_FAKE_THRESH", "0.75"))
except Exception:
    FINAL_FAKE_THRESH = 0.75
if not (0.0 < FINAL_REAL_THRESH < FINAL_FAKE_THRESH < 1.0):
    FINAL_REAL_THRESH, FINAL_FAKE_THRESH = 0.45, 0.75
try:
    FINAL_LOGIT_SHRINK = float(os.getenv("FINAL_LOGIT_SHRINK", "0.85"))
except Exception:
    FINAL_LOGIT_SHRINK = 0.85
FINAL_LOGIT_SHRINK = max(0.5, min(1.0, FINAL_LOGIT_SHRINK))
try:
    CFA_WEIGHT = float(os.getenv("CFA_WEIGHT", "0.6"))
except Exception:
    CFA_WEIGHT = 0.6
CFA_WEIGHT = max(0.0, min(1.0, CFA_WEIGHT))
try:
    SORA_TAMPERED_THRESH = float(os.getenv("SORA_TAMPERED_THRESH", "0.15"))
except Exception:
    SORA_TAMPERED_THRESH = 0.15
try:
    SORA_FAKE_THRESH = float(os.getenv("SORA_FAKE_THRESH", "0.20"))
except Exception:
    SORA_FAKE_THRESH = 0.20
try:
    SORA_MIN_FAKE_PROB = float(os.getenv("SORA_MIN_FAKE_PROB", "0.40"))
except Exception:
    SORA_MIN_FAKE_PROB = 0.40
try:
    SORA_ODDS_LOW = float(os.getenv("SORA_ODDS_LOW", "1.12"))
except Exception:
    SORA_ODDS_LOW = 1.12
try:
    SORA_ODDS_MED = float(os.getenv("SORA_ODDS_MED", "1.25"))
except Exception:
    SORA_ODDS_MED = 1.25
try:
    SORA_ODDS_HIGH = float(os.getenv("SORA_ODDS_HIGH", "1.45"))
except Exception:
    SORA_ODDS_HIGH = 1.45
try:
    IMAGE_GEN_TAMPERED_THRESH = float(os.getenv("IMAGE_GEN_TAMPERED_THRESH", "0.45"))
except Exception:
    IMAGE_GEN_TAMPERED_THRESH = 0.45
try:
    IMAGE_GEN_FAKE_THRESH = float(os.getenv("IMAGE_GEN_FAKE_THRESH", "0.70"))
except Exception:
    IMAGE_GEN_FAKE_THRESH = 0.70
try:
    IMAGE_GEN_MIN_FAKE_PROB = float(os.getenv("IMAGE_GEN_MIN_FAKE_PROB", "0.50"))
except Exception:
    IMAGE_GEN_MIN_FAKE_PROB = 0.50
try:
    IMAGE_GEN_ODDS_LOW = float(os.getenv("IMAGE_GEN_ODDS_LOW", "1.06"))
except Exception:
    IMAGE_GEN_ODDS_LOW = 1.06
try:
    IMAGE_GEN_ODDS_MED = float(os.getenv("IMAGE_GEN_ODDS_MED", "1.12"))
except Exception:
    IMAGE_GEN_ODDS_MED = 1.12
try:
    IMAGE_GEN_ODDS_HIGH = float(os.getenv("IMAGE_GEN_ODDS_HIGH", "1.20"))
except Exception:
    IMAGE_GEN_ODDS_HIGH = 1.20

SORA_TAMPERED_THRESH = float(np.clip(SORA_TAMPERED_THRESH, 0.10, 0.95))
SORA_FAKE_THRESH = float(np.clip(SORA_FAKE_THRESH, 0.20, 0.98))
if SORA_FAKE_THRESH <= SORA_TAMPERED_THRESH:
    SORA_FAKE_THRESH = min(0.98, SORA_TAMPERED_THRESH + 0.20)
SORA_MIN_FAKE_PROB = float(np.clip(SORA_MIN_FAKE_PROB, 0.20, 0.90))
SORA_ODDS_LOW = float(np.clip(SORA_ODDS_LOW, 1.0, 2.0))
SORA_ODDS_MED = float(np.clip(SORA_ODDS_MED, 1.0, 2.5))
SORA_ODDS_HIGH = float(np.clip(SORA_ODDS_HIGH, 1.0, 3.0))
IMAGE_GEN_TAMPERED_THRESH = float(np.clip(IMAGE_GEN_TAMPERED_THRESH, 0.10, 0.95))
IMAGE_GEN_FAKE_THRESH = float(np.clip(IMAGE_GEN_FAKE_THRESH, 0.20, 0.98))
if IMAGE_GEN_FAKE_THRESH <= IMAGE_GEN_TAMPERED_THRESH:
    IMAGE_GEN_FAKE_THRESH = min(0.98, IMAGE_GEN_TAMPERED_THRESH + 0.15)
IMAGE_GEN_MIN_FAKE_PROB = float(np.clip(IMAGE_GEN_MIN_FAKE_PROB, 0.20, 0.90))
IMAGE_GEN_ODDS_LOW = float(np.clip(IMAGE_GEN_ODDS_LOW, 1.0, 2.0))
IMAGE_GEN_ODDS_MED = float(np.clip(IMAGE_GEN_ODDS_MED, 1.0, 2.5))
IMAGE_GEN_ODDS_HIGH = float(np.clip(IMAGE_GEN_ODDS_HIGH, 1.0, 3.0))
SCENE_DETECT_STRIDE = max(1, SCENE_DETECT_STRIDE)
SCENE_DETECT_MAX_SAMPLES = max(50, SCENE_DETECT_MAX_SAMPLES)
SCENE_CUT_THRESH = float(np.clip(SCENE_CUT_THRESH, 0.10, 0.90))
ADAPTIVE_SAMPLE_RATIO = float(np.clip(ADAPTIVE_SAMPLE_RATIO, 0.30, 0.80))

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("[debug] HF_TOKEN detected: True (login ok)")
    except Exception as e:
        print(f"[debug] HF_TOKEN detected but login failed: {e.__class__.__name__}")
else:
    print("[debug] HF_TOKEN detected: False")

# ============================================================
#           OPTIONAL LLM EXPLANATION (HF-first)
# ============================================================

# Default LLM model for explanations (HF router by default, OpenAI-compatible fallback).
LLM_MODEL = (
    os.getenv("LLM_MODEL")
    or os.getenv("HF_LLM_MODEL")
    or "meta-llama/Meta-Llama-3.1-70B-Instruct"
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY") or HF_TOKEN or OPENAI_API_KEY
_has_hf_creds = bool(os.getenv("LLM_API_KEY") or HF_TOKEN)
_default_llm_base = "https://router.huggingface.co/v1" if _has_hf_creds else "https://api.openai.com/v1"
LLM_BASE_URL = (
    os.getenv("LLM_BASE_URL")
    or os.getenv("HF_LLM_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or _default_llm_base
)
LLM_ENABLED = os.getenv("LLM_ENABLED", "1").strip() in {"1","true","True"}
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "220"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))

_openai_llm_client = None
if LLM_ENABLED and _OpenAIClient is not None and LLM_API_KEY:
    try:
        _openai_llm_client = _OpenAIClient(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
        print(f"[llm] OpenAI-compatible client initialized for model: {LLM_MODEL} (base: {LLM_BASE_URL})")
    except Exception as _e:
        _openai_llm_client = None
        print(f"[llm] Failed to init LLM client: {_e.__class__.__name__}")

_hf_inference_client = None
if LLM_ENABLED and InferenceClient is not None and HF_TOKEN:
    try:
        _hf_inference_client = InferenceClient(token=HF_TOKEN, timeout=LLM_TIMEOUT)
        print("[llm] HF InferenceClient ready for fallback.")
    except Exception as _e:
        _hf_inference_client = None
        print(f"[llm] Failed to init HF InferenceClient: {_e.__class__.__name__}")


def _extract_json_from_text(text: str):
    if not text:
        return None
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("json"):
                chunk = "\n".join(chunk.splitlines()[1:]).strip()
            try:
                return _json.loads(chunk)
            except Exception:
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _call_llm_openai(model: str, system_msg: str, user_msg: str):
    if _openai_llm_client is None:
        return None
    try:
        resp = _openai_llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content if resp.choices else ""
    except Exception as e:
        print(f"[llm] OpenAI-compatible call failed: {e.__class__.__name__}")
        return None


def _call_llm_hf(model: str, system_msg: str, user_msg: str):
    if _hf_inference_client is None:
        return None
    # Try HF chat.completions interface when available
    try:
        if hasattr(_hf_inference_client, "chat") and hasattr(_hf_inference_client.chat, "completions"):
            resp = _hf_inference_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            return resp.choices[0].message.content if resp.choices else ""
    except Exception as e:
        print(f"[llm] HF chat call failed: {e.__class__.__name__}")

    # Fallback to text_generation
    prompt = f"{system_msg}\n\n{user_msg}\n\nJSON:"
    try:
        text = _hf_inference_client.text_generation(
            prompt,
            model=model,
            max_new_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=LLM_TEMPERATURE > 0.0,
        )
        if isinstance(text, dict):
            return text.get("generated_text", "")
        if isinstance(text, list) and text:
            return text[0].get("generated_text", "") if isinstance(text[0], dict) else str(text[0])
        return text if isinstance(text, str) else str(text)
    except Exception as e:
        print(f"[llm] HF text generation failed: {e.__class__.__name__}")
        return None


def _heuristic_explanation(metrics: dict) -> str:
    label = metrics.get("prediction") or metrics.get("label_v2") or "INCONCLUSIVE"
    try:
        p_final = float(metrics.get("final_prob", metrics.get("fusion_prob", 0.0)) or 0.0)
    except Exception:
        p_final = 0.0
    try:
        certainty = float(metrics.get("certainty", 0.0) or 0.0)
    except Exception:
        certainty = 0.0

    if certainty >= 0.75:
        conf = "high"
    elif certainty >= 0.45:
        conf = "medium"
    else:
        conf = "low"

    signals = []

    def add_high(key, thresh, msg):
        val = metrics.get(key)
        if val is None:
            return
        try:
            if float(val) >= thresh:
                signals.append(msg)
        except Exception:
            return

    def add_low(key, thresh, msg):
        val = metrics.get(key)
        if val is None:
            return
        try:
            if float(val) <= thresh:
                signals.append(msg)
        except Exception:
            return

    add_high("sora_likelihood", 0.20, "Sora-like temporal drift")
    add_high("image_gen_likelihood", 0.45, "Image generator artifacts")
    add_high("temporal_consistency_score", 0.60, "Temporal inconsistencies")
    add_high("diffusion_score", 0.55, "Diffusion artifacts")
    add_high("forensic_score", 0.60, "Forensic anomalies")
    add_high("cfa_fake_score", 0.55, "CFA mismatch")
    add_high("rendering_pipeline_score", 0.65, "Over-regular rendering")
    add_high("face_retouch_score", 0.55, "Face retouch patterns")
    add_high("esrgan_grid_score", 0.50, "Upscaling grid artifacts")

    if label == "REAL":
        add_high("prnu_strength_scaled", 0.50, "Sensor PRNU consistent")
        add_high("real_image_prior_v4", 0.55, "Camera pipeline appears natural")
        add_low("diffusion_score", 0.35, "Low diffusion artifacts")

    if not signals:
        signals = ["Mixed forensic/visual signals", "No single dominant artifact"]

    signals = signals[:4]
    summary = f"Decision: {label} with {conf} confidence (p(fake) {p_final*100:.1f}%)."
    signals_text = "\n".join(f"- {s}" for s in signals)
    return f"{summary}\n\nKey signals:\n{signals_text}\n\nConfidence: {conf} (heuristic fallback)"


def explain_with_llm(metrics: dict) -> str:
    """
    Use an OpenAI-compatible LLM (HF router by default, Llama 3.1 70B default model)
    to generate a short, human-readable explanation of the detector metrics.
    """
    if not LLM_ENABLED:
        return ""

    metrics_json = _json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=True)
    system_msg = (
        "You are a deepfake forensics assistant. "
        "Return strictly valid JSON only, no markdown."
    )
    user_msg = f"""
Given the following detector metrics JSON, return a JSON object with:
- summary: string (<= 60 words)
- label: REAL | TAMPERED | FAKE | INCONCLUSIVE | UNCERTAIN
- key_signals: array of 2-4 short phrases
- confidence: low | medium | high
Use only the provided metrics. Avoid math formulas.

Metrics JSON:
{metrics_json}
"""

    attempts = []
    if _openai_llm_client is not None:
        attempts.append(("openai", LLM_MODEL))
        if LLM_FALLBACK_MODEL and LLM_FALLBACK_MODEL != LLM_MODEL:
            attempts.append(("openai", LLM_FALLBACK_MODEL))
    if _hf_inference_client is not None:
        attempts.append(("hf", LLM_MODEL))
        if LLM_FALLBACK_MODEL and LLM_FALLBACK_MODEL != LLM_MODEL:
            attempts.append(("hf", LLM_FALLBACK_MODEL))

    text = None
    for provider, model in attempts:
        if provider == "openai":
            text = _call_llm_openai(model, system_msg, user_msg)
        else:
            text = _call_llm_hf(model, system_msg, user_msg)
        if text:
            text = str(text).strip()
            break

    if not text:
        return _heuristic_explanation(metrics)

    # Strip hidden reasoning markers, if present
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    parsed = _extract_json_from_text(text)
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary", "")).strip()
        label = str(parsed.get("label", "")).strip()
        confidence = str(parsed.get("confidence", "")).strip()
        key_signals = parsed.get("key_signals", [])
        if not isinstance(key_signals, list):
            key_signals = []
        if summary:
            lines = [summary]
            if key_signals:
                signals_text = "\n".join(f"- {s}" for s in key_signals[:4])
                lines.append(f"\nKey signals:\n{signals_text}")
            if label or confidence:
                label_text = label or "n/a"
                conf_text = confidence or "n/a"
                lines.append(f"\nLabel: {label_text} | Confidence: {conf_text}")
            return "\n".join(lines).strip()

    return text.strip()

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

CORAL_CUTS, CORAL_TEMP, CORAL_BINS = None, 1.0, None
CORAL_LOCK = threading.Lock()

# ============================================================
#                 DEVICE + CONSTANTS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FREQ_DEVICE = "cpu"   # keeps frequency MLP cheap

# SigLIP WebLI backbone is trained at 384px and the open_clip
# model enforces this input size; keep IMG_SIZE=384 to avoid
# patch_embed assertion errors.
IMG_SIZE = 384
EPS = 1e-6

MIN_SIDE = 64      # Reject tiny images
MAX_SIDE = 2048    # Auto-downscale very large images

# ============================================================
#                    MEDIA LOADERS (IMAGE/VIDEO)
# ============================================================

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}


def _is_video_file(path: str) -> bool:
    ext = os.path.splitext(str(path))[-1].lower()
    return ext in VIDEO_EXTS


def load_image_any(path: str) -> Image.Image:
    """
    Load an image with AVIF support when available; falls back to imageio for AVIF.
    """
    try:
        return Image.open(path).convert("RGB").copy()
    except Exception as e:
        lower = str(path).lower()
        if lower.endswith(".avif") and iio is not None:
            try:
                arr = iio.imread(path)
                return Image.fromarray(arr).convert("RGB")
            except Exception as _e:
                print(f"[decode] imageio AVIF decode failed: {_e}")
        print(f"[decode] PIL load failed: {e}")
        raise


def _quick_gray_hist(gray: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _scan_video_changes(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = SCENE_DETECT_STRIDE
    if total_frames > 0:
        stride = max(stride, int(np.ceil(total_frames / SCENE_DETECT_MAX_SAMPLES)))

    sample_idxs = []
    diffs = []
    prev_hist = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride != 0:
            idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        hist = _quick_gray_hist(gray)
        diff = 0.0 if prev_hist is None else float(
            cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        )
        sample_idxs.append(idx)
        diffs.append(diff)
        prev_hist = hist
        idx += 1
        if len(sample_idxs) >= SCENE_DETECT_MAX_SAMPLES:
            break

    cap.release()
    if total_frames <= 0:
        total_frames = idx
    return total_frames, sample_idxs, diffs


def _pick_primary_scene(total_frames: int, sample_idxs: list, diffs: list):
    if total_frames <= 0:
        if sample_idxs:
            return 0, max(sample_idxs)
        return 0, 0
    if not sample_idxs:
        return 0, total_frames - 1

    diffs_arr = np.asarray(diffs, dtype=np.float32)
    dyn_thresh = float(np.median(diffs_arr) + 2.0 * np.std(diffs_arr))
    cut_thresh = max(SCENE_CUT_THRESH, dyn_thresh)
    min_len = max(8, SCENE_DETECT_STRIDE * 2)

    segments = []
    start = 0
    for idx, diff in zip(sample_idxs, diffs):
        if diff >= cut_thresh and (idx - start) >= min_len:
            segments.append((start, max(start, idx - 1)))
            start = idx
    segments.append((start, total_frames - 1))
    segments = [seg for seg in segments if seg[1] >= seg[0]]
    if not segments:
        return 0, total_frames - 1
    return max(segments, key=lambda seg: seg[1] - seg[0])


def _adaptive_sample_indices(
    start_idx: int,
    end_idx: int,
    sample_idxs: list,
    diffs: list,
    max_frames: int,
):
    if end_idx < start_idx or max_frames <= 0:
        return []

    seg_len = end_idx - start_idx + 1
    max_frames = min(max_frames, seg_len)
    n_uniform = max(2, int(np.ceil(max_frames * ADAPTIVE_SAMPLE_RATIO)))
    uniform = np.linspace(start_idx, end_idx, num=min(n_uniform, seg_len), dtype=int).tolist()
    selected = set(int(i) for i in uniform)
    n_adaptive = max_frames - len(selected)

    candidates = [
        (int(idx), float(diff))
        for idx, diff in zip(sample_idxs, diffs)
        if start_idx <= idx <= end_idx
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    min_gap = max(1, int(seg_len / max_frames / 1.5))

    if n_adaptive > 0 and candidates:
        for idx, _diff in candidates:
            if len(selected) >= max_frames:
                break
            if all(abs(idx - s) >= min_gap for s in selected):
                selected.add(idx)

    if len(selected) < max_frames:
        filler = np.linspace(start_idx, end_idx, num=max_frames, dtype=int).tolist()
        for idx in filler:
            if len(selected) >= max_frames:
                break
            selected.add(int(idx))

    return sorted(selected)


def extract_video_frames(
    video_path: str,
    max_frames: int = DETECT_MAX_VIDEO_FRAMES,
    scene_detect=None,
    adaptive_sample=None,
):
    """
    Sample up to max_frames evenly spaced frames from a video and return PIL RGB images.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[video] Failed to open video file")
        return frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = None
    scene_start = 0
    scene_end = max(0, total_frames - 1)
    sample_idxs = []
    diffs = []

    if scene_detect is None:
        scene_detect = VIDEO_SCENE_DETECT
    if adaptive_sample is None:
        adaptive_sample = VIDEO_ADAPTIVE_SAMPLE
    scene_detect = bool(scene_detect)
    adaptive_sample = bool(adaptive_sample)

    if scene_detect or adaptive_sample:
        total_frames, sample_idxs, diffs = _scan_video_changes(video_path)
        if scene_detect:
            scene_start, scene_end = _pick_primary_scene(total_frames, sample_idxs, diffs)

    if adaptive_sample and total_frames > 0:
        idxs = _adaptive_sample_indices(scene_start, scene_end, sample_idxs, diffs, max_frames)
    if not idxs:
        if total_frames > 0:
            idxs = np.linspace(scene_start, scene_end, num=min(max_frames, scene_end - scene_start + 1), dtype=int)
        else:
            idxs = list(range(max_frames))

    for idx in idxs:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        except Exception as _e:
            print(f"[video] frame {idx} decode error: {_e}")
            continue

    if not frames:
        # Fallback: sequential read until we have at least 1 frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            if len(frames) >= max_frames:
                break

    cap.release()
    return frames


def aggregate_video_probs(
    probs: np.ndarray,
    frame_preds: list,
    agg_mode: str = "topk_mean",
    topk_frac: float = 0.30,
    strictness: str = "balanced",
    min_agree: int = 2,
    weights=None,
):
    """
    Returns: (video_prob, video_label, chosen_frame_index, metrics_dict)
    """
    probs = np.asarray(probs, dtype=np.float32)
    n = int(probs.size)
    if n == 0:
        return 0.5, "INCONCLUSIVE", 0, {"n": 0}

    weights_arr = None
    if weights is not None:
        try:
            weights_arr = np.asarray(weights, dtype=np.float32)
            if int(weights_arr.size) != n:
                weights_arr = None
            else:
                weights_arr = np.clip(weights_arr, 0.05, None)
        except Exception:
            weights_arr = None

    # thresholds by strictness
    if strictness == "conservative":
        th_fake, th_real = 0.65, 0.35
    elif strictness == "aggressive":
        th_fake, th_real = 0.55, 0.45
    else:
        th_fake, th_real = 0.60, 0.40

    topk_frac = float(np.clip(topk_frac, 0.05, 1.0))
    k = max(1, int(np.ceil(topk_frac * n)))
    srt = np.sort(probs)
    weighted_median_idx = None

    if weights_arr is None:
        if agg_mode == "max":
            video_prob = float(srt[-1])
        elif agg_mode == "median":
            video_prob = float(np.median(probs))
        else:
            # default: robust top-k mean
            video_prob = float(np.mean(srt[-k:]))
    else:
        scores = probs * weights_arr
        if agg_mode == "max":
            idx = int(np.argmax(scores))
            video_prob = float(probs[idx])
        elif agg_mode == "median":
            order = np.argsort(probs)
            cumw = np.cumsum(weights_arr[order])
            cutoff = 0.5 * float(cumw[-1])
            median_pos = int(np.searchsorted(cumw, cutoff))
            weighted_median_idx = int(order[min(median_pos, n - 1)])
            video_prob = float(probs[weighted_median_idx])
        else:
            top_idx = np.argsort(scores)[-k:]
            video_prob = float(np.average(probs[top_idx], weights=weights_arr[top_idx]))

    video_std = float(np.std(probs))
    n_fake = int(np.sum(probs >= th_fake))
    n_real = int(np.sum(probs <= th_real))

    # count model labels
    counts = {"REAL": 0, "TAMPERED": 0, "FAKE": 0, "INCONCLUSIVE": 0, "UNCERTAIN": 0}
    for p in frame_preds:
        if p in counts:
            counts[p] += 1
        else:
            counts["INCONCLUSIVE"] += 1

    # conservative label rules
    if (video_prob >= th_fake and n_fake >= min_agree) or counts["FAKE"] >= min_agree:
        video_label = "FAKE"
    elif counts["TAMPERED"] >= min_agree and counts["FAKE"] == 0:
        video_label = "TAMPERED"
    elif (video_prob <= th_real and n_real >= min_agree) and counts["FAKE"] == 0:
        video_label = "REAL"
    else:
        video_label = "INCONCLUSIVE"

    if DISABLE_TAMPERED and video_label == "TAMPERED":
        video_label = "FAKE" if video_prob >= th_fake else "REAL"
    if DISABLE_INCONCLUSIVE and video_label in ("INCONCLUSIVE", "UNCERTAIN"):
        video_label = "FAKE" if video_prob >= th_fake else "REAL"

    image_level_p_fake = float(np.max(probs))
    if image_level_p_fake > 0.75:
        video_label = "FAKE"
        video_prob = max(video_prob, image_level_p_fake * 0.9)

    # choose display frame
    score_for_pick = probs if weights_arr is None else probs * weights_arr
    if video_label == "FAKE":
        chosen = int(np.argmax(score_for_pick))
    elif video_label == "REAL":
        if weighted_median_idx is not None:
            chosen = int(weighted_median_idx)
        else:
            chosen = int(np.argmin(np.abs(probs - np.median(probs))))
    elif video_label == "TAMPERED":
        # prefer a frame that the image model calls TAMPERED
        idxs = [i for i, lab in enumerate(frame_preds) if lab == "TAMPERED"]
        if idxs:
            local_scores = score_for_pick[idxs]
            chosen = int(idxs[int(np.argmax(local_scores))])
        else:
            chosen = int(np.argmin(np.abs(probs - video_prob)))
    else:
        chosen = int(np.argmin(np.abs(probs - video_prob)))

    metrics = {
        "n": n,
        "k": k,
        "agg_mode": agg_mode,
        "topk_frac": float(topk_frac),
        "video_prob": float(video_prob),
        "video_std": float(video_std),
        "th_fake": float(th_fake),
        "th_real": float(th_real),
        "n_fake_frames": int(n_fake),
        "n_real_frames": int(n_real),
        "label_counts": counts,
    }
    if weights_arr is not None:
        metrics["weights_used"] = True
        metrics["weights_summary"] = {
            "min": float(weights_arr.min()),
            "max": float(weights_arr.max()),
            "mean": float(weights_arr.mean()),
        }
    return video_prob, video_label, chosen, metrics

# ============================================================
#                BAYESIAN FUSION (CALIBRATION-READY)
# ============================================================

class BayesianFusionV2:
    """
    Adaptive Bayesian fusion for multi-cue deepfake detection.
    Combines vision, frequency, and forensic heads into a calibrated posterior.
    """

    def __init__(self, calibrate=False, logistic=False):
        self.calibrate = calibrate
        self.logistic = logistic
        self.iso_models = {}
        self.logit_model = None
        self.feature_names = [
            "visual", "freq", "forensic", "cfa",
            "jpeg", "prnu", "patch"
        ]

    def fit_calibration(self, X, y):
        """
        Fit calibration models for feature reliability.
        X: ndarray or dict of feature arrays
        y: 0=real, 1=fake
        """
        if isinstance(X, dict):
            X = np.column_stack([X[k] for k in self.feature_names if k in X])

        if self.logistic:
            if LogisticRegression is None:
                raise ImportError("scikit-learn is required for logistic calibration.")
            self.logit_model = LogisticRegression(max_iter=500)
            self.logit_model.fit(X, y)
        elif self.calibrate:
            if IsotonicRegression is None:
                raise ImportError("scikit-learn is required for isotonic calibration.")
            for i, name in enumerate(self.feature_names):
                self.iso_models[name] = IsotonicRegression(
                    out_of_bounds="clip"
                ).fit(X[:, i], y)

    def calibrate_feature(self, name, value):
        if self.logit_model is not None:
            return value
        if name in self.iso_models:
            return float(self.iso_models[name].predict([value])[0])
        return value

    def _get_reliability(self, features):
        return {
            "visual": 0.6,
            "freq": 0.6,
            "forensic": 1.0,
            "cfa": 0.9,
            "jpeg": 0.7,
            "prnu": 0.7,
            "patch": 0.8,
        }

    def fuse(self, features, prior_fake=0.5):
        eps = 1e-6
        # Fill missing features with neutral priors
        base = {}
        for name in self.feature_names:
            v = features.get(name, 0.5)
            try:
                if v is None or not np.isfinite(v):
                    v = 0.5
            except Exception:
                v = 0.5
            base[name] = float(v)

        calibrated = {
            k: np.clip(self.calibrate_feature(k, v) if self.calibrate else v, eps, 1 - eps)
            for k, v in base.items()
        }
        reliability = self._get_reliability(calibrated)

        log_odds_sum = 0.0
        for k, p in calibrated.items():
            w = reliability.get(k, 1.0)
            log_term = w * np.log(p / (1 - p))
            log_odds_sum += float(np.clip(log_term, -2.0, 2.0))

        prnu_fake = calibrated.get("prnu", 0.5)
        if prnu_fake < 0.4:
            log_odds_sum += math.log(0.5)

        prior_fake = float(np.clip(prior_fake, eps, 1 - eps))
        log_prior = np.log(prior_fake / (1 - prior_fake))
        log_post = log_odds_sum + log_prior
        posterior_fake = 1 / (1 + np.exp(-log_post))

        vals = np.array(list(calibrated.values()))
        mean_p, std_p = np.mean(vals), np.std(vals)
        agreement = 1 - np.tanh(std_p * 2)
        certainty = np.clip(agreement * (0.5 + abs(0.5 - posterior_fake) * 2), 0, 1)

        return dict(
            posterior_fake=float(posterior_fake),
            posterior_real=float(1 - posterior_fake),
            certainty=float(certainty),
            log_odds=float(log_post),
            calibrated=calibrated,
            reliability=reliability,
            mean_prob=float(mean_p),
            std_prob=float(std_p),
        )

# ============================================================
#           DIRICHLET BAYESIAN FUSION (EVIDENCE-BASED)
# ============================================================

class DirichletBayesianFusion:
    """
    Bayesian fusion using Dirichlet evidence accumulation.
    Each feature contributes pseudo-counts (alpha_real, alpha_fake).
    """

    def __init__(self, base_strength=3.0):
        self.base_strength = base_strength
        self.feature_weights = {
            "visual": 1.0,
            "freq": 1.0,
            "forensic": 1.2,
            "cfa": 0.9,
            "jpeg": 0.8,
            "prnu": 0.7,
            "patch": 0.9,
        }

    def evidence_from_prob(self, p, w=1.0):
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        alpha_fake = w * (p * self.base_strength) + 1
        alpha_real = w * ((1 - p) * self.base_strength) + 1
        return np.array([alpha_real, alpha_fake], dtype=np.float32)

    def fuse(self, features):
        total_alpha = np.zeros(2, dtype=np.float32)

        for k, w in self.feature_weights.items():
            if k not in features:
                continue
            alpha = self.evidence_from_prob(features[k], w)
            total_alpha += alpha

        S = float(np.sum(total_alpha))
        alpha_real, alpha_fake = float(total_alpha[0]), float(total_alpha[1])

        mean_fake = alpha_fake / S
        mean_real = alpha_real / S

        epistemic = 2.0 / S  # smaller when more evidence
        aleatoric = mean_fake * (1 - mean_fake)
        uncertainty = float(np.clip(epistemic + aleatoric, 0.0, 1.0))
        conflict = float(np.abs(alpha_real - alpha_fake) / S)

        return dict(
            posterior_fake=float(mean_fake),
            posterior_real=float(mean_real),
            alpha_real=alpha_real,
            alpha_fake=alpha_fake,
            total_strength=S,
            uncertainty=uncertainty,
            conflict=conflict,
        )

# ============================================================
#        DIRICHLET TRIANGLE VISUALIZATION (OPTIONAL)
# ============================================================

def plot_dirichlet_triangle(dirichlet_result, title="Dirichlet Fusion Posterior", show=False):
    """
    Visualize Real / Fake / Uncertainty relationship as a barycentric triangle.
    Returns a matplotlib figure; set show=True to display inline (blocking).
    """
    try:
        fake = float(dirichlet_result.get("posterior_fake", 0.5))
        real = float(dirichlet_result.get("posterior_real", 0.5))
        unc = float(dirichlet_result.get("uncertainty", 0.1))
    except Exception:
        fake, real, unc = 0.5, 0.5, 0.1

    s = max(fake + real + unc, 1e-6)
    fake /= s
    real /= s
    unc /= s

    vertices = np.array([
        [0.5, math.sqrt(3) / 2.0],  # FAKE (top)
        [0.0, 0.0],                 # REAL (left)
        [1.0, 0.0],                 # UNCERTAINTY (right)
    ])

    p = fake * vertices[0] + real * vertices[1] + unc * vertices[2]

    fig, ax = plt.subplots(figsize=(5, 5))
    triangle = plt.Polygon(vertices, closed=True, fill=None, edgecolor="gray", lw=1.5)
    ax.add_patch(triangle)

    ax.text(0.5, math.sqrt(3) / 2.0 + 0.05, "FAKE", ha="center", va="bottom", fontsize=11, color="crimson")
    ax.text(-0.05, -0.05, "REAL", ha="right", va="top", fontsize=11, color="seagreen")
    ax.text(1.05, -0.05, "UNCERTAINTY", ha="left", va="top", fontsize=11, color="steelblue")

    ax.scatter(p[0], p[1], s=120, color="gold", edgecolor="black", zorder=5)
    ax.text(
        p[0],
        p[1] + 0.04,
        f"Fake={fake:.2f}\\nReal={real:.2f}\\nUnc={unc:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.1, math.sqrt(3) / 2.0 + 0.2)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")

    if show:
        plt.show()
    return fig

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

# Final decision threshold for p_final (simple rule)
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

def extract_prnu(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Extract PRNU sensor noise from a BGR or RGB image.
    Returns a 2D float32 map, zero-mean, unit-ish variance.
    """
    if image.dtype != np.float32:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.copy()

    # Convert to grayscale (PRNU is luminance-dominant)
    if img.ndim == 3:
        # img_np in this app is RGB; if BGR is passed, the effect is minor
        gray_u8 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = gray_u8.astype(np.float32) / 255.0
    else:
        gray = img

    # Wavelet-like denoising via Gaussian smoothing
    smooth = gaussian_filter(gray, sigma)
    noise = gray - smooth
    noise -= float(noise.mean())
    noise /= (float(noise.std()) + 1e-8)
    return noise.astype(np.float32)

def prnu_consistency_score(img_np: np.ndarray) -> float:
    prnu = extract_prnu(img_np)
    return float(np.var(prnu.flatten()))

# ============================================================
#   CROSS-FRAME PRNU INCOHERENCE (ANTI-SORA)
# ============================================================

def prnu_temporal_incoherence(frames):
    """
    Real sensors: PRNU correlates across frames
    Sora: noise injected per-frame
    """
    if frames is None or len(frames) < 3:
        return 0.0

    prnus = []
    for f in frames:
        img = np.asarray(f)
        prnus.append(extract_prnu(img))

    corrs = []
    for i in range(len(prnus) - 1):
        a = prnus[i].flatten()
        b = prnus[i + 1].flatten()
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr):
            corrs.append(corr)

    if not corrs:
        return 0.0

    mean_corr = float(np.mean(corrs))

    # Real cameras ~ 0.4-0.7, Sora ~ 0.0-0.2
    return float(np.clip((0.35 - mean_corr) / 0.35, 0.0, 1.0))


def prnu_temporal_incoherence_flat(frames):
    """
    PRNU correlation in low-texture regions after denoising.
    Helps when global PRNU is masked by noise.
    """
    if frames is None or len(frames) < 3:
        return 0.0

    prnus = []
    masks = []
    for f in frames:
        try:
            img = np.asarray(f)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            try:
                den = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
                den_f = den.astype(np.float32) / 255.0
                prnu = extract_prnu(den_f)
            except Exception:
                prnu = extract_prnu(gray)

            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.magnitude(gx, gy)
            mask = grad < 8.0
            if float(np.mean(mask)) < 0.05:
                mask = None

            prnus.append(prnu)
            masks.append(mask)
        except Exception:
            continue

    if len(prnus) < 2:
        return 0.0

    corrs = []
    for i in range(len(prnus) - 1):
        m0 = masks[i]
        m1 = masks[i + 1]
        mask = m0 & m1 if (m0 is not None and m1 is not None) else None
        if mask is not None and float(np.mean(mask)) >= 0.02:
            a = prnus[i][mask].flatten()
            b = prnus[i + 1][mask].flatten()
        else:
            a = prnus[i].flatten()
            b = prnus[i + 1].flatten()
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr):
            corrs.append(corr)

    if not corrs:
        return 0.0

    mean_corr = float(np.mean(corrs))
    return float(np.clip((0.30 - mean_corr) / 0.30, 0.0, 1.0))


def prnu_strength(noise: np.ndarray) -> float:
    """
    Scalar PRNU strength; real sensors show stable non-zero values.
    """
    return float(np.mean(np.abs(noise)))


def jpeg_block_consistency(img_np: np.ndarray) -> float:
    """JPEG 8x8 block variance consistency; higher → more real."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    blk = []
    for y in range(0, h - 8, 8):
        for x in range(0, w - 8, 8):
            patch = gray[y:y+8, x:x+8].astype(np.float32)
            blk.append(float(np.var(patch)))
    if not blk:
        return 0.0
    blk = np.array(blk, dtype=np.float32)
    return float(1.0 - min(np.std(blk) / 50.0, 1.0))


def jpeg_block_drift(frames):
    """
    Measures drift in JPEG block grid statistics across frames.
    Lower correlation -> more suspicious compression inconsistency.
    """
    if frames is None or len(frames) < 3:
        return 0.0

    maps = []
    for f in frames:
        try:
            gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
            h8 = h - (h % 8)
            w8 = w - (w % 8)
            if h8 < 16 or w8 < 16:
                continue
            g = gray[:h8, :w8].astype(np.float32)
            blocks = g.reshape(h8 // 8, 8, w8 // 8, 8)
            blocks = blocks.swapaxes(1, 2)
            var_map = blocks.var(axis=(2, 3))
            var_map = var_map - float(var_map.mean())
            var_map = var_map / (float(var_map.std()) + 1e-6)
            maps.append(var_map)
        except Exception:
            continue

    if len(maps) < 2:
        return 0.0

    corrs = []
    for i in range(len(maps) - 1):
        a = maps[i].flatten()
        b = maps[i + 1].flatten()
        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr):
            corrs.append(corr)

    if not corrs:
        return 0.0

    mean_corr = float(np.mean(corrs))
    return float(np.clip((0.40 - mean_corr) / 0.40, 0.0, 1.0))


def highlight_clipping_realness(img_np: np.ndarray) -> float:
    """Highlight clipping prior; higher → more real."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    bright = np.mean(gray > 245)
    return float(min(bright / 0.05, 1.0))


def crop_consistency_score(pil: Image.Image) -> float:
    """Random crop variance stability; higher → more real."""
    img = np.asarray(pil)
    h, w, _ = img.shape
    if h < 4 or w < 4:
        return 0.0
    scores = []
    for _ in range(8):
        y = np.random.randint(0, max(1, h // 3))
        x = np.random.randint(0, max(1, w // 3))
        crop = img[y:y + h // 3, x:x + w // 3]
        if crop.size == 0:
            continue
        scores.append(float(np.var(crop)))
    if not scores:
        return 0.0
    scores = np.array(scores, dtype=np.float32)
    return float(1.0 - min(np.std(scores) / 100.0, 1.0))


def grain_likelihood(img_np: np.ndarray) -> float:
    """Photographic grain prior; higher → more real."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    mean_noise = np.mean(np.abs(hp))
    return float(min(mean_noise / 3.0, 1.0))


def _extract_prnu_std(img_gray: np.ndarray) -> float:
    """
    PRNU std-based real prior component.
    Simple wavelet-like denoising residual; higher → more real.
    """
    denoised = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)
    noise = img_gray.astype(np.float32) - denoised.astype(np.float32)
    prnu_std = noise.std() / 255.0
    prnu_std = np.clip(prnu_std * 4.0, 0.0, 1.0)
    return float(prnu_std)


def extract_prnu_std(img_gray: np.ndarray) -> float:
    """
    Public PRNU std helper matching real_image_prior_v3 signature.
    """
    return _extract_prnu_std(img_gray)


def extract_cfa_strength(img_bgr: np.ndarray) -> float:
    """
    CFA periodicity strength in the green channel.
    Higher = more camera-like CFA structure.
    """
    h, w, _ = img_bgr.shape
    if h < 2 or w < 2:
        return 0.0
    g = img_bgr[:, :, 1].astype(np.float32)
    diff = np.abs(g[:, 1:] - g[:, :-1])
    avg = float(diff.mean()) if diff.size else 0.0
    cfa_strength = 1.0 - np.clip(avg / 32.0, 0.0, 1.0)
    return float(np.clip(cfa_strength, 0.0, 1.0))


def jpeg_residual_dct(img_gray: np.ndarray) -> float:
    """
    JPEG residual magnitude from 8x8 DCT AC coefficients.
    Higher values indicate stronger camera-like quantization structure.
    """
    h, w = img_gray.shape
    blocks = []
    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            block = img_gray[y:y+8, x:x+8].astype(np.float32) - 128.0
            blocks.append(dct(dct(block.T, norm="ortho").T, norm="ortho"))
    if not blocks:
        return 0.0
    blocks = np.stack(blocks, axis=0)
    ac = np.abs(blocks[:, 1:, 1:])  # ignore DC
    mean_ac = float(np.mean(ac))
    res = np.clip(mean_ac / 40.0, 0.0, 1.0)
    return float(res)


def real_image_prior_v2(img_bgr: np.ndarray) -> float:
    """
    Alternative REAL-image prior v2 (PRNU + CFA + JPEG DCT).
    Returns a score in [0,1]; tuned so real photos with retouching
    tend to get boosted while heavily synthetic / CFA-broken images do not.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. PRNU std (real cameras have stable residual noise)
    prnu_std = _extract_prnu_std(img_gray)

    # 2. CFA inverse (AI / resynthesis tends to break CFA)
    cfa_strength = extract_cfa_strength(img_bgr)
    cfa_inverse = 1.0 - cfa_strength

    # 3. JPEG residual (small real camera signature)
    jpeg_res = jpeg_residual_dct(img_gray)

    score = (
        prnu_std * 0.40 +
        cfa_inverse * 0.35 +
        jpeg_res * 0.25
    )
    return float(np.clip(score, 0.0, 1.0))


def extract_prnu_acorr(img_gray: np.ndarray) -> float:
    """
    PRNU autocorrelation peak: higher when sensor pattern is self-consistent.
    """
    den = cv2.fastNlMeansDenoising(img_gray, None, 10, 7, 21)
    noise = img_gray.astype(np.float32) - den.astype(np.float32)
    try:
        corr = cv2.matchTemplate(noise, noise, cv2.TM_CCORR_NORMED)
        ac_peak = float(corr.mean())
    except Exception:
        return 0.0
    return float(np.clip((ac_peak - 0.95) * 20.0, 0.0, 1.0))


def extract_cfa_inverse(img_bgr: np.ndarray) -> float:
    """
    CFA inverse: higher when CFA structure is broken / AI-like.
    """
    strength = extract_cfa_strength(img_bgr)
    return float(1.0 - strength)


def extract_demosaic_error(img_bgr: np.ndarray) -> float:
    """
    Demosaic interpolation error (green channel); higher = less camera-like.
    """
    g = img_bgr[:, :, 1].astype(np.float32)
    if g.size == 0:
        return 0.0
    kernel = np.array([[0.25, 0.5, 0.25]], dtype=np.float32)
    recon = cv2.filter2D(g, -1, kernel)
    err = float(np.abs(g - recon).mean())
    score = np.clip(err / 20.0, 0.0, 1.0)
    return float(score)


def jpeg_residual(img_gray: np.ndarray) -> float:
    """
    JPEG residual (AC mean) on 8x8 blocks; higher = stronger JPEG structure.
    """
    h, w = img_gray.shape
    vals = []
    for y in range(0, h - 7, 8):
        for x in range(0, w - 7, 8):
            block = img_gray[y:y+8, x:x+8].astype(np.float32) - 128.0
            d = dct(dct(block.T, norm="ortho").T, norm="ortho")
            vals.append(float(np.mean(np.abs(d[1:, 1:]))))
    if not vals:
        return 0.0
    mean_ac = float(np.mean(vals))
    return float(np.clip(mean_ac / 40.0, 0.0, 1.0))


def qtable_consistency(img_gray: np.ndarray) -> float:
    """
    JPEG Q-table consistency: higher = more consistent camera-like compression.
    """
    h, w = img_gray.shape
    blocks = []
    for y in range(0, h - 15, 16):
        for x in range(0, w - 15, 16):
            block = img_gray[y:y+16, x:x+16]
            blocks.append(float(block.std()))
    if not blocks:
        return 0.0
    blocks_arr = np.array(blocks, dtype=np.float32)
    var = float(blocks_arr.std())
    score = 1.0 - np.clip(var / 20.0, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def hf_glcm_contrast(img_gray: np.ndarray) -> float:
    """
    High-frequency GLCM contrast; higher = more real photographic structure.
    Falls back to 0.0 if skimage is unavailable.
    """
    if greycomatrix is None or greycoprops is None:
        return 0.0
    try:
        hf = cv2.Laplacian(img_gray, cv2.CV_32F)
        hf = cv2.normalize(hf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gl = greycomatrix(hf, [1], [0], levels=256, symmetric=True, normed=True)
        contrast = float(greycoprops(gl, "contrast")[0, 0])
        score = np.clip(contrast / 2000.0, 0.0, 1.0)
        return float(score)
    except Exception:
        return 0.0


def real_image_prior_v3(img_bgr: np.ndarray) -> float:
    """
    REAL-image prior v3 (0=fake, 1=real) combining:
      - PRNU std + autocorrelation
      - CFA inverse (as real = 1-cfa_inv)
      - demosaic error structure
      - JPEG residual + Q-table consistency
      - high-frequency GLCM contrast
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    prnu_std = extract_prnu_std(img_gray)
    prnu_ac = extract_prnu_acorr(img_gray)
    cfa_inv = extract_cfa_inverse(img_bgr)
    dem_err = extract_demosaic_error(img_bgr)
    jpeg_res = jpeg_residual(img_gray)
    jpeg_q = qtable_consistency(img_gray)
    glcm_hf = hf_glcm_contrast(img_gray)

    score = (
        prnu_std * 0.22 +
        prnu_ac * 0.18 +
        (1.0 - cfa_inv) * 0.12 +
        (1.0 - dem_err) * 0.12 +
        jpeg_res * 0.12 +
        jpeg_q * 0.12 +
        glcm_hf * 0.12
    )

    return float(np.clip(score, 0.0, 1.0))


def multiscale_fft_confidence(pil: Image.Image) -> bool:
    """
    Multi-scale FFT consistency: True → real, False → fake.
    Checks stability of spectral energy across 256/128/64 crops.
    """
    sizes = [256, 128, 64]
    scores = []
    for sz in sizes:
        arr = np.asarray(pil.resize((sz, sz)))
        if arr.ndim != 3 or arr.shape[2] < 3:
            continue
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
        F = np.fft.fft2(gray)
        mag = np.abs(np.fft.fftshift(F))
        scores.append(float(np.std(mag)))
    if len(scores) < 2:
        return False
    scores_arr = np.array(scores, dtype=np.float32)
    diff = float(scores_arr.max() - scores_arr.min())
    mean = float(scores_arr.mean() + 1e-6)
    return bool(diff < 0.15 * mean)


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


# ============================================================
#  FFT VISUALIZATION UTILITIES (ADVANCED PANEL)
# ============================================================

def fft_maps(img_gray: np.ndarray):
    """
    Return log-magnitude and phase FFT maps for a grayscale image.
    """
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    phase = np.angle(fshift)
    return magnitude, phase


# ============================================================
#   FORENSIC MAP HELPERS (FFT / PRNU / CFA / JPEG / GRAIN)
# ============================================================

def fft_mag_phase(gray: np.ndarray):
    """
    FFT magnitude + phase using numpy. Returns (mag_log, phase).
    """
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    mag = np.log1p(np.abs(Fshift))
    phase = np.angle(Fshift)
    return mag, phase


def prnu_autocorr(gray: np.ndarray):
    """
    PRNU autocorrelation (v4): returns (scalar_ac, prnu_map).
    """
    den = cv2.fastNlMeansDenoising(gray.astype(np.uint8), None, 10, 7, 21)
    noise = gray.astype(np.float32) - den.astype(np.float32)
    try:
        corr = cv2.matchTemplate(noise, noise, cv2.TM_CCORR_NORMED)
        ac = float(corr.mean())
    except Exception:
        ac = 0.0
    ac = float(np.clip(ac, 0.0, 1.0))
    return ac, noise


def cfa_consistency(img_rgb: np.ndarray):
    """
    CFA periodicity consistency map on green channel.
    """
    g = img_rgb[:, :, 1].astype(np.float32)
    diff = np.abs(g[:, 1:] - g[:, :-1])
    cfa_map = gaussian_filter(diff, sigma=1.2)
    cfa_map = (cfa_map - cfa_map.min()) / (cfa_map.max() - cfa_map.min() + 1e-6)
    return cfa_map


def jpeg_block_coherence(gray: np.ndarray):
    """
    JPEG 8x8 block coherence: scalar + visual grid map.
    """
    h, w = gray.shape
    blocks = []
    for y in range(0, h - 8, 8):
        for x in range(0, w - 8, 8):
            block = gray[y:y+8, x:x+8].astype(np.float32)
            blocks.append(float(block.std()))
    if not blocks:
        coherence = 0.0
    else:
        blocks = np.array(blocks, dtype=np.float32)
        std = float(blocks.std())
        coherence = 1.0 - min(std / 30.0, 1.0)
    vis = np.zeros_like(gray, dtype=np.float32)
    vis[::8, :] = 1.0
    vis[:, ::8] = 1.0
    vis = gaussian_filter(vis, sigma=1.0)
    return float(np.clip(coherence, 0.0, 1.0)), vis


def hf_phase_randomness(gray: np.ndarray):
    """
    High-frequency phase randomness: real > random; AI tends to be structured.
    Returns (score, phase_map).
    """
    F = np.fft.fft2(gray)
    Fshift = np.fft.fftshift(F)
    phase = np.angle(Fshift)
    s = float(np.std(phase))
    score = 1.0 - min(s / np.pi, 1.0)
    return float(np.clip(score, 0.0, 1.0)), phase


def hf_lf_fusion(gray: np.ndarray, cutoff: int = 20):
    h, w = gray.shape
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    crow, ccol = h // 2, w // 2
    mask_low = np.zeros_like(fshift)
    mask_low[crow-cutoff:crow+cutoff, ccol-cutoff:ccol+cutoff] = 1
    mask_high = 1 - mask_low
    low = np.log1p(np.abs(fshift * mask_low))
    high = np.log1p(np.abs(fshift * mask_high))
    return low, high


def radial_profile(data: np.ndarray):
    h, w = data.shape
    y, x = np.indices((h, w))
    cy, cx = h // 2, w // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), weights=data.ravel())
    nr = np.bincount(r.ravel())
    radial_mean = tbin / nr
    return radial_mean[: min(h, w) // 2]


def patch_fft_anomaly(gray: np.ndarray, patch: int = 32):
    H, W = gray.shape
    if H < patch or W < patch:
        return np.zeros_like(gray, dtype=np.float32)
    out = np.zeros((H // patch, W // patch), np.float32)
    for i in range(0, H - patch, patch):
        for j in range(0, W - patch, patch):
            blk = gray[i:i+patch, j:j+patch]
            F = np.fft.fft2(blk)
            Fshift = np.fft.fftshift(F)
            mag = np.log1p(np.abs(Fshift))
            out[i // patch, j // patch] = float(mag.mean())
    out = (out - out.min()) / (out.max() - out.min() + 1e-6)
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)
    return gaussian_filter(out, sigma=1.0)


def srm_energy(gray: np.ndarray):
    k = [
        np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]], np.float32),
        np.array([[-1, 2, -1],[2, -4, 2],[-1, 2, -1]], np.float32),
        np.array([[1, -2, 1],[-2, 4, -2],[1, -2, 1]], np.float32),
    ]
    energies = []
    maps = []
    for f in k:
        r = cv2.filter2D(gray, -1, f)
        energies.append(float((r**2).mean()))
        maps.append(r)
    energy = float(np.clip(sum(energies)/len(energies)/2000.0, 0.0, 1.0))
    return energy, maps


def grain_likelihood_map(gray: np.ndarray):
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    grain = np.abs(hp)
    grain_norm = (grain - grain.min()) / (grain.max() - grain.min() + 1e-6)
    score = float(min(grain_norm.mean() / 0.15, 1.0))
    return score, grain_norm


def prnu_fft_consistency(noise: np.ndarray) -> float:
    """
    PRNU FFT consistency: real sensors → structured radial patterns,
    synthetic PRNU → more random. Higher diff = less consistent.
    """
    try:
        fft = np.fft.fft2(noise)
        mag = np.abs(fft)
        radial = mag.mean(axis=0)
        smooth = gaussian_filter(radial, 3.0)
        diff = np.mean(np.abs(radial - smooth))
        return float(diff)
    except Exception:
        return 0.0


def forensic_panel(img_bgr: np.ndarray) -> Image.Image:
    """
    3x3 forensic diagnostics grid:
      - FFT magnitude/phase
      - PRNU map & autocorr
      - CFA consistency
      - JPEG block coherence
      - HF phase randomness
      - SRM residual energy
      - Patch FFT anomaly
      - Grain likelihood map
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    fft_mag, fft_phase = fft_mag_phase(gray)
    prnu_ac, prnu_map_img = prnu_autocorr(gray)
    cfa_img = cfa_consistency(img_rgb)
    jpeg_coh, jpeg_map = jpeg_block_coherence(gray)
    hf_rand, phase_map = hf_phase_randomness(gray)
    srm_val, srm_maps = srm_energy(gray)
    patch_fft = patch_fft_anomaly(gray)
    grain_val, grain_map = grain_likelihood_map(gray)

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    axs[0, 0].imshow(fft_mag, cmap="inferno"); axs[0, 0].set_title("FFT Magnitude"); axs[0, 0].axis("off")
    axs[0, 1].imshow(fft_phase, cmap="twilight"); axs[0, 1].set_title("FFT Phase"); axs[0, 1].axis("off")
    axs[0, 2].imshow(prnu_map_img, cmap="gray"); axs[0, 2].set_title(f"PRNU Autocorr {prnu_ac:.2f}"); axs[0, 2].axis("off")

    axs[1, 0].imshow(cfa_img, cmap="plasma"); axs[1, 0].set_title("CFA Consistency"); axs[1, 0].axis("off")
    axs[1, 1].imshow(jpeg_map, cmap="gray"); axs[1, 1].set_title(f"JPEG Block Coherence {jpeg_coh:.2f}"); axs[1, 1].axis("off")
    axs[1, 2].imshow(phase_map, cmap="twilight"); axs[1, 2].set_title(f"HF Phase Random {hf_rand:.2f}"); axs[1, 2].axis("off")

    axs[2, 0].imshow(sum(srm_maps)/len(srm_maps), cmap="gray"); axs[2, 0].set_title(f"SRM Residual {srm_val:.2f}"); axs[2, 0].axis("off")
    axs[2, 1].imshow(patch_fft, cmap="viridis"); axs[2, 1].set_title("Patch FFT Anomaly"); axs[2, 1].axis("off")
    axs[2, 2].imshow(grain_map, cmap="hot"); axs[2, 2].set_title(f"Grain Likelihood {grain_val:.2f}"); axs[2, 2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ============================================================
#   IMPROVEMENT HELPERS (REAL OVERRIDES / UPSCALER / ETC.)
# ============================================================

def real_hard_override(cfa, grain, jpeg):
    if (
        cfa is not None and cfa < 0.18 and
        grain is not None and grain > 0.80 and
        jpeg is not None and jpeg < 0.002
    ):
        return True
    return False


def esrgan_grid_score(gray):
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))
    v = float(mag[:, ::8].mean())
    h = float(mag[::8, :].mean())
    return float(np.clip((v + h) / 50.0, 0.0, 1.0))


def saturation_peak_score(img_np):
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    peak_ratio = float(np.mean(s > 200))
    return float(min(peak_ratio / 0.05, 1.0))


def jpeg_q_mismatch(gray):
    blocks = []
    for y in range(0, gray.shape[0] - 8, 8):
        for x in range(0, gray.shape[1] - 8, 8):
            blk = gray[y:y+8, x:x+8].astype("float32")
            blocks.append(cv2.Laplacian(blk, cv2.CV_32F).var())
    if not blocks:
        return 0.0
    blocks = np.array(blocks, dtype=np.float32)
    return float(min(blocks.std() / 30.0, 1.0))


def face_region_retouch_score(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    perlin = perlin_diffusion_score_fixed(face_bgr)
    HF = cv2.Laplacian(gray, cv2.CV_32F).var()
    hf_flat = float(np.clip(1 - HF / 200.0, 0.0, 1.0))
    return 0.5 * perlin + 0.5 * hf_flat


def exposure_variation(gray):
    hist = cv2.equalizeHist(gray)
    return float(np.std(hist) / 60.0)


# ============================================================
#   RENDERING PIPELINE REGULARITY (ANTI-PERFECTION)
# ============================================================

def rendering_pipeline_score(frames):
    """
    Detects over-regular camera simulation.
    Higher = suspiciously 'too perfect'.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    # ---- Exposure continuity ----
    hists = []
    for f in frames:
        gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist)

    diffs = [
        cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_BHATTACHARYYA)
        for i in range(len(hists) - 1)
    ]
    exposure_perfection = 1.0 - float(np.mean(diffs))

    # ---- Motion blur regularity ----
    blur_vals = []
    for i in range(1, len(frames)):
        prev = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY)
        curr = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        edges = cv2.Canny(curr, 100, 200)
        blur_vals.extend(mag[edges > 0])

    if len(blur_vals) > 50:
        blur_vals = np.array(blur_vals)
        blur_regularity = np.exp(-np.var(blur_vals))
    else:
        blur_regularity = 0.0

    score = 0.55 * exposure_perfection + 0.45 * blur_regularity
    return float(np.clip(score, 0.0, 1.0))


def image_generator_likelihood(
    diffusion_score=None,
    perlin_score=None,
    texture_noise=None,
    render_score=0.0,
    jpeg_q_score=None,
    sat_peak=None,
    spectral_score=0.0,
    cfa_fake_score=None,
    esrgan_score=None,
    embedding_anomaly=None,
    patch_spread=None,
    head_delta=None,
    prnu_scaled=None,
    grain_real=None,
    real_prior_v4=None,
    hc_score=None,
):
    """
    Image-only generator likelihood from static cues.
    Higher = more synthetic-like.
    """
    signals = []

    def _add(val, weight):
        if val is None:
            return
        try:
            v = float(np.clip(val, 0.0, 1.0))
        except Exception:
            return
        signals.append((v, weight))

    _add(diffusion_score, 0.18)
    _add(perlin_score, 0.12)
    _add(texture_noise, 0.10)
    _add(render_score, 0.08)
    _add(jpeg_q_score, 0.10)
    _add(sat_peak, 0.08)
    _add(spectral_score, 0.08)
    _add(cfa_fake_score, 0.10)
    _add(esrgan_score, 0.06)
    _add(embedding_anomaly, 0.05)
    _add(patch_spread, 0.04)
    _add(head_delta, 0.03)

    if not signals:
        return 0.0

    total_w = sum(w for _, w in signals)
    raw = sum(v * w for v, w in signals) / max(1e-6, total_w)

    real_signals = []

    def _add_real(val, weight):
        if val is None:
            return
        try:
            v = float(np.clip(val, 0.0, 1.0))
        except Exception:
            return
        real_signals.append((v, weight))

    _add_real(prnu_scaled, 0.25)
    _add_real(grain_real, 0.20)
    _add_real(real_prior_v4, 0.25)
    _add_real(hc_score, 0.15)
    if jpeg_q_score is not None:
        _add_real(1.0 - float(np.clip(jpeg_q_score, 0.0, 1.0)), 0.15)

    real_guard = 0.0
    if real_signals:
        real_w = sum(w for _, w in real_signals)
        real_guard = sum(v * w for v, w in real_signals) / max(1e-6, real_w)

    score = raw * (1.0 - 0.55 * real_guard)
    return float(np.clip(score, 0.0, 1.0))

# ============================================================
#   MOTION / PARALLAX CONSISTENCY (SORA FAIL MODE)
# ============================================================

def parallax_inconsistency(frames):
    """
    Checks depth-motion agreement.
    Sora struggles with parallax under camera motion.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    errs = []
    for i in range(1, len(frames)):
        a = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)

        edges = cv2.Canny(b, 100, 200)
        if edges.sum() > 0:
            errs.append(float(np.var(mag[edges > 0])))

    if not errs:
        return 0.0

    v = float(np.mean(errs))
    return float(np.clip(v / 15.0, 0.0, 1.0))


def real_confidence_stabilizer(real_prior, forensic):
    if real_prior is not None and real_prior > 0.55 and forensic < 0.60:
        return True
    return False


def stable_patch_score(res):
    return 0.5 * res.get("visual_prob", 0.5) + 0.5 * res.get("freq_prob", 0.5)


def low_res_penalty(w, h):
    if min(w, h) < 256:
        return 0.9
    return 1.0


def confidence_text(cert):
    if cert > 0.55:
        return "Confidence: HIGH"
    elif cert > 0.30:
        return "Confidence: MEDIUM"
    else:
        return "Confidence: LOW – verify manually"

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
        pr = prnu_consistency_score(img_np)
        pr_n = 1.0 - min(pr / 3.5, 1.0)
        return float(np.clip(pr_n, 0.0, 1.0))
    except Exception:
        return 0.0

# ============================================================
#  FORENSIC V2: Classic + Diffusion Deepfake Predictors
# ============================================================

def perlin_diffusion_score_fixed(img_bgr: np.ndarray) -> float:
    """
    Fixed Perlin Diffusion Score:
    - Much lower false positives on real photos
    - Still very sensitive to diffusion-style smoothness
    Expects BGR array in OpenCV format; returns [0,1] fake score.
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0

    img = img_bgr.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 1. Gradient Smoothness Map (Perlin Core)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_norm = grad_mag / (grad_mag.mean() + 1e-6)
    smoothness = float(np.exp(-np.std(grad_norm)))

    # 2. High-Frequency Residual Energy (AI images have too little high-freq)
    high_pass = gray - gaussian_filter(gray, sigma=1.2)
    hf_std = float(high_pass.std())
    hf_penalty = float(np.clip(1.0 - (hf_std / 0.03), 0.0, 1.0))

    # 3. Local Entropy Check (Real images have more entropy)
    entropy = cv2.Laplacian(gray, cv2.CV_32F)
    entropy_score = float(np.exp(-np.std(entropy)))

    # 4. PRNU-lite Camera Noise Check (Real sensors leave micro-noise)
    prnu_map = gray - gaussian_filter(gray, sigma=2.5)
    prnu_std = float(prnu_map.std())
    prnu_penalty = float(np.clip(1.0 - (prnu_std / 0.01), 0.0, 1.0))

    score = (
        0.45 * smoothness +
        0.25 * hf_penalty +
        0.15 * entropy_score +
        0.15 * prnu_penalty
    )

    return float(np.clip(score, 0.0, 1.0))

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
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    perlin_score = perlin_diffusion_score_fixed(img_bgr)
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


def asymmetry_score(img_np: np.ndarray) -> float:
    """
    Local left-right asymmetry; higher → more fake.
    Particularly sensitive to symmetric AI portraits / products.
    """
    h, w, _ = img_np.shape
    if w < 4:
        return 0.0
    mid = w // 2
    left = img_np[:, :mid]
    right = img_np[:, mid:]
    if right.shape[1] == 0:
        return 0.0
    right_flip = np.flip(right, axis=1)
    min_w = min(left.shape[1], right_flip.shape[1])
    left = left[:, :min_w].astype(np.float32)
    right_flip = right_flip[:, :min_w].astype(np.float32)
    diff = np.mean(np.abs(left - right_flip))
    score = 1.0 - min(diff / 25.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def color_harmony_score(img_np: np.ndarray) -> float:
    """
    Color harmony histogram dispersion; higher → more fake.
    Captures unnatural hue clumping / plastic skin / synthetic skies.
    """
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].flatten()
    hist, _ = np.histogram(h, bins=36, range=(0, 180))
    return float(min(np.std(hist) / 200.0, 1.0))


def histogram_consistency(img_bgr: np.ndarray, block: int = 64, bins: int = 32) -> float:
    """
    Histogram Consistency (HC)
    - Measures color histogram similarity across blocks.
    - Low HC = consistent (real)
    - High HC = inconsistent (tampering/splicing/local editing)
    """
    h, w, _ = img_bgr.shape
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    H, W = h // block, w // block
    if H <= 0 or W <= 0:
        return 0.0

    histograms = []
    for i in range(H):
        for j in range(W):
            tile = img_hsv[i * block : (i + 1) * block, j * block : (j + 1) * block]
            hist = cv2.calcHist(
                [tile],
                [0, 1, 2],
                None,
                [bins, bins, bins],
                [0, 180, 0, 256, 0, 256],
            )
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)

    histograms = np.array(histograms, dtype=np.float32)
    if histograms.shape[0] < 2:
        return 0.0

    norms = np.linalg.norm(histograms, axis=1, keepdims=True) + 1e-8
    norm_hist = histograms / norms
    sim = np.dot(norm_hist, norm_hist.T)

    inconsistency = 1.0 - float(np.mean(sim))
    return float(np.clip(inconsistency, 0.0, 1.0))


def real_prior_v2(pil: Image.Image) -> float:
    """
    Aggregated REAL-image prior ∈ [0,1].
    Higher → stronger evidence of a real camera pipeline.
    Uses JPEG block patterns, highlight clipping, crop consistency,
    CFA Bayer pattern, PRNU variance, grain, and multiscale FFT stability.
    """
    img_np = np.asarray(pil.convert("RGB"))
    r1 = jpeg_block_consistency(img_np)
    r2 = highlight_clipping_realness(img_np)
    r3 = crop_consistency_score(pil)
    r4 = 1.0 - cfa_bayer_score(img_np)
    r5 = prnu_consistency_score(img_np)
    r6 = grain_likelihood(img_np)
    r7 = float(multiscale_fft_confidence(pil))
    prior = (r1 + r2 + r3 + r4 + r5 + r6 + r7) / 7.0
    return float(np.clip(prior, 0.0, 1.0))

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
# CLUE 2: Color Correlation Drift (AI colors decorrelated)
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

    # Second SigLIP head: 90° rotated view (dual-view stabilizer)
    try:
        pil_rot = pil.rotate(90, expand=False)
        x_rot = preprocess(pil_rot).unsqueeze(0).to(DEVICE)
        z_rot = float(siglip(x_rot).item())
        base_prob = float(torch.sigmoid(torch.tensor(z_sig)).item())
        rot_prob = float(torch.sigmoid(torch.tensor(z_rot)).item())
        visual_prob = 0.6 * base_prob + 0.4 * rot_prob
        z_sig = float(_logit(visual_prob))
    except Exception:
        visual_prob = float(torch.sigmoid(torch.tensor(z_sig)).item())

    # Head-wise probabilities for diagnostics
    p_sig = visual_prob
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
        "visual_prob": float(p_sig),
        "freq_prob": float(p_freq),
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
    """
    9-crop ensemble:
      - center
      - left / right
      - top / bottom
      - 4 quadrants
    """
    w, h = pil.size
    if w < 4 or h < 4:
        return [pil.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)], torch.tensor([1.0])

    mid_w, mid_h = w // 2, h // 2

    # Center crop (50% size)
    cw, ch = max(1, w // 2), max(1, h // 2)
    cx0 = max(0, (w - cw) // 2)
    cy0 = max(0, (h - ch) // 2)
    center = pil.crop((cx0, cy0, cx0 + cw, cy0 + ch))

    left = pil.crop((0, 0, mid_w, h))
    right = pil.crop((w - mid_w, 0, w, h))
    top = pil.crop((0, 0, w, mid_h))
    bottom = pil.crop((0, h - mid_h, w, h))

    tl = pil.crop((0, 0, mid_w, mid_h))
    tr = pil.crop((w - mid_w, 0, w, mid_h))
    bl = pil.crop((0, h - mid_h, mid_w, h))
    br = pil.crop((w - mid_w, h - mid_h, w, h))

    crops = [center, left, right, top, bottom, tl, tr, bl, br]
    weights = torch.tensor(
        [0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        dtype=torch.float32,
    )
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

    # Build a matplotlib figure with a colorbar key for the heatmap
    cmap = plt.get_cmap("jet")
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(patches_norm, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Local fake score", fontsize=8)
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    heat_raw = Image.open(buf).convert("RGBA").resize((w, h), Image.BILINEAR)

    # Blend original image with the colored heatmap
    base = pil.convert("RGBA")
    return Image.blend(base, heat_raw, alpha=0.45)


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
#   TEMPORAL IDENTITY DRIFT (SORA DETECTOR CORE)
# ============================================================

@torch.no_grad()
def temporal_identity_drift(frames, siglip):
    """
    Measures latent identity instability across frames.
    Real cameras -> stable embeddings
    Sora -> subtle non-rigid drift
    """
    if frames is None or len(frames) < 3:
        return 0.0

    embeds = []
    for f in frames:
        x = preprocess(f).unsqueeze(0).to(DEVICE)
        e = siglip.backbone.encode_image(x)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-6)
        embeds.append(e.cpu().numpy()[0])

    embeds = np.stack(embeds, axis=0)

    # Pairwise cosine distance
    sims = []
    for i in range(len(embeds) - 1):
        cos = float(np.dot(embeds[i], embeds[i + 1]))
        sims.append(cos)

    sims = np.array(sims)
    drift = 1.0 - float(np.mean(sims))

    # Normalize: real ~ 0.05-0.10, Sora ~ 0.25-0.45
    return float(np.clip((drift - 0.08) / 0.35, 0.0, 1.0))

# ============================================================
#   TEMPORAL CONSISTENCY SIGNALS (SORA ATTRIBUTION)
# ============================================================

def face_topology_drift(frames):
    """
    Measures drift in face geometry across frames.
    Uses 5-point landmarks when available.
    """
    if not HAS_FACE or FACE_MODEL is None or frames is None or len(frames) < 3:
        return None

    vectors = []
    for f in frames:
        try:
            img = np.asarray(f)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            faces = FACE_MODEL.get(img_bgr)
            if not faces:
                continue
            face = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])
            kps = getattr(face, "kps", None)
            if kps is None or len(kps) < 5:
                continue
            kps = np.asarray(kps, dtype=np.float32)
            x0, y0, x1, y1 = [float(v) for v in face.bbox]
            bw = max(1.0, x1 - x0)
            bh = max(1.0, y1 - y0)

            eye_dist = float(np.linalg.norm(kps[0] - kps[1]) / bw)
            mouth_dist = float(np.linalg.norm(kps[3] - kps[4]) / bw)
            eye_center = (kps[0] + kps[1]) * 0.5
            mouth_center = (kps[3] + kps[4]) * 0.5
            eye_to_mouth = float(abs(mouth_center[1] - eye_center[1]) / bh)
            nose_to_eye = float(abs(kps[2][1] - eye_center[1]) / bh)

            vec = np.array(
                [eye_dist, mouth_dist, eye_to_mouth, nose_to_eye],
                dtype=np.float32,
            )
            vectors.append(vec)
        except Exception:
            continue

    if len(vectors) < 3:
        return None

    diffs = [
        float(np.linalg.norm(vectors[i] - vectors[i + 1]))
        for i in range(len(vectors) - 1)
    ]
    drift = float(np.mean(diffs))
    return float(np.clip((drift - 0.03) / 0.12, 0.0, 1.0))


def face_embedding_drift(frames):
    """
    Measures drift in face identity embeddings across frames.
    Uses insightface embeddings when available.
    """
    if not HAS_FACE or FACE_MODEL is None or frames is None or len(frames) < 3:
        return None

    embeds = []
    for f in frames:
        try:
            img = np.asarray(f)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            faces = FACE_MODEL.get(img_bgr)
            if not faces:
                continue
            face = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                emb = getattr(face, "embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(emb) + 1e-6
            embeds.append(emb / norm)
        except Exception:
            continue

    if len(embeds) < 3:
        return None

    sims = []
    for i in range(len(embeds) - 1):
        sims.append(float(np.dot(embeds[i], embeds[i + 1])))
    drift = 1.0 - float(np.mean(sims))
    return float(np.clip((drift - 0.04) / 0.20, 0.0, 1.0))


def face_track_consistency(frames):
    """
    Tracks the dominant face across frames and measures identity/geometry drift.
    """
    if not HAS_FACE or FACE_MODEL is None or frames is None or len(frames) < 3:
        return None

    def _bbox_iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
        area_b = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
        denom = area_a + area_b - inter + 1e-6
        return float(inter / denom)

    track_kps = []
    track_embeds = []
    prev_bbox = None
    prev_emb = None

    for f in frames:
        try:
            img = np.asarray(f)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            faces = FACE_MODEL.get(img_bgr)
            if not faces:
                continue

            chosen = None
            if prev_bbox is None:
                chosen = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])
            else:
                best_score = -1.0
                for face in faces:
                    bbox = [float(v) for v in face.bbox]
                    iou = _bbox_iou(prev_bbox, bbox)
                    emb = getattr(face, "normed_embedding", None)
                    if emb is None:
                        emb = getattr(face, "embedding", None)
                    sim = 0.0
                    if emb is not None and prev_emb is not None:
                        emb = np.asarray(emb, dtype=np.float32)
                        emb = emb / (np.linalg.norm(emb) + 1e-6)
                        sim = float(np.dot(emb, prev_emb))
                    score = 0.6 * iou + 0.4 * sim
                    if score > best_score:
                        best_score = score
                        chosen = face
            if chosen is None:
                continue

            bbox = [float(v) for v in chosen.bbox]
            prev_bbox = bbox

            emb = getattr(chosen, "normed_embedding", None)
            if emb is None:
                emb = getattr(chosen, "embedding", None)
            if emb is not None:
                emb = np.asarray(emb, dtype=np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-6)
                prev_emb = emb
                track_embeds.append(emb)

            kps = getattr(chosen, "kps", None)
            if kps is not None and len(kps) >= 5:
                kps = np.asarray(kps, dtype=np.float32)
                x0, y0, x1, y1 = bbox
                bw = max(1.0, x1 - x0)
                bh = max(1.0, y1 - y0)
                eye_dist = float(np.linalg.norm(kps[0] - kps[1]) / bw)
                mouth_dist = float(np.linalg.norm(kps[3] - kps[4]) / bw)
                eye_center = (kps[0] + kps[1]) * 0.5
                mouth_center = (kps[3] + kps[4]) * 0.5
                eye_to_mouth = float(abs(mouth_center[1] - eye_center[1]) / bh)
                nose_to_eye = float(abs(kps[2][1] - eye_center[1]) / bh)
                vec = np.array(
                    [eye_dist, mouth_dist, eye_to_mouth, nose_to_eye],
                    dtype=np.float32,
                )
                track_kps.append(vec)
        except Exception:
            continue

    scores = []
    if len(track_embeds) >= 3:
        sims = [float(np.dot(track_embeds[i], track_embeds[i + 1])) for i in range(len(track_embeds) - 1)]
        drift = 1.0 - float(np.mean(sims))
        embed_score = float(np.clip((drift - 0.04) / 0.20, 0.0, 1.0))
        scores.append((embed_score, 0.6))

    if len(track_kps) >= 3:
        diffs = [float(np.linalg.norm(track_kps[i] - track_kps[i + 1])) for i in range(len(track_kps) - 1)]
        drift = float(np.mean(diffs))
        geom_score = float(np.clip((drift - 0.03) / 0.12, 0.0, 1.0))
        scores.append((geom_score, 0.4))

    if not scores:
        return None

    total_w = sum(w for _, w in scores)
    return float(sum(val * w for val, w in scores) / total_w)


def object_identity_inconsistency(frames):
    """
    Uses ORB feature persistence as a proxy for object identity stability.
    Higher = more inconsistent across frames.
    """
    if frames is None or len(frames) < 3:
        return None

    try:
        orb = cv2.ORB_create(nfeatures=600)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    except Exception:
        return None

    ratios = []
    for i in range(1, len(frames)):
        try:
            a = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY)
            kpa, desa = orb.detectAndCompute(a, None)
            kpb, desb = orb.detectAndCompute(b, None)
            if desa is None or desb is None or not kpa or not kpb:
                continue
            matches = matcher.match(desa, desb)
            if not matches:
                continue
            good = [m for m in matches if m.distance < 50]
            denom = max(1, min(len(kpa), len(kpb)))
            ratio = float(len(good) / denom)
            ratios.append(ratio)
        except Exception:
            continue

    if not ratios:
        return None

    mean_ratio = float(np.mean(ratios))
    return float(np.clip((0.25 - mean_ratio) / 0.25, 0.0, 1.0))


def background_temporal_inconsistency(frames):
    """
    Measures background histogram instability on border regions.
    Higher = more inconsistent across frames.
    """
    if frames is None or len(frames) < 2:
        return None

    hists = []
    for f in frames:
        try:
            gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            b = int(min(h, w) * 0.12)
            if b < 4:
                continue
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[:b, :] = 255
            mask[-b:, :] = 255
            mask[:, :b] = 255
            mask[:, -b:] = 255
            hist = cv2.calcHist([gray], [0], mask, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hists.append(hist)
        except Exception:
            continue

    if len(hists) < 2:
        return None

    diffs = [
        cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_BHATTACHARYYA)
        for i in range(len(hists) - 1)
    ]
    mean_diff = float(np.mean(diffs))
    return float(np.clip(mean_diff / 0.35, 0.0, 1.0))


def temporal_texture_flicker(frames):
    """
    Measures flicker in high-frequency energy across frames.
    """
    if frames is None or len(frames) < 3:
        return 0.0

    vals = []
    for f in frames:
        try:
            gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
            vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        except Exception:
            continue

    if len(vals) < 3:
        return 0.0

    mean_val = float(np.mean(vals))
    if mean_val <= 0.0:
        return 0.0

    cv = float(np.std(vals) / mean_val)
    return float(np.clip((cv - 0.15) / 0.60, 0.0, 1.0))


def flow_reprojection_error(frames):
    """
    Measures photometric error after optical flow reprojection.
    Higher = less motion-consistent.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    errs = []
    for i in range(1, len(frames)):
        try:
            prev = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY).astype(np.float32)
            curr = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY).astype(np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            h, w = prev.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(prev, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            diff = cv2.absdiff(curr, warped)
            edges = cv2.Canny(curr.astype(np.uint8), 80, 160)
            if edges.sum() > 0:
                diff_val = float(np.mean(diff[edges > 0]))
            else:
                diff_val = float(np.mean(diff))
            denom = float(np.mean(curr) + 1e-6)
            errs.append(diff_val / denom)
        except Exception:
            continue

    if not errs:
        return 0.0

    mean_err = float(np.mean(errs))
    return float(np.clip((mean_err - 0.03) / 0.12, 0.0, 1.0))


def temporal_edge_flicker(frames):
    """
    Measures instability in edge density across frames.
    """
    if frames is None or len(frames) < 3:
        return 0.0

    densities = []
    for f in frames:
        try:
            gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            densities.append(float(np.mean(edges > 0)))
        except Exception:
            continue

    if len(densities) < 3:
        return 0.0

    mean_val = float(np.mean(densities))
    if mean_val <= 0.0:
        return 0.0

    cv = float(np.std(densities) / mean_val)
    return float(np.clip((cv - 0.15) / 0.50, 0.0, 1.0))


def temporal_color_drift(frames):
    """
    Measures average color drift across frames.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    means = []
    for f in frames:
        try:
            lab = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2LAB)
            means.append(lab.reshape(-1, 3).mean(axis=0))
        except Exception:
            continue

    if len(means) < 2:
        return 0.0

    diffs = [
        float(np.linalg.norm(means[i] - means[i + 1]))
        for i in range(len(means) - 1)
    ]
    mean_diff = float(np.mean(diffs))
    return float(np.clip((mean_diff - 4.0) / 16.0, 0.0, 1.0))


def noise_residual_incoherence(frames):
    """
    Measures correlation of noise residuals in low-texture regions.
    Lower correlation -> more suspicious.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    residuals = []
    masks = []
    for f in frames:
        try:
            gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY).astype(np.float32)
            blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
            resid = gray - blur
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = cv2.magnitude(gx, gy)
            mask = grad < 10.0
            if float(np.mean(mask)) < 0.05:
                mask = None
            residuals.append(resid)
            masks.append(mask)
        except Exception:
            continue

    if len(residuals) < 2:
        return 0.0

    corrs = []
    for i in range(len(residuals) - 1):
        m0 = masks[i]
        m1 = masks[i + 1]
        if m0 is not None and m1 is not None:
            mask = m0 & m1
        else:
            mask = None
        if mask is not None and float(np.mean(mask)) >= 0.02:
            a = residuals[i][mask].flatten()
            b = residuals[i + 1][mask].flatten()
        else:
            a = residuals[i].flatten()
            b = residuals[i + 1].flatten()
        a = a - float(a.mean())
        b = b - float(b.mean())
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        corr = float(np.dot(a, b) / denom)
        if np.isfinite(corr):
            corrs.append(corr)

    if not corrs:
        return 0.0

    mean_corr = float(np.mean(corrs))
    return float(np.clip((0.15 - mean_corr) / 0.15, 0.0, 1.0))


def spectral_profile_drift(frames):
    """
    Measures drift in radial FFT magnitude profiles across frames.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    profiles = []
    for f in frames:
        try:
            gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
            F = np.fft.fftshift(np.fft.fft2(gray))
            mag = np.log1p(np.abs(F)).astype(np.float32)
            h, w = mag.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            r_norm = r / (r.max() + 1e-6)
            bins = 20
            hist = []
            for i in range(bins):
                m = (r_norm >= i / bins) & (r_norm < (i + 1) / bins)
                hist.append(float(mag[m].mean()) if np.any(m) else 0.0)
            hist = np.array(hist, dtype=np.float32)
            hist = hist / (hist.sum() + 1e-6)
            profiles.append(hist)
        except Exception:
            continue

    if len(profiles) < 2:
        return 0.0

    diffs = []
    for i in range(len(profiles) - 1):
        p = profiles[i]
        q = profiles[i + 1]
        bc = float(np.sum(np.sqrt(p * q)))
        dist = 1.0 - bc
        diffs.append(dist)

    mean_diff = float(np.mean(diffs))
    return float(np.clip(mean_diff / 0.25, 0.0, 1.0))


def flow_forward_backward_inconsistency(frames):
    """
    Measures forward-backward optical flow inconsistency.
    Higher = less physically consistent motion.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    errs = []
    for i in range(1, len(frames)):
        try:
            prev = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY).astype(np.float32)
            curr = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY).astype(np.float32)
            flow_f = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_b = cv2.calcOpticalFlowFarneback(
                curr, prev, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            h, w = prev.shape
            step = max(2, min(h, w) // 64)
            acc = []
            for y in range(0, h, step):
                for x in range(0, w, step):
                    fx, fy = flow_f[y, x]
                    x2 = int(round(x + fx))
                    y2 = int(round(y + fy))
                    if x2 < 0 or y2 < 0 or x2 >= w or y2 >= h:
                        continue
                    bx, by = flow_b[y2, x2]
                    err = float(np.hypot(fx + bx, fy + by))
                    acc.append(err)
            if acc:
                errs.append(float(np.mean(acc)))
        except Exception:
            continue

    if not errs:
        return 0.0

    mean_err = float(np.mean(errs))
    return float(np.clip(mean_err / 2.0, 0.0, 1.0))


def flow_direction_incoherence(frames):
    """
    Measures instability of dominant motion direction across frame pairs.
    """
    if frames is None or len(frames) < 3:
        return 0.0

    hists = []
    for i in range(1, len(frames)):
        try:
            prev = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask = mag > 0.5
            if not np.any(mask):
                continue
            ang = ang[mask]
            bins = 16
            hist, _ = np.histogram(ang, bins=bins, range=(0, 2 * np.pi))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-6)
            hists.append(hist)
        except Exception:
            continue

    if len(hists) < 2:
        return 0.0

    diffs = []
    for i in range(1, len(hists)):
        h0 = hists[i - 1]
        h1 = hists[i]
        bc = float(np.sum(np.sqrt(h0 * h1)))
        diffs.append(1.0 - bc)

    mean_diff = float(np.mean(diffs))
    return float(np.clip(mean_diff / 0.6, 0.0, 1.0))


def temporal_frame_scores(frames):
    """
    Per-frame temporal change score based on grayscale difference.
    Used to weight aggregation toward motion/anomaly spikes.
    """
    if frames is None:
        return []
    if len(frames) < 2:
        return [0.0 for _ in frames]

    diffs = []
    prev = None
    for f in frames:
        try:
            gray = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        except Exception:
            diffs.append(0.0)
            prev = None
            continue
        if prev is None:
            diffs.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev)
            diffs.append(float(np.mean(diff)) / 255.0)
        prev = gray

    arr = np.asarray(diffs, dtype=np.float32)
    if arr.size == 0:
        return []
    lo = float(np.percentile(arr, 25))
    hi = float(np.percentile(arr, 90))
    if hi <= lo + 1e-6:
        norm = np.zeros_like(arr)
    else:
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return norm.tolist()


def temporal_frame_weights(frames):
    """
    Returns (scores, weights) aligned with frames.
    """
    scores = temporal_frame_scores(frames)
    if not scores:
        return [], []
    weights = [0.6 + 1.0 * s for s in scores]
    return scores, weights


def klt_track_instability(frames):
    """
    Measures instability of KLT feature tracks across frames.
    Higher = more track dropouts and erratic motion.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    losses = []
    errs = []
    for i in range(1, len(frames)):
        try:
            prev = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY)
            p0 = cv2.goodFeaturesToTrack(
                prev,
                maxCorners=240,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7,
            )
            if p0 is None or len(p0) < 10:
                continue
            p1, st, _err = cv2.calcOpticalFlowPyrLK(
                prev,
                curr,
                p0,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            if p1 is None or st is None:
                continue
            st = st.reshape(-1)
            total = len(st)
            good = int(np.sum(st == 1))
            if total > 0:
                losses.append(1.0 - (good / total))
            if good > 0:
                diffs = p1[st == 1] - p0[st == 1]
                mags = np.linalg.norm(diffs.reshape(-1, 2), axis=1)
                errs.append(float(np.mean(mags)))
        except Exception:
            continue

    if not losses and not errs:
        return 0.0

    loss_mean = float(np.mean(losses)) if losses else 0.0
    err_mean = float(np.mean(errs)) if errs else 0.0
    loss_score = float(np.clip((loss_mean - 0.10) / 0.40, 0.0, 1.0))
    err_score = float(np.clip(err_mean / 6.0, 0.0, 1.0))
    return float(0.6 * loss_score + 0.4 * err_score)


def affine_inlier_inconsistency(frames):
    """
    Measures geometric consistency via affine inlier ratio.
    Lower inlier ratio -> higher inconsistency.
    """
    if frames is None or len(frames) < 2:
        return 0.0

    ratios = []
    for i in range(1, len(frames)):
        try:
            prev = cv2.cvtColor(np.asarray(frames[i - 1]), cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(np.asarray(frames[i]), cv2.COLOR_RGB2GRAY)
            p0 = cv2.goodFeaturesToTrack(
                prev,
                maxCorners=240,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7,
            )
            if p0 is None or len(p0) < 10:
                continue
            p1, st, _err = cv2.calcOpticalFlowPyrLK(
                prev,
                curr,
                p0,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )
            if p1 is None or st is None:
                continue
            st = st.reshape(-1)
            good = st == 1
            if int(np.sum(good)) < 6:
                continue
            src = p0[good].reshape(-1, 2)
            dst = p1[good].reshape(-1, 2)
            _, inliers = cv2.estimateAffinePartial2D(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
            )
            if inliers is None:
                continue
            ratio = float(np.mean(inliers))
            ratios.append(ratio)
        except Exception:
            continue

    if not ratios:
        return 0.0

    mean_ratio = float(np.mean(ratios))
    return float(np.clip((0.60 - mean_ratio) / 0.60, 0.0, 1.0))

# ============================================================
#                      TRAFFIC-LIGHT LOGIC
# ============================================================

BAND_COLORS = {
    "GREEN": "#6ef3a5",
    "YELLOW": "#ffd666",
    "ORANGE": "#f59e0b",
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


@dataclass
class Verdict:
    label: str
    band: str
    risk_level: str
    prob_fake: float
    certainty: float
    reason: str


def verdict_clamp01(x, default=0.0):
    if x is None:
        return float(default)
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return float(default)


def verdict_safe_get(d: Dict[str, Any], k: str, default: Any = 0.0):
    try:
        v = d.get(k, default)
    except Exception:
        v = default
    if v is None:
        return None if default is None else default
    try:
        return float(v)
    except Exception:
        if default is None:
            return None
        return float(default)


def choose_band(prob_fake: float, certainty: float) -> Tuple[str, str]:
    """
    Bands reflect both probability and certainty.
    """
    p = verdict_clamp01(prob_fake)
    c = verdict_clamp01(certainty)

    # High certainty zones
    if p <= 0.20 and c >= 0.65:
        return "GREEN", "LEAN_REAL"
    if 0.20 < p < 0.50 and c >= 0.65:
        return "YELLOW", "LEAN_REAL"
    if 0.50 <= p < 0.75 and c >= 0.70:
        return "ORANGE", "NEUTRAL"
    if p >= 0.75 and c >= 0.75:
        return "RED", "LEAN_FAKE"

    # Low certainty -> keep it conservative
    if p <= 0.35:
        return "YELLOW", "LEAN_REAL"
    if p <= 0.60:
        return "YELLOW", "NEUTRAL"
    return "ORANGE", "NEUTRAL"


def apply_benign_jpeg_penalty(
    metrics: Dict[str, Any],
    prob_fake: float,
    certainty: float,
) -> Tuple[float, float, str]:
    """
    High JPEG/recompression signals should lower confidence and soften the posterior.
    """
    jpeg = verdict_safe_get(metrics, "jpeg", 0.0) or 0.0
    jpeg_q = verdict_safe_get(metrics, "jpeg_q_mismatch_score", 0.0) or 0.0
    sat = verdict_safe_get(metrics, "saturation_peak_score", 0.0) or 0.0

    benign = verdict_clamp01(0.45 * jpeg + 0.35 * jpeg_q + 0.20 * sat)
    if benign < 0.80:
        return prob_fake, certainty, ""

    new_certainty = verdict_clamp01(certainty * (1.0 - 0.35 * benign))
    shrink = 0.20 * benign
    new_prob = verdict_clamp01(prob_fake * (1.0 - shrink) + 0.5 * shrink)

    return new_prob, new_certainty, f"benign_jpeg_penalty={benign:.3f}"


def decide_verdict(result: Dict[str, Any]) -> Verdict:
    """
    Uses bayesian_fusion_posterior if present; otherwise falls back to final_prob.
    """
    p = verdict_safe_get(result, "bayesian_fusion_posterior", None)
    if p is None:
        p = verdict_safe_get(result, "final_prob", 0.5)
    p = verdict_clamp01(p, default=0.5)

    c = verdict_safe_get(result, "bayesian_fusion_certainty", None)
    if c is None:
        c = verdict_safe_get(result, "certainty", 0.5)
    c = verdict_clamp01(c, default=0.5)

    du = verdict_clamp01(verdict_safe_get(result, "dirichlet_uncertainty", 0.0) or 0.0)
    dc = verdict_clamp01(verdict_safe_get(result, "dirichlet_conflict", 0.0) or 0.0)
    c = verdict_clamp01(c * (1.0 - 0.35 * du) * (1.0 - 0.50 * dc))

    p2, c2, jpeg_note = apply_benign_jpeg_penalty(result, p, c)

    cal = result.get("bayesian_fusion_calibrated", {}) or {}
    if not isinstance(cal, dict):
        cal = {}

    visual = verdict_clamp01(
        verdict_safe_get(cal, "visual", verdict_safe_get(result, "visual_head", 0.0) or 0.0)
    )
    freq = verdict_clamp01(
        verdict_safe_get(cal, "freq", verdict_safe_get(result, "freq_head", 0.0) or 0.0)
    )
    forensic = verdict_clamp01(
        verdict_safe_get(cal, "forensic", verdict_safe_get(result, "forensic_score", 0.0) or 0.0)
    )
    cfa = verdict_clamp01(
        verdict_safe_get(cal, "cfa", verdict_safe_get(result, "cfa_fake_score", 0.0) or 0.0)
    )
    patch = verdict_clamp01(
        verdict_safe_get(cal, "patch", verdict_safe_get(result, "patch_mean", 0.0) or 0.0)
    )
    prnu = verdict_clamp01(
        verdict_safe_get(cal, "prnu", verdict_safe_get(result, "prnu_strength_raw", 0.0) or 0.0)
    )
    jpeg = verdict_clamp01(
        verdict_safe_get(cal, "jpeg", verdict_safe_get(result, "jpeg_q_mismatch_score", 0.0) or 0.0)
    )

    benign_jpeg = verdict_clamp01(
        0.45 * jpeg
        + 0.25 * (verdict_safe_get(result, "jpeg_q_mismatch_score", 0.0) or 0.0)
        + 0.30 * (verdict_safe_get(result, "saturation_peak_score", 0.0) or 0.0)
    )
    synth_evidence = verdict_clamp01(0.35 * visual + 0.25 * freq + 0.20 * cfa + 0.20 * patch)
    edit_evidence = verdict_clamp01(0.55 * forensic + 0.25 * patch + 0.20 * benign_jpeg)

    band, risk = choose_band(p2, c2)

    if c2 < 0.55 and 0.35 < p2 < 0.65:
        return Verdict(
            label="UNCERTAIN",
            band=band,
            risk_level=risk,
            prob_fake=p2,
            certainty=c2,
            reason=f"low_confidence p={p2:.3f} c={c2:.3f} {jpeg_note}".strip(),
        )

    if p2 >= 0.75 and c2 >= 0.75:
        if synth_evidence > edit_evidence and benign_jpeg < 0.85:
            lab = "SYNTHETIC"
            why = f"high_p_high_c synth={synth_evidence:.3f} edit={edit_evidence:.3f}"
        else:
            lab = "EDITED"
            why = f"high_p_high_c edit={edit_evidence:.3f} jpeg={benign_jpeg:.3f}"
        reason = (why + (" " + jpeg_note if jpeg_note else "")).strip()
        return Verdict(lab, band, risk, p2, c2, reason)

    if p2 >= 0.50:
        if synth_evidence >= 0.70 and synth_evidence > edit_evidence + 0.10 and benign_jpeg < 0.85:
            return Verdict(
                "SYNTHETIC",
                band,
                risk,
                p2,
                c2,
                f"moderate_p synth={synth_evidence:.3f} edit={edit_evidence:.3f}",
            )
        return Verdict(
            "EDITED",
            band,
            risk,
            p2,
            c2,
            f"moderate_p edit={edit_evidence:.3f} jpeg={benign_jpeg:.3f}",
        )

    return Verdict(
        "LIKELY_REAL",
        band,
        risk,
        p2,
        c2,
        f"low_p p={p2:.3f} c={c2:.3f} {jpeg_note}".strip(),
    )


def verdict_to_ui(verdict: Verdict) -> Dict[str, Any]:
    pred_map = {
        "LIKELY_REAL": "REAL",
        "EDITED": "TAMPERED",
        "SYNTHETIC": "FAKE",
        "UNCERTAIN": "UNCERTAIN",
    }
    return {
        "prediction": pred_map.get(verdict.label, verdict.label),
        "band": verdict.band,
        "risk_level": verdict.risk_level,
        "final_prob": verdict.prob_fake,
        "certainty": verdict.certainty,
        "reason": verdict.reason,
        "label_v2": verdict.label,
    }


def verdict_band_text(band: str, risk_level: str) -> str:
    if band == "GREEN":
        return "GREEN - lean real"
    if band == "YELLOW":
        if risk_level == "LEAN_REAL":
            return "YELLOW - lean real"
        if risk_level == "NEUTRAL":
            return "YELLOW - neutral"
        if risk_level == "LEAN_FAKE":
            return "YELLOW - lean fake"
        return "YELLOW"
    if band == "ORANGE":
        return "ORANGE - neutral"
    if band == "RED":
        return "RED - high fake"
    return band


def label_code_from_prediction(label: str):
    if label == "REAL":
        return 0.0
    if label in ("TAMPERED", "RBR", "RETOUCHED_REAL"):
        return 0.5
    if label == "FAKE":
        return 1.0
    return None


def real_gate(p_final, forensic, jpeg_q, hist, prnu_scaled, patch_spread):
    # Must be confidently non-fake
    if p_final > 0.35:
        return False

    # Strong anomaly signals block REAL
    if forensic is not None and forensic > 0.65:
        return False
    if jpeg_q is not None and jpeg_q > 0.70:
        return False
    if hist is not None and hist > 0.75:
        return False

    # PRNU must be present
    if prnu_scaled is not None and prnu_scaled < 0.30:
        return False

    # No localized weirdness
    if patch_spread is not None and patch_spread > 0.15:
        return False

    return True


def tamper_votes(forensic, jpeg_q, hist):
    votes = 0
    if forensic is not None and forensic > 0.70:
        votes += 1
    if jpeg_q is not None and jpeg_q > 0.80:
        votes += 1
    if hist is not None and hist > 0.85:
        votes += 1
    return votes


def real_pass(cfa_fake, prnu_scaled, real_prior_v3):
    # camera-native signature
    if cfa_fake is not None and cfa_fake < 0.25:
        if prnu_scaled is not None and prnu_scaled > 0.45:
            return True
    if real_prior_v3 is not None and real_prior_v3 > 0.65:
        return True
    return False


def finalize_label_and_risk(label, p_fake, forensic_val, allow_real=True, override_label=None):
    """
    Enforce consistency between probability and label, then recompute band/risk.
    Assumes p_fake is P(FAKE).
    """
    p_fake = _clamp(p_fake)
    f = float(np.clip(forensic_val if forensic_val is not None else 0.5, 0.0, 1.0))

    if override_label is not None:
        label = override_label
    elif label == "INCONCLUSIVE":
        if not DISABLE_INCONCLUSIVE:
            return label, None, "INCONCLUSIVE", "#9ca3af", "GRAY", "INCONCLUSIVE"
        label = "FAKE" if p_fake >= FINAL_FAKE_THRESH else "REAL"
    elif label == "UNCERTAIN":
        if not DISABLE_INCONCLUSIVE:
            return label, None, "UNCERTAIN - low confidence", "#9ca3af", "GRAY", "UNCERTAIN"
        label = "FAKE" if p_fake >= FINAL_FAKE_THRESH else "REAL"

    if override_label is None:
        if p_fake >= FINAL_FAKE_THRESH:
            label = "FAKE"
        elif p_fake <= FINAL_REAL_THRESH:
            label = "REAL" if allow_real else "TAMPERED"
        else:
            label = "TAMPERED"

    if DISABLE_TAMPERED and label in ("TAMPERED", "RBR", "RETOUCHED_REAL"):
        label = "FAKE" if p_fake >= FINAL_FAKE_THRESH else "REAL"
    if DISABLE_INCONCLUSIVE and label in ("INCONCLUSIVE", "UNCERTAIN"):
        label = "FAKE" if p_fake >= FINAL_FAKE_THRESH else "REAL"

    # prediction_code (REAL=0, TAMPERED=0.5, FAKE=1)
    if label == "REAL":
        code = 0.0
    elif label in ("TAMPERED", "RBR", "RETOUCHED_REAL"):
        code = 0.5
        label = "TAMPERED"  # normalize naming
    elif label == "FAKE":
        code = 1.0
    else:
        code = None

    band_text, band_color, band, risk_level = traffic_light_label(label, p_fake, f)
    return label, code, band_text, band_color, band, risk_level


def is_uncertain(p, risk, patch_mean, head_delta):
    return (0.45 <= p <= 0.55) and risk <= 2 and patch_mean < 0.6 and head_delta >= 0.25


def is_inconclusive(p, pg, patch_mean, risk, entropy, head_delta):
    return (
        0.40 <= p <= 0.60 and
        0.40 <= pg <= 0.60 and
        patch_mean < 0.75 and
        risk in (1, 2) and
        entropy > 1.0 and
        head_delta >= 0.15
    )


# ============================================================
#     RETOUCHED-BUT-REAL CLASSIFIER (RBR, 3rd CLASS)
# ============================================================

def classify_rbr(
    fake_score,
    real_prior,
    forensic,
    cfa_fake,
    perlin,
    grain,
    fft_conf,
    patch_mean,
    patch_spread,
):
    """
    Returns: ("REAL" | "RBR" | "FAKE", numeric_code: 0.0 | 0.5 | 1.0)
    REAL  = camera-native real
    RBR   = real base photo but cosmetically retouched / AI-polished
    FAKE  = fully synthetic / deepfake-like
    """
    fake_score = float(np.clip(fake_score, 0.0, 1.0))
    real_prior = float(np.clip(real_prior, 0.0, 1.0))
    forensic = float(np.clip(forensic, 0.0, 1.0))
    cfa_fake = float(np.clip(cfa_fake, 0.0, 1.0))
    perlin = float(np.clip(perlin, 0.0, 1.0))
    grain = float(np.clip(grain, 0.0, 1.0))
    fft_conf = float(np.clip(fft_conf, 0.0, 1.0))
    patch_mean = float(np.clip(patch_mean, 0.0, 1.0))
    patch_spread = float(np.clip(patch_spread, 0.0, 1.0))

    # Strong REAL conditions
    if real_prior > 0.75 and fake_score < 0.35:
        return "REAL", 0.0

    # Strong FAKE conditions
    if fake_score > 0.75 and real_prior < 0.30:
        return "FAKE", 1.0

    # --- RBR core logic ---
    rbr_conditions = 0

    # Real-prior moderate
    if 0.35 <= real_prior <= 0.75:
        rbr_conditions += 1

    # Fake-score moderate
    if 0.30 <= fake_score <= 0.70:
        rbr_conditions += 1

    # Forensic mid-level anomalies
    if 0.40 <= forensic <= 0.75:
        rbr_conditions += 1

    # CFA partially broken (usually retouching)
    if 0.35 <= cfa_fake <= 0.70:
        rbr_conditions += 1

    # Perlin low (not fully AI)
    if perlin < 0.40:
        rbr_conditions += 1

    # Natural grain present
    if grain > 0.80:
        rbr_conditions += 1

    # FFT "flatness" too low → retouched (low confidence in multiscale FFT)
    if fft_conf < 0.25:
        rbr_conditions += 1

    # Very smooth patch grid → denoised/enhanced
    if patch_mean < 0.60 and patch_spread < 0.05:
        rbr_conditions += 1

    # Threshold: 4 signals is enough to mark RBR
    if rbr_conditions >= 4:
        return "RBR", 0.5

    # If unsure fallback to FAKE/REAL
    if fake_score >= 0.60:
        return "FAKE", 1.0
    else:
        return "REAL", 0.0


def classify_three_way(
    fake_score,
    real_prior_v3,
    forensic_score,
    cfa_fake,
    perlin,
    grain,
    fft_conf,
    patch_mean,
    patch_spread,
    jpeg_resid,
    hist_consistency,
    texture_noise,
):
    """
    Simplified 3-way classifier:
      REAL / TAMPERED / FAKE
    """
    # Normalize / default
    S = float(np.clip(fake_score, 0.0, 1.0))
    R = float(np.clip(real_prior_v3 if real_prior_v3 is not None else 0.0, 0.0, 1.0))
    F = float(np.clip(forensic_score if forensic_score is not None else 0.0, 0.0, 1.0))
    C = float(np.clip(cfa_fake if cfa_fake is not None else 0.0, 0.0, 1.0))
    P = float(np.clip(perlin if perlin is not None else 0.0, 0.0, 1.0))
    G = float(np.clip(grain if grain is not None else 0.0, 0.0, 1.0))
    FFT = bool(fft_conf)
    M = float(np.clip(patch_mean if patch_mean is not None else 0.0, 0.0, 1.0))
    PS = float(np.clip(patch_spread if patch_spread is not None else 0.0, 0.0, 1.0))
    J = float(np.clip(jpeg_resid if jpeg_resid is not None else 0.0, 0.0, 1.0))
    HC = float(np.clip(hist_consistency if hist_consistency is not None else 0.0, 0.0, 1.0))
    T = float(np.clip(texture_noise if texture_noise is not None else 0.0, 0.0, 1.0))

    # --------------------------
    # 1 — DEFINITE FAKE
    # --------------------------
    if S > 0.75 and R < 0.30:
        return "FAKE"
    if P > 0.80 and F > 0.60:
        return "FAKE"
    if C > 0.85:
        return "FAKE"

    # --------------------------
    # 2 — DEFINITE REAL
    # --------------------------
    if R > 0.70 and C < 0.25 and P < 0.40:
        return "REAL"
    if G > 0.80 and C < 0.20:
        return "REAL"
    if FFT and F < 0.50:
        return "REAL"

    # --------------------------
    # 3 — TAMPERED (simplified)
    # Must have:
    #   - moderately broken CFA
    #   - at least one other anomaly
    # --------------------------
    tamper_flag = (
        (0.35 < C < 0.80)
        and (
            F > 0.60
            or P > 0.55
            or HC > 0.75
            or J > 0.80
            or PS < 0.04
            or T > 0.65
        )
    )

    if tamper_flag:
        return "TAMPERED"

    # --------------------------
    # Default REAL
    # --------------------------
    return "REAL"

def _to_prob01(v, default=0.5):
    """Robustly coerce any scalar into [0,1]."""
    try:
        if v is None:
            return float(default)
        if isinstance(v, (bool, np.bool_)):
            return float(1.0 if v else 0.0)
        if not np.isfinite(v):
            return float(default)
        return float(np.clip(float(v), 0.0, 1.0))
    except Exception:
        return float(default)

def soften(p, cap=0.98):
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    if p > cap:
        return cap + (p - cap) * 0.1
    return p


def _clamp(p):
    return float(np.clip(p, EPS, 1.0 - EPS))


def logit(p):
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def build_fusion_features_pfake(
    visual_prob,
    freq_prob,
    forensic_val,
    cfa_fake_score,
    jpeg_q_score,
    prnu_strength_scaled,
    patch_mean,
    default=0.5,
):
    """
    IMPORTANT:
    BayesianFusionV2 / DirichletBayesianFusion expect every input to be P(FAKE).
    PRNU strength is a REALNESS signal, so we invert it here.
    """
    # Already fake-ish probabilities
    p_fake_visual = _to_prob01(visual_prob, default)
    p_fake_freq = _to_prob01(freq_prob, default)
    p_fake_forensic = _to_prob01(forensic_val, 0.5)
    p_fake_cfa = _to_prob01(cfa_fake_score, 0.5)
    p_fake_jpeg = _to_prob01(jpeg_q_score, 0.5)
    p_fake_prnu = 1.0 - _to_prob01(prnu_strength_scaled, 0.5)
    p_fake_patch = _to_prob01(patch_mean, 0.5)

    return {
        "visual": float(np.clip(p_fake_visual, 0.0, 1.0)),
        "freq": float(np.clip(p_fake_freq, 0.0, 1.0)),
        "forensic": float(np.clip(p_fake_forensic, 0.0, 1.0)),
        "cfa": float(np.clip(p_fake_cfa, 0.0, 1.0)),
        "jpeg": float(np.clip(p_fake_jpeg, 0.0, 1.0)),
        "prnu": float(np.clip(p_fake_prnu, 0.0, 1.0)),
        "patch": float(np.clip(p_fake_patch, 0.0, 1.0)),
    }

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
    color_score=0.0,
    face_boost=0.0,
    cfa_fake_score=None,
    real_prior=None,
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
    p_color    = _clamp01(color_score)

    p_patch_mean = _clamp01(patch_mean if patch_mean is not None else 0.5)
    p_patch_max  = _clamp01(max_patch   if max_patch   is not None else 0.5)

    # REAL prior → fake probability (1 - prior)
    p_real_prior = None
    if real_prior is not None:
        rp = _clamp01(real_prior)
        p_real_prior = _clamp01(1.0 - rp)

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
        probs   = [p_diff_raw, p_spec, p_color],
        weights = [1.30, 0.80, 0.80],
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
        probs   = [p_core_fake, p_forensic, p_patch_mean, p_real_prior],
        weights = [1.00, 0.40, 0.25, 0.60],
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
        probs   = [p_core_fake, p_diff_raw, p_spec, p_color, p_patch_max],
        weights = [1.00, 0.70, 0.55, 0.55, 0.50],
        prior   = prior_diff_fake,
    )

    # ---------------------------
    # Level 4: Mixture over generator type
    # ---------------------------
    p_final = p_gen_diff * p_fake_diff + (1.0 - p_gen_diff) * p_fake_cam

    # CFA-based REAL tilt for moderate fake scores
    if cfa_fake_score is not None and cfa_fake_score < 0.45:
        odds = _odds(p_final)
        odds *= 0.65   # 35% tilt toward real
        p_final = _from_odds(odds)

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
    THRESH_FAKE = FINAL_FAKE_THRESH

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

def _predict_single_image(image, fast_mode=False, generate_explanation=True):

    if image is None:
        return "No image uploaded.", None, None, None, None, ""

    # --------------------------------------------------------
    #                Decode + basic validations
    # --------------------------------------------------------
    try:
        if isinstance(image, Image.Image):
            pil = image.convert("RGB").copy()
            render_frames = [pil]
        else:
            pil = load_image_any(image)
            render_frames = [pil]
    except Exception as e:
        return f"Decode error: {e}", None, None, None, None, ""

    # Reject blank images
    if is_near_constant(pil):
        return (
            "Image appears nearly blank/uniform - insufficient signal.",
            None,
            None,
            None,
            None,
            "",
        )

    w, h = pil.size
    if min(w, h) < MIN_SIDE:
        return (
            f"Image too small (min side {min(w,h)} px, need >= {MIN_SIDE}).",
            None,
            None,
            None,
            None,
            "",
        )

    # Downscale very large images
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        pil = pil.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        w, h = pil.size
        render_frames[0] = pil

    global CORAL_CUTS, CORAL_TEMP, CORAL_BINS, CORAL
    if CORAL_CUTS is None:
        with CORAL_LOCK:
            if CORAL_CUTS is None:
                CORAL_CUTS, CORAL_TEMP, CORAL_BINS = load_coral()
                CORAL = CoralCalibrator()

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
        render_frames.append(pil_flip)
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
        grid_scores = None
        patch_scores = []
        if not fast_mode:
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
        embed_score = 0.0
        embed_l2 = 0.0
        embed_cos = 0.0
        if not fast_mode and REAL_REF_DIR and os.path.isdir(REAL_REF_DIR):
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
        if not fast_mode:
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
        asym_score = None
        color_harmony = None
        real_prior = None             # combined real prior
        real_prior_alt = None         # v2 PRNU/CFA/JPEG prior
        real_prior_v3_val = None      # v3 PRNU/CFA/DCT/GLCM prior
        real_prior_v4 = None          # v4 PRNU-strength/CFA/JPEG/patch prior
        grain_real = None
        fft_conf_real = None
        jpeg_is_real = None
        jpeg_resid_v3 = None
        esrgan_score = None
        sat_peak = None
        jpeg_q_score = None
        exposure_score = None
        render_score = 0.0
        image_gen_score = 0.0
        face_retouch = None
        prnu_noise = None
        prnu_val_raw = 0.0
        prnu_scaled = 0.0
        prnu_fft_cons = 0.0
        hc_score = None
        face_p_fake = None
        if DETECT_USE_FORENSICS and not fast_mode:
            try:
                img_np = np.asarray(pil.convert("RGB"))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # PRNU v4: strength + FFT consistency
                try:
                    prnu_noise = extract_prnu(img_np)
                    prnu_val_raw = prnu_strength(prnu_noise)
                    prnu_scaled = float(np.clip(prnu_val_raw * 12.0, 0.0, 1.0))
                    prnu_fft_cons = prnu_fft_consistency(prnu_noise)
                except Exception as _e:
                    print(f"[prnu_v4] error: {_e}")
                    prnu_noise = None
                    prnu_val_raw = 0.0
                    prnu_scaled = 0.0
                    prnu_fft_cons = 0.0
                forensic_v2_score, diff_score = forensic_v2(img_np)
                perlin_score = perlin_diffusion_score_fixed(img_bgr)
                cfa_fake_score = cfa_bayer_score(img_np)
                if cfa_fake_score is not None:
                    cfa_fake_score = 0.5 + (cfa_fake_score - 0.5) * CFA_WEIGHT
                texture_noise = texture_noise_score(img_np)
                # Legacy real prior (FFT / CFA / PRNU / grain / crops)
                real_prior_legacy = real_prior_v2(pil)
                # New real priors based on PRNU/CFA/JPEG/GLCM in BGR pipeline
                real_prior_alt = real_image_prior_v2(img_bgr)
                real_prior_v3_val = real_image_prior_v3(img_bgr)
                # Improvement 4: JPEG Q mismatch
                jpeg_q_score = jpeg_q_mismatch(gray_full)
                if jpeg_q_score is not None:
                    jpeg_q_score = soften(_to_prob01(jpeg_q_score, 0.5))

                # Real prior v4: PRNU-strength + CFA-real + JPEG-real + patch consistency
                try:
                    cfa_real = 1.0 - float(np.clip(cfa_fake_score, 0.0, 1.0)) if cfa_fake_score is not None else 0.5
                    jpeg_real = 1.0 - float(np.clip(jpeg_q_score, 0.0, 1.0)) if jpeg_q_score is not None else 0.5
                    patch_consistency = 1.0 - float(np.clip(p_patch_spread, 0.0, 1.0))
                    real_prior_v4 = float(np.clip(
                        0.35 * prnu_scaled +
                        0.25 * cfa_real +
                        0.20 * jpeg_real +
                        0.20 * patch_consistency,
                        0.0,
                        1.0,
                    ))
                except Exception as _e:
                    print(f"[real_prior_v4] error: {_e}")
                    real_prior_v4 = None

                priors = [
                    p for p in (real_prior_legacy, real_prior_alt, real_prior_v3_val, real_prior_v4)
                    if p is not None
                ]
                if priors:
                    real_prior = float(np.clip(float(sum(priors)) / len(priors), 0.0, 1.0))
                grain_real = grain_likelihood(img_np)
                fft_conf_real = float(multiscale_fft_confidence(pil))
                # Improvement 2: Upscaler fingerprints
                esrgan_score = esrgan_grid_score(gray_full)
                # Improvement 3: Beautification
                sat_peak = saturation_peak_score(img_np)
                if sat_peak is not None:
                    sat_peak = soften(_to_prob01(sat_peak, 0.5))
                # Improvement 6: Exposure consistency
                exposure_score = exposure_variation(gray_full)

                # --------------------------------------------------------
                # Rendering pipeline regularity (new)
                # --------------------------------------------------------
                render_score = 0.0
                try:
                    render_score = rendering_pipeline_score(render_frames)
                except Exception as _e:
                    print(f"[rendering] error: {_e}")

                # Histogram Consistency (HC) – block-wise color histogram similarity
                try:
                    hc_score = histogram_consistency(img_bgr)
                    if hc_score is not None:
                        hc_score = soften(_to_prob01(hc_score, 0.5))
                except Exception as _e:
                    print(f"[hc] histogram_consistency error: {_e}")
                    hc_score = None
                # JPEG camera-likeness from v3 components
                try:
                    img_gray_prior = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    jpeg_res_v3 = jpeg_residual(img_gray_prior)
                    jpeg_q_val = qtable_consistency(img_gray_prior)
                    jpeg_is_real = bool(jpeg_q_val >= 0.5)
                    jpeg_resid_v3 = float(jpeg_res_v3)
                except Exception:
                    jpeg_is_real = None
                    jpeg_resid_v3 = None

                # New spectral / color clues (edge continuity and harmony disabled)
                try:
                    spectral_score = spectral_flatness_score(img_np)
                    edge_score = 0.0
                    color_score = color_correlation_score(img_np)
                    asym_score = asymmetry_score(img_np)
                    color_harmony = None  # disabled
                except Exception as e:
                    print(f"[clues] error: {e}")
                    spectral_score = color_score = 0.0
                    asym_score = color_harmony = None

                # Optional face-specific boost for strong diffusion evidence on faces
                if HAS_FACE:
                    try:
                        faces = FACE_MODEL.get(img_bgr)
                        if len(faces) >= 1:
                            face = max(faces, key=lambda x: x.bbox[2] - x.bbox[0])
                            x0, y0, x1, y1 = [int(v) for v in face.bbox]
                            x0 = max(0, x0); y0 = max(0, y0)
                            x1 = min(img_np.shape[1], x1); y1 = min(img_np.shape[0], y1)
                            if x1 > x0 and y1 > y0:
                                face_crop = img_np[y0:y1, x0:x1]
                                face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                                perlin_face = perlin_diffusion_score_fixed(face_bgr)
                                if perlin_face > 0.85:
                                    face_boost = 0.12
                                elif perlin_face > 0.70:
                                    face_boost = 0.08
                                face_h, face_w = face_crop.shape[:2]
                                img_area = float(img_np.shape[0] * img_np.shape[1])
                                face_area = float(face_h * face_w)
                                if img_area > 0.0 and face_area / img_area >= 0.08 and min(face_h, face_w) >= 96:
                                    face_pil = Image.fromarray(face_crop)
                                    face_res = detect_core(face_pil, siglip, freq_mlp, multicrop=True)
                                    face_p_fake = float(face_res.get("p_fake_raw", 0.5))
                    except Exception:
                        face_boost = 0.0

                # v3 forensic fusion: classic + diffusion/perlin/texture/noiseprint (forensic_v2)
                forensic_val = forensic_v2_score
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
                asym_score = None
                color_harmony = None
                real_prior = None
                real_prior_alt = None
                real_prior_v3_val = None
                grain_real = None
                fft_conf_real = None
                jpeg_is_real = None
                jpeg_resid_v3 = None

        # Keep spectral/color/asymmetry scores for reporting only (not fused).
        if not fast_mode:
            try:
                image_gen_score = image_generator_likelihood(
                    diffusion_score=diff_score,
                    perlin_score=perlin_score,
                    texture_noise=texture_noise,
                    render_score=render_score,
                    jpeg_q_score=jpeg_q_score,
                    sat_peak=sat_peak,
                    spectral_score=spectral_score,
                    cfa_fake_score=cfa_fake_score,
                    esrgan_score=esrgan_score,
                    embedding_anomaly=embed_score,
                    patch_spread=p_patch_spread,
                    head_delta=head_delta,
                    prnu_scaled=prnu_scaled if prnu_noise is not None else None,
                    grain_real=grain_real,
                    real_prior_v4=real_prior_v4,
                    hc_score=hc_score,
                )
            except Exception as _e:
                print(f"[image_gen] error: {_e}")
                image_gen_score = 0.0

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
                        0.0,                    # 9 (edge continuity disabled)
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

        p_moe = base["p_moe"] if base["p_moe"] is not None else None

        # --------------------------------------------------------
        #         Bayesian fusion (single probability authority)
        # --------------------------------------------------------
        fusion_calibrated = None
        fusion_reliability = None
        fusion_result = None
        dirichlet_out = None
        dirichlet_p = None
        dirichlet_uncertainty = None
        dirichlet_conflict = None
        dirichlet_strength = None

        fusion_features = build_fusion_features_pfake(
            visual_prob=visual_prob,
            freq_prob=freq_prob,
            forensic_val=forensic_val,
            cfa_fake_score=cfa_fake_score,
            jpeg_q_score=jpeg_q_score,
            prnu_strength_scaled=prnu_scaled if prnu_noise is not None else None,
            patch_mean=p_patch_mean,
            default=0.5,
        )

        p_final = 0.5
        certainty = 0.0
        try:
            fusion = BayesianFusionV2(calibrate=False)
            fusion_result = fusion.fuse(fusion_features, prior_fake=0.30)
            p_final = float(fusion_result["posterior_fake"])
            certainty = float(fusion_result.get("certainty", certainty))

            # Over-perfect rendering = slight suspicion bump
            if render_score > 0.65:
                odds = _odds(p_final)
                odds *= 1.15
                p_final = _from_odds(odds)

            # Over-perfect rendering lowers confidence
            if render_score > 0.60:
                certainty *= (1.0 - 0.30 * render_score)

            # Static generator likelihood (image-only) → gentle odds bump
            if image_gen_score > IMAGE_GEN_TAMPERED_THRESH:
                odds = _odds(p_final)
                if image_gen_score >= IMAGE_GEN_FAKE_THRESH:
                    odds *= IMAGE_GEN_ODDS_HIGH
                elif image_gen_score >= (IMAGE_GEN_TAMPERED_THRESH + 0.15):
                    odds *= IMAGE_GEN_ODDS_MED
                else:
                    odds *= IMAGE_GEN_ODDS_LOW
                p_final = _from_odds(odds)

            # Generator cues lower confidence
            if image_gen_score > 0.0:
                certainty *= (1.0 - 0.25 * image_gen_score)

            fusion_calibrated = fusion_result.get("calibrated")
            fusion_reliability = fusion_result.get("reliability")
        except Exception as _e:
            print(f"[fusion_v2] error: {_e}")

        # Dirichlet evidence is label-only (uncertainty guard)
        try:
            dirichlet = DirichletBayesianFusion(base_strength=4.0)
            dirichlet_out = dirichlet.fuse(fusion_features)
            dirichlet_p = float(dirichlet_out["posterior_fake"])
            dirichlet_uncertainty = float(dirichlet_out["uncertainty"])
            dirichlet_conflict = float(dirichlet_out["conflict"])
            dirichlet_strength = float(dirichlet_out["total_strength"])
        except Exception as _e:
            print(f"[dirichlet_fusion] error: {_e}")

        if (
            not DISABLE_INCONCLUSIVE
            and dirichlet_uncertainty is not None
            and dirichlet_uncertainty > 0.40
        ):
            label = "INCONCLUSIVE"
        elif p_final >= FINAL_FAKE_THRESH:
            label = "FAKE"
        elif p_final <= FINAL_REAL_THRESH:
            label = "REAL"
        else:
            label = "TAMPERED"

        # --------------------------------------------------------
        #                Uncertain / Inconclusive
        # --------------------------------------------------------
        uncertain = is_uncertain(
            p_final, base["risk_idx"], p_patch_mean, head_delta
        )

        inconclusive = is_inconclusive(
            p_final,
            p_global,
            p_patch_mean,
            base["risk_idx"],
            base["entropy"],
            head_delta,
        )

        band_text, band_color, band, risk_level = traffic_light_label(
            label, p_final, forensic_val
        )

        if inconclusive and not DISABLE_INCONCLUSIVE:
            label = "INCONCLUSIVE"
            band_text = "INCONCLUSIVE - borderline evidence"
            band_color = "#cccccc"
        elif uncertain and not DISABLE_INCONCLUSIVE:
            label = "UNCERTAIN"
            band_text = "UNCERTAIN - low confidence"
            band_color = "#cccccc"

        # --------------------------------------------------------
        #         RBR (Retouched-but-Real) 3rd-class label
        # --------------------------------------------------------
        rbr_label = None
        rbr_code = None
        try:
            # fft_conf_real is often stored as 0/1 float here; treat it as probability-like.
            try:
                fft_conf_val = float(np.clip(float(fft_conf_real), 0.0, 1.0)) if fft_conf_real is not None else 0.5
            except Exception:
                fft_conf_val = 0.5

            real_prior_v3_safe = real_prior_v3_val if real_prior_v3_val is not None else 0.0
            forensic_safe = forensic_val if forensic_val is not None else 0.5
            cfa_safe = cfa_fake_score if cfa_fake_score is not None else 0.5
            perlin_safe = perlin_score if perlin_score is not None else 0.0
            grain_safe = grain_real if grain_real is not None else 0.0

            rbr_label, rbr_code = classify_rbr(
                fake_score=float(np.clip(p_final, 0.0, 1.0)),
                real_prior=float(real_prior_v3_safe),
                forensic=float(forensic_safe),
                cfa_fake=float(cfa_safe),
                perlin=float(perlin_safe),
                grain=float(grain_safe),
                fft_conf=float(fft_conf_val),
                patch_mean=float(np.clip(p_patch_mean, 0.0, 1.0)),
                patch_spread=float(np.clip(p_patch_spread, 0.0, 1.0)),
            )
        except Exception as _e:
            print(f"[rbr] classify_rbr error: {_e}")
            rbr_label, rbr_code = None, None

        # Apply RBR only when not INCONCLUSIVE and base label is REAL-ish.
        # Normalize all RBR outputs to the canonical label: TAMPERED.
        if rbr_label == "RBR" and label not in ("INCONCLUSIVE", "UNCERTAIN", "FAKE"):
            label = "TAMPERED"
            risk_level = "TAMPERED"
            band = "YELLOW"
            band_color = BAND_COLORS[band]
            band_text = "TAMPERED"

        # CFA-driven REAL override: strong Bayer pattern → trust camera
        if cfa_fake_score is not None and cfa_fake_score < 0.20:
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
            and p_final >= 0.70
            and forensic_val >= 0.60
        ):
            label = "FAKE"
            band_text, band_color, band, risk_level = traffic_light_label(
                label, p_final, forensic_val
            )

        # ============================================================
        #   APPLY IMPROVEMENTS TO FINAL REAL / TAMPERED / FAKE DECISION
        # ============================================================

        # 1. REAL HARD OVERRIDE (cannot be fake)
        if real_hard_override(cfa_fake_score, grain_real, jpeg_resid_v3):
            label = "REAL"

        # 2. Upscaler fingerprints → TAMPERED
        if esrgan_score is not None and esrgan_score > 0.45 and label != "FAKE":
            label = "TAMPERED"

        # 3. Beautification detector → TAMPERED
        if sat_peak is not None and sat_peak > 0.50 and label == "REAL":
            label = "TAMPERED"

        # 4. JPEG Q mismatch → TAMPERED
        if jpeg_q_score is not None and jpeg_q_score > 0.60 and label != "FAKE":
            label = "TAMPERED"

        # 5. Face retouch evidence → TAMPERED
        if face_retouch is not None and face_retouch > 0.55 and label == "REAL":
            label = "TAMPERED"

        # 6. Exposure uniformity → TAMPERED
        if (
            exposure_score is not None
            and exposure_score < 0.30
            and real_prior_v3_val is not None
            and real_prior_v3_val > 0.30
            and label != "FAKE"
        ):
            label = "TAMPERED"

        # 7. Rendering perfection often implies AI-enhanced real
        if render_score > 0.70 and label == "REAL":
            label = "TAMPERED"

        # --------------------------------------------------------
        #          Final 3-way classifier (REAL/TAMPERED/FAKE)
        # --------------------------------------------------------
        try:
            three_way_label = classify_three_way(
                fake_score=p_final,
                real_prior_v3=real_prior_v3_val,
                forensic_score=forensic_val,
                cfa_fake=cfa_fake_score,
                perlin=perlin_score,
                grain=grain_real,
                fft_conf=fft_conf_real,
                patch_mean=p_patch_mean,
                patch_spread=p_patch_spread,
                jpeg_resid=jpeg_q_score,
                hist_consistency=hc_score,
                texture_noise=texture_noise,
            )
            label = three_way_label
        except Exception as _e:
            print(f"[three_way] classify_three_way error: {_e}")

        # Image-only generator attribution → TAMPERED/FAKE
        if image_gen_score >= IMAGE_GEN_FAKE_THRESH and p_final >= IMAGE_GEN_MIN_FAKE_PROB:
            label = "FAKE"
        elif image_gen_score >= IMAGE_GEN_TAMPERED_THRESH and label in ("REAL", "INCONCLUSIVE", "UNCERTAIN"):
            label = "TAMPERED"

        # --------------------------------------------------------
        #          Face-only escalation (large faces)
        # --------------------------------------------------------
        override_label = None
        if face_p_fake is not None and face_p_fake > 0.65:
            label = "FAKE"
            override_label = "FAKE"
            p_final = max(p_final, face_p_fake * 0.9)

        # --------------------------------------------------------
        #          REAL gate + model-based escalation
        # --------------------------------------------------------
        real_gate_ok = real_gate(
            p_final,
            forensic_val,
            jpeg_q_score,
            hc_score,
            prnu_scaled,
            p_patch_spread,
        )
        if label == "REAL" and not real_gate_ok:
            label = "TAMPERED"

        if (
            label == "REAL"
            and (visual_prob > 0.65 or freq_prob > 0.65)
            and p_patch_mean > 0.60
        ):
            label = "FAKE" if p_final > 0.60 else "TAMPERED"
            override_label = label

        votes = tamper_votes(forensic_val, jpeg_q_score, hc_score)
        real_ok = real_pass(cfa_fake_score, prnu_scaled, real_prior_v3_val)
        if (
            label == "TAMPERED"
            and votes >= 2
            and forensic_val is not None
            and forensic_val > 0.70
            and (visual_prob > 0.65 or freq_prob > 0.65)
        ):
            label = "FAKE"
            override_label = "FAKE"
            p_final = max(p_final, 0.70)
        if label == "TAMPERED" and real_ok and votes < 2:
            label = "REAL"
            if override_label in (None, "TAMPERED"):
                override_label = "REAL"
        if label == "TAMPERED" and votes < 2:
            label = "REAL"
            if override_label in (None, "TAMPERED"):
                override_label = "REAL"
        if label == "REAL" and votes < 2 and override_label is None:
            override_label = "REAL"

        # --------------------------------------------------------
        #          Simplified band text (3-way classes)
        # --------------------------------------------------------
        # For the main display, we expose only:
        #   REAL, TAMPERED (RBR), or FAKE.
        if label not in ("INCONCLUSIVE", "UNCERTAIN"):
            if label == "REAL":
                band_text = "REAL"
            elif label in ("RBR", "RETOUCHED_REAL", "TAMPERED"):
                band_text = "TAMPERED"
            elif label == "FAKE":
                band_text = "FAKE"

        # ---- FINAL CONSISTENCY PASS (must be near the end) ----
        label, label_code, band_text, band_color, band, risk_level = finalize_label_and_risk(
            label, p_final, forensic_val, allow_real=real_gate_ok, override_label=override_label
        )
        if label == "TAMPERED" and votes < 2:
            label = "REAL"
            label_code = 0.0
            band_text, band_color, band, risk_level = traffic_light_label(
                label, p_final, forensic_val
            )
        # Final guard: force binary labels if configured.
        if (
            (DISABLE_INCONCLUSIVE and label in ("INCONCLUSIVE", "UNCERTAIN"))
            or (DISABLE_TAMPERED and label in ("TAMPERED", "RBR", "RETOUCHED_REAL"))
        ):
            label = "FAKE" if p_final >= FINAL_FAKE_THRESH else "REAL"
            label_code = 1.0 if label == "FAKE" else 0.0
            band_text, band_color, band, risk_level = traffic_light_label(
                label, p_final, forensic_val
            )

        # --------------------------------------------------------
        #          Verdict v2 (JPEG penalty + synthetic/edited split)
        # --------------------------------------------------------
        p_final_base = float(p_final)
        certainty_base = float(certainty)

        decision_payload = {
            "bayesian_fusion_posterior": p_final_base,
            "bayesian_fusion_certainty": certainty_base,
            "bayesian_fusion_calibrated": fusion_calibrated,
            "dirichlet_uncertainty": float(dirichlet_uncertainty) if dirichlet_uncertainty is not None else None,
            "dirichlet_conflict": float(dirichlet_conflict) if dirichlet_conflict is not None else None,
            "final_prob": p_final_base,
            "certainty": certainty_base,
            "visual_head": float(visual_prob),
            "freq_head": float(freq_prob),
            "forensic_score": float(forensic_val) if forensic_val is not None else None,
            "cfa_fake_score": float(cfa_fake_score) if cfa_fake_score is not None else None,
            "patch_mean": float(p_patch_mean),
            "prnu_strength_raw": float(prnu_val_raw),
            "jpeg_q_mismatch_score": float(jpeg_q_score) if jpeg_q_score is not None else None,
            "saturation_peak_score": float(sat_peak) if sat_peak is not None else None,
        }
        if isinstance(fusion_calibrated, dict) and "jpeg" in fusion_calibrated:
            decision_payload["jpeg"] = fusion_calibrated.get("jpeg")

        verdict = decide_verdict(decision_payload)
        verdict_ui = verdict_to_ui(verdict)

        label_v2 = verdict_ui.get("label_v2")
        verdict_reason = verdict_ui.get("reason", "")
        label = verdict_ui.get("prediction", label)
        p_final = float(verdict_ui.get("final_prob", p_final_base))
        certainty = float(verdict_ui.get("certainty", certainty_base))
        band = verdict_ui.get("band", band)
        risk_level = verdict_ui.get("risk_level", risk_level)
        band_text = verdict_band_text(band, risk_level)
        band_color = BAND_COLORS.get(band, band_color)

        forced_override = False
        if DISABLE_INCONCLUSIVE and label in ("INCONCLUSIVE", "UNCERTAIN"):
            label = "FAKE" if p_final >= FINAL_FAKE_THRESH else "REAL"
            forced_override = True
        if DISABLE_TAMPERED and label in ("TAMPERED", "RBR", "RETOUCHED_REAL"):
            label = "FAKE" if p_final >= FINAL_FAKE_THRESH else "REAL"
            forced_override = True
        if forced_override:
            band_text, band_color, band, risk_level = traffic_light_label(
                label, p_final, forensic_val
            )

        label_code = label_code_from_prediction(label)

        # --------------------------------------------------------
        #          Frequency-spectrum forensic panel (FFT)
        # --------------------------------------------------------
        fft_panel_img = None
        if not fast_mode:
            if "img_bgr" in locals() and img_bgr is not None:
                try:
                    fft_panel_img = forensic_panel(img_bgr)
                except Exception as _e:
                    print(f"[fft] forensic_panel error: {_e}")

            # Fallback: simple magnitude heatmap if advanced panel fails
            if fft_panel_img is None:
                _, F = fft_features(pil)
                fig, ax = plt.subplots(figsize=(3,3))
                im = ax.imshow(np.log(F + 1e-3), cmap="inferno")
                ax.set_xticks([])
                ax.set_yticks([])
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Log |FFT|", fontsize=8)
                sbuf = io.BytesIO()
                plt.tight_layout(pad=0.5)
                plt.savefig(sbuf, format="png")
                plt.close(fig)
                sbuf.seek(0)
                fft_panel_img = Image.open(sbuf)

        # --------------------------------------------------------
        #                    Heatmap overlay
        # --------------------------------------------------------
        if fast_mode:
            heatmap_img = None
        else:
            heatmap_img = (
                make_heatmap_overlay(pil, grid_scores)
                if grid_scores is not None else pil
            )

        # --------------------------------------------------------
        #                  Jitter collage preview
        # --------------------------------------------------------
        jitter_img = None if fast_mode else make_jitter_collage(pil, n=4, cols=2)

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
        bar_color = (
            "#6ef3a5" if label == "REAL"
            else "#ffd666" if label in ("RBR", "RETOUCHED_REAL", "TAMPERED")
            else "#9ca3af" if label in ("INCONCLUSIVE", "UNCERTAIN")
            else "#ff6b6b"
        )
        moe_text = f"{p_moe:.1%}" if p_moe is not None else "n/a"
        forensic_text = f"{forensic_val:.2f}" if forensic_val is not None else "n/a"
        cfa_text = f"{cfa_fake_score:.2f}" if cfa_fake_score is not None else "n/a"
        perlin_text = f"{perlin_score:.2f}" if perlin_score is not None else "n/a"
        texture_noise_text = f"{texture_noise:.2f}" if texture_noise is not None else "n/a"
        spectral_text = f"{spectral_score:.2f}" if spectral_score is not None else "n/a"
        color_text = f"{color_score:.2f}" if color_score is not None else "n/a"
        asym_text = f"{asym_score:.2f}" if asym_score is not None else "n/a"
        real_prior_text = f"{real_prior:.2f}" if real_prior is not None else "n/a"
        real_prior_v3_text = f"{real_prior_v3_val:.2f}" if real_prior_v3_val is not None else "n/a"
        grain_text = f"{grain_real:.2f}" if grain_real is not None else "n/a"
        fft_conf_text = f"{fft_conf_real:.2f}" if fft_conf_real is not None else "n/a"
        hc_text = f"{hc_score:.2f}" if hc_score is not None else "n/a"
        # Human-readable title for prediction (REAL / TAMPERED / FAKE)
        if label == "INCONCLUSIVE":
            display_label = "INCONCLUSIVE (insufficient evidence)"
        elif label == "UNCERTAIN":
            display_label = "UNCERTAIN (low confidence)"
        elif label == "REAL":
            display_label = "REAL (camera-native)"
        elif label in ("RBR", "RETOUCHED_REAL", "TAMPERED"):
            display_label = "TAMPERED (AI-enhanced / edited real photo)"
        else:
            display_label = "FAKE (synthetic / deepfake)"

        v2_label_text = f"<br>V2 verdict: {label_v2}" if label_v2 else ""
        v2_reason_text = f"<br>V2 reason: {verdict_reason}" if verdict_reason else ""

        certainty_warning = (
            "<br><b>WARNING: Low certainty (<20%) - manual review recommended.</b>"
            if certainty < 0.20
            else ""
        )
        code_str = f" &nbsp;|&nbsp; Code: {label_code}" if label_code is not None else ""
        html = (
            f"<span style='color:{bar_color};font-weight:bold'>Prediction: {display_label}</span>"
            f" &nbsp;|&nbsp; Band: <span style='color:{band_color};font-weight:bold'>{band_text}</span>"
            f"<br>Risk level: {risk_level}"
            f"{v2_label_text}"
            f"{v2_reason_text}"
            f"<br>Global prob: {p_global:.1%} &nbsp;|&nbsp; Final blended: {p_final:.1%}{code_str}"
            f"<br>Max patch: {p_patch_max:.1%} (mean {p_patch_mean:.1%}, spread {p_patch_spread:.1%})"
            f"<br>Visual head: {visual_prob:.1%} &nbsp;|&nbsp; Freq head: {freq_prob:.1%}"
            f"<br>MoE fusion: {moe_text}"
            f"<br>Forensic score: {forensic_text} (0=real, 1=fake)"
            f"<br>CFA fake score: {cfa_text} (0=real, 1=fake)"
            f"<br>Perlin diffusion score: {perlin_text} (0=real, 1=fake)"
            f"<br>Texture/noise score: {texture_noise_text} (0=real, 1=fake)"
            f"<br>Spectral flatness: {spectral_text} (0=real, 1=fake)"
            f"<br>Color correlation: {color_text} (0=real, 1=fake)"
            f"<br>Asymmetry score: {asym_text} (0=real, 1=fake)"
            f"<br>Real prior (combined): {real_prior_text} (0=fake, 1=real)"
            f"<br>Real prior v3: {real_prior_v3_text} (0=fake, 1=real)"
            f"<br>Grain likelihood: {grain_text} (0=fake, 1=real)"
            f"<br>Multiscale FFT confidence: {fft_conf_text} (0=fake, 1=real)"
            f"<br>Histogram consistency: {hc_text} (0=consistent, 1=anomalous)"
            f"<br>JPEG residual: {jpeg_score:.4f} &nbsp;|&nbsp; Embedding anomaly: {embed_score:.3f}"
            f"<br>Sharpness: {sharpness:.1f} &nbsp;|&nbsp; Head Delta: {head_delta:.3f}"
            f"<br>Certainty: {certainty:.1%}"
            "<br><div style='width:240px;background:#444;height:10px;border-radius:4px;margin-top:4px;'>"
            f"<div style='width:{int(p_final*240)}px;background:{bar_color};height:10px;border-radius:4px;'></div></div>"
            f"{suspicious_text}"
            f"{certainty_warning}"
            "<br><small>Note: This is a forensic risk estimate only - use context & provenance.</small>"
        )

        # Append human-readable confidence text
        html += "<br><b>" + confidence_text(certainty) + "</b>"

        # --------------------------------------------------------
        #                   JSON SUMMARY REPORT
        # --------------------------------------------------------
        report = {
            "prediction": label,
            "label_v2": label_v2,
            "band": band,
            "risk_level": risk_level,
            "final_prob": float(p_final),
            "decision_reason": verdict_reason,
            "global_prob": float(p_global),
            "bayesian_fusion_posterior": float(p_final_base),
            "bayesian_fusion_certainty": float(certainty_base),
            "bayesian_fusion_calibrated": fusion_calibrated,
            "bayesian_fusion_reliability": fusion_reliability,
            "dirichlet_posterior": float(dirichlet_p) if dirichlet_p is not None else None,
            "dirichlet_uncertainty": float(dirichlet_uncertainty) if dirichlet_uncertainty is not None else None,
            "dirichlet_conflict": float(dirichlet_conflict) if dirichlet_conflict is not None else None,
            "dirichlet_strength": float(dirichlet_strength) if dirichlet_strength is not None else None,
            "prediction_code": float(label_code) if label_code is not None else None,
            "certainty": float(certainty),
            "forensic_score": float(forensic_val) if forensic_val is not None else None,
            "perlin_score": float(perlin_score) if perlin_score is not None else None,
            "texture_noise_score": float(texture_noise) if texture_noise is not None else None,
            "cfa_fake_score": float(cfa_fake_score) if cfa_fake_score is not None else None,
            "spectral_flatness_score": float(spectral_score),
            "color_correlation_score": float(color_score),
            "asymmetry_score": float(asym_score) if asym_score is not None else None,
            "real_prior_v2": float(real_prior) if real_prior is not None else None,
            "real_image_prior_v2": float(real_prior_alt) if real_prior_alt is not None else None,
            "real_image_prior_v3": float(real_prior_v3_val) if real_prior_v3_val is not None else None,
            "real_image_prior_v4": float(real_prior_v4) if real_prior_v4 is not None else None,
            "prnu_strength_raw": float(prnu_val_raw),
            "prnu_strength_scaled": float(prnu_scaled),
            "prnu_fft_consistency": float(prnu_fft_cons),
            "jpeg_residual_v3": float(jpeg_resid_v3) if jpeg_resid_v3 is not None else None,
            "grain_likelihood": float(grain_real) if grain_real is not None else None,
            "multiscale_fft_confidence": float(fft_conf_real) if fft_conf_real is not None else None,
            "histogram_consistency": float(hc_score) if hc_score is not None else None,
            "esrgan_grid_score": float(esrgan_score) if esrgan_score is not None else None,
            "saturation_peak_score": float(sat_peak) if sat_peak is not None else None,
            "jpeg_q_mismatch_score": float(jpeg_q_score) if jpeg_q_score is not None else None,
            "face_retouch_score": float(face_retouch) if face_retouch is not None else None,
            "exposure_score": float(exposure_score) if exposure_score is not None else None,
            "rendering_pipeline_score": float(render_score),
            "image_gen_likelihood": float(image_gen_score),
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
            "label_v2": label_v2,
            "decision_reason": verdict_reason,
            "prediction_code": float(label_code) if label_code is not None else None,
            "final_prob": float(p_final),
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
            "color_correlation_score": float(color_score),
            "asymmetry_score": float(asym_score) if asym_score is not None else None,
            "real_prior_v2": float(real_prior) if real_prior is not None else None,
            "real_image_prior_v2": float(real_prior_alt) if real_prior_alt is not None else None,
            "real_image_prior_v3": float(real_prior_v3_val) if real_prior_v3_val is not None else None,
            "real_image_prior_v4": float(real_prior_v4) if real_prior_v4 is not None else None,
            "prnu_strength_raw": float(prnu_val_raw),
            "prnu_strength_scaled": float(prnu_scaled),
            "prnu_fft_consistency": float(prnu_fft_cons),
            "jpeg_residual_v3": float(jpeg_resid_v3) if jpeg_resid_v3 is not None else None,
            "grain_likelihood": float(grain_real) if grain_real is not None else None,
            "multiscale_fft_confidence": float(fft_conf_real) if fft_conf_real is not None else None,
            "histogram_consistency": float(hc_score) if hc_score is not None else None,
            "esrgan_grid_score": float(esrgan_score) if esrgan_score is not None else None,
            "saturation_peak_score": float(sat_peak) if sat_peak is not None else None,
            "jpeg_q_mismatch_score": float(jpeg_q_score) if jpeg_q_score is not None else None,
            "face_retouch_score": float(face_retouch) if face_retouch is not None else None,
            "exposure_score": float(exposure_score) if exposure_score is not None else None,
            "rendering_pipeline_score": float(render_score),
            "image_gen_likelihood": float(image_gen_score),
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
        explanation = explain_with_llm(llm_metrics) if generate_explanation else ""

        report_str = _json.dumps(report, indent=2)

        return html, heatmap_img, fft_panel_img, jitter_img, report_str, explanation

@spaces.GPU
def predict(
    media_path,
    video_frames,
    video_agg,
    video_topk_frac,
    strictness,
    video_scene_detect,
    video_adaptive_sample,
    temporal_weighting,
    llm_explain,
):
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    enable_llm = bool(llm_explain) and LLM_ENABLED

    # outputs must match GUI:
    # (html, heatmap, fft_panel, jitter, json_report, explanation, frame_table, frame_gallery)
    empty_table = []
    empty_gallery = []

    if media_path is None:
        return "No file uploaded.", None, None, None, "", "", empty_table, empty_gallery

    # If gr.File(type="filepath") is used, media_path is a string path.
    if isinstance(media_path, str) and _is_video_file(media_path):
        frames = extract_video_frames(
            media_path,
            max_frames=int(video_frames),
            scene_detect=video_scene_detect,
            adaptive_sample=video_adaptive_sample,
        )
        if not frames:
            return "Video decode error: no frames found.", None, None, None, "", "", empty_table, empty_gallery

        sora_fake_thresh_local = float(np.clip(SORA_FAKE_THRESH, 0.05, 0.98))
        sora_tampered_thresh_local = float(np.clip(SORA_TAMPERED_THRESH, 0.05, 0.95))
        if sora_fake_thresh_local <= sora_tampered_thresh_local:
            sora_fake_thresh_local = min(0.98, sora_tampered_thresh_local + 0.05)

        frame_temporal_scores = temporal_frame_scores(frames)
        if frame_temporal_scores and len(frame_temporal_scores) != len(frames):
            frame_temporal_scores = []
        if temporal_weighting and frame_temporal_scores:
            frame_temporal_weights = [0.6 + 1.0 * s for s in frame_temporal_scores]
        else:
            frame_temporal_weights = []

        # --------------------------------------------------------
        #          SORA DETECTION (GENERATOR ATTRIBUTION)
        # --------------------------------------------------------
        sora_likelihood = 0.0
        sora_flag = False
        id_drift = 0.0
        prnu_drift = 0.0
        prnu_flat_drift = 0.0
        parallax_err = 0.0
        face_drift = 0.0
        face_embed_drift = 0.0
        face_track_drift = 0.0
        object_inconsistency = 0.0
        background_inconsistency = 0.0
        texture_flicker = 0.0
        jpeg_drift = 0.0
        flow_err = 0.0
        flow_fb_inconsistency = 0.0
        flow_dir_incoherence = 0.0
        klt_instability = 0.0
        affine_inconsistency = 0.0
        edge_flicker = 0.0
        color_drift = 0.0
        noise_incoherence = 0.0
        spectral_drift = 0.0
        temporal_consistency_score = 0.0
        try:
            siglip = get_siglip_model()
            id_drift = temporal_identity_drift(frames, siglip)
            prnu_drift = prnu_temporal_incoherence(frames)
            prnu_flat_drift = prnu_temporal_incoherence_flat(frames)
            parallax_err = parallax_inconsistency(frames)
            face_drift = face_topology_drift(frames)
            face_embed_drift = face_embedding_drift(frames)
            face_track_drift = face_track_consistency(frames)
            object_inconsistency = object_identity_inconsistency(frames)
            background_inconsistency = background_temporal_inconsistency(frames)
            texture_flicker = temporal_texture_flicker(frames)
            jpeg_drift = jpeg_block_drift(frames)
            flow_err = flow_reprojection_error(frames)
            flow_fb_inconsistency = flow_forward_backward_inconsistency(frames)
            flow_dir_incoherence = flow_direction_incoherence(frames)
            klt_instability = klt_track_instability(frames)
            affine_inconsistency = affine_inlier_inconsistency(frames)
            edge_flicker = temporal_edge_flicker(frames)
            color_drift = temporal_color_drift(frames)
            noise_incoherence = noise_residual_incoherence(frames)
            spectral_drift = spectral_profile_drift(frames)

            signals = [
                ("id_drift", id_drift, 0.18),
                ("prnu_drift", prnu_drift, 0.14),
                ("prnu_flat_drift", prnu_flat_drift, 0.10),
                ("parallax_err", parallax_err, 0.09),
                ("face_topology_drift", face_drift, 0.07),
                ("face_embedding_drift", face_embed_drift, 0.07),
                ("face_track_drift", face_track_drift, 0.08),
                ("object_inconsistency", object_inconsistency, 0.08),
                ("background_inconsistency", background_inconsistency, 0.06),
                ("texture_flicker", texture_flicker, 0.04),
                ("flow_fb_inconsistency", flow_fb_inconsistency, 0.05),
                ("flow_dir_incoherence", flow_dir_incoherence, 0.03),
                ("klt_instability", klt_instability, 0.04),
                ("affine_inconsistency", affine_inconsistency, 0.02),
                ("jpeg_block_drift", jpeg_drift, 0.06),
            ]
            total_w = sum(w for _, v, w in signals if v is not None)
            if total_w > 0:
                sora_likelihood = float(
                    sum(w * float(v) for _, v, w in signals if v is not None) / total_w
                )

            general_signals = [
                ("flow_reprojection", flow_err, 0.12),
                ("flow_fb_inconsistency", flow_fb_inconsistency, 0.10),
                ("flow_dir_incoherence", flow_dir_incoherence, 0.08),
                ("parallax_err", parallax_err, 0.10),
                ("object_inconsistency", object_inconsistency, 0.09),
                ("background_inconsistency", background_inconsistency, 0.07),
                ("edge_flicker", edge_flicker, 0.07),
                ("texture_flicker", texture_flicker, 0.07),
                ("color_drift", color_drift, 0.05),
                ("noise_incoherence", noise_incoherence, 0.04),
                ("spectral_drift", spectral_drift, 0.04),
                ("klt_instability", klt_instability, 0.08),
                ("affine_inconsistency", affine_inconsistency, 0.06),
                ("prnu_flat_drift", prnu_flat_drift, 0.06),
                ("jpeg_block_drift", jpeg_drift, 0.07),
            ]
            total_gw = sum(w for _, v, w in general_signals if v is not None)
            if total_gw > 0:
                temporal_consistency_score = float(
                    sum(w * float(v) for _, v, w in general_signals if v is not None) / total_gw
                )
        except Exception as _e:
            print(f"[sora] error: {_e}")

        face_drift_val = float(face_drift) if face_drift is not None else 0.0
        face_embed_val = float(face_embed_drift) if face_embed_drift is not None else 0.0
        face_track_val = float(face_track_drift) if face_track_drift is not None else 0.0
        object_val = float(object_inconsistency) if object_inconsistency is not None else 0.0
        background_val = float(background_inconsistency) if background_inconsistency is not None else 0.0
        texture_val = float(texture_flicker) if texture_flicker is not None else 0.0

        core_hits = (
            int(id_drift > 0.55) +
            int(prnu_drift > 0.50) +
            int(prnu_flat_drift > 0.50) +
            int(face_drift_val > 0.50) +
            int(face_embed_val > 0.50) +
            int(face_track_val > 0.50)
        )
        motion_hits = (
            int(parallax_err > 0.50) +
            int(object_val > 0.60) +
            int(background_val > 0.60) +
            int(texture_val > 0.60) +
            int(flow_fb_inconsistency > 0.55) +
            int(flow_dir_incoherence > 0.55) +
            int(klt_instability > 0.55) +
            int(affine_inconsistency > 0.55)
        )
        sora_flag = bool(
            (sora_likelihood > 0.60 and core_hits >= 2)
            or (sora_likelihood > 0.75 and core_hits >= 1 and motion_hits >= 1)
        )

        max_workers = int(os.getenv("DETECT_VIDEO_WORKERS", "2"))
        if max_workers < 1:
            max_workers = 1
        max_workers = min(max_workers, len(frames))

        def _process_frame(idx, frame):
            try:
                html, heatmap_img, fft_panel_img, jitter_img, report_str, explanation = _predict_single_image(
                    frame,
                    fast_mode=True,
                    generate_explanation=False,
                )
            except Exception as e:
                print(f"[video] frame {idx} inference error: {e}")
                html, heatmap_img, fft_panel_img, jitter_img, report_str, explanation = "", None, None, None, "", ""

            try:
                rd = _json.loads(report_str) if report_str else {}
                p = float(rd.get("final_prob", 0.0))
                pred = rd.get("prediction", "INCONCLUSIVE")
            except Exception:
                p = 0.0
                pred = "INCONCLUSIVE"

            if DISABLE_TAMPERED and pred in ("TAMPERED", "RBR", "RETOUCHED_REAL"):
                pred = "FAKE" if p >= FINAL_FAKE_THRESH else "REAL"
            if DISABLE_INCONCLUSIVE and pred in ("INCONCLUSIVE", "UNCERTAIN"):
                pred = "FAKE" if p >= FINAL_FAKE_THRESH else "REAL"

            return {
                "idx": idx,
                "p": p,
                "pred": pred,
                "html": html,
                "heatmap": heatmap_img,
                "fft": fft_panel_img,
                "jitter": jitter_img,
                "report_str": report_str,
                "explanation": explanation,
                "frame": frame,
            }

        per_frame = []
        if max_workers > 1 and len(frames) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(_process_frame, idx, frame)
                    for idx, frame in enumerate(frames)
                ]
                for fut in as_completed(futures):
                    per_frame.append(fut.result())
        else:
            for idx, frame in enumerate(frames):
                per_frame.append(_process_frame(idx, frame))

        per_frame.sort(key=lambda x: x["idx"])
        frame_probs = [x["p"] for x in per_frame]
        frame_preds = [x["pred"] for x in per_frame]
        gallery_items = [
            (x["frame"], f"frame {x['idx']} | p_fake={x['p']:.2f} | {x['pred']}")
            for x in per_frame
        ]

        probs = np.array(frame_probs, dtype=np.float32)

        video_prob, video_label, chosen_idx, metrics = aggregate_video_probs(
            probs=probs,
            frame_preds=frame_preds,
            agg_mode=str(video_agg),
            topk_frac=float(video_topk_frac),
            strictness=str(strictness),
            min_agree=2,
            weights=frame_temporal_weights if frame_temporal_weights else None,
        )
        if 0.30 <= video_prob <= 0.45 and metrics.get("video_std", 0.0) < 0.03:
            video_label = "TAMPERED"

        chosen = per_frame[int(chosen_idx)]

        chosen_full = _predict_single_image(
            frames[int(chosen_idx)],
            fast_mode=False,
            generate_explanation=enable_llm,
        )
        html, heatmap_img, fft_panel_img, jitter_img, report_str, explanation = chosen_full

        # Build a frame table for the GUI
        frame_table = [[x["idx"], float(x["p"]), x["pred"]] for x in per_frame]

        # Merge JSON report: chosen frame detail + video summary
        try:
            report_data = _json.loads(report_str) if report_str else {}
        except Exception:
            report_data = {}
        chosen_pred = report_data.get("prediction")
        if video_label == "REAL" and chosen_pred in ("TAMPERED", "FAKE"):
            video_label = chosen_pred
            try:
                chosen_p = float(report_data.get("final_prob", video_prob))
                video_prob = max(video_prob, chosen_p)
            except Exception:
                pass

        if temporal_consistency_score > 0.75:
            odds = _odds(video_prob)
            odds *= 1.18
            video_prob = _from_odds(odds)
        elif temporal_consistency_score > 0.60:
            odds = _odds(video_prob)
            odds *= 1.12
            video_prob = _from_odds(odds)
        if temporal_consistency_score > 0.70 and video_label in ("REAL", "TAMPERED", "INCONCLUSIVE"):
            video_label = "TAMPERED"

        # Factor Sora attribution into final 3-way label
        if sora_likelihood >= sora_tampered_thresh_local:
            odds = _odds(video_prob)
            if sora_likelihood >= sora_fake_thresh_local:
                odds *= SORA_ODDS_HIGH
            elif sora_likelihood >= (sora_tampered_thresh_local + 0.15):
                odds *= SORA_ODDS_MED
            else:
                odds *= SORA_ODDS_LOW
            video_prob = _from_odds(odds)

        if sora_likelihood > sora_fake_thresh_local:
            video_label = "FAKE"
        elif sora_likelihood >= sora_tampered_thresh_local and video_label in ("REAL", "INCONCLUSIVE", "UNCERTAIN"):
            video_label = "TAMPERED"
        elif sora_flag and video_label in ("REAL", "INCONCLUSIVE", "UNCERTAIN"):
            video_label = "TAMPERED"

        report_data.update({
            "video_label": video_label,
            "video_prob": float(video_prob),
            "video_metrics": metrics,
            "video_selected_frame": int(chosen_idx),
            "video_total_sampled_frames": int(len(per_frame)),
            "video_frame_probs": [
                {
                    "frame_index": x["idx"],
                    "final_prob": float(x["p"]),
                    "prediction": x["pred"],
                    "temporal_score": float(frame_temporal_scores[x["idx"]]) if frame_temporal_scores else None,
                    "weight": float(frame_temporal_weights[x["idx"]]) if frame_temporal_weights else None,
                }
                for x in per_frame
            ],
            "sora_likelihood": float(sora_likelihood),
            "sora_flag": bool(sora_flag),
            "temporal_consistency_score": float(temporal_consistency_score),
            "analysis_config": {
                "scene_detect": bool(video_scene_detect),
                "adaptive_sample": bool(video_adaptive_sample),
                "temporal_weighting": bool(temporal_weighting),
                "sora_fake_thresh": float(sora_fake_thresh_local),
                "sora_tampered_thresh": float(sora_tampered_thresh_local),
            },
            "sora_signals": {
                "id_drift": float(id_drift),
                "prnu_drift": float(prnu_drift),
                "prnu_flat_drift": float(prnu_flat_drift),
                "parallax_err": float(parallax_err),
                "face_topology_drift": float(face_drift) if face_drift is not None else None,
                "face_embedding_drift": float(face_embed_drift) if face_embed_drift is not None else None,
                "face_track_drift": float(face_track_drift) if face_track_drift is not None else None,
                "object_inconsistency": float(object_inconsistency) if object_inconsistency is not None else None,
                "background_inconsistency": float(background_inconsistency) if background_inconsistency is not None else None,
                "texture_flicker": float(texture_flicker),
                "flow_fb_inconsistency": float(flow_fb_inconsistency),
                "flow_dir_incoherence": float(flow_dir_incoherence),
                "klt_instability": float(klt_instability),
                "affine_inconsistency": float(affine_inconsistency),
                "jpeg_block_drift": float(jpeg_drift),
            },
            "temporal_signals": {
                "flow_reprojection_error": float(flow_err),
                "flow_fb_inconsistency": float(flow_fb_inconsistency),
                "flow_dir_incoherence": float(flow_dir_incoherence),
                "edge_flicker": float(edge_flicker),
                "color_drift": float(color_drift),
                "texture_flicker": float(texture_flicker),
                "noise_incoherence": float(noise_incoherence),
                "spectral_drift": float(spectral_drift),
                "klt_instability": float(klt_instability),
                "affine_inconsistency": float(affine_inconsistency),
                "prnu_flat_drift": float(prnu_flat_drift),
                "jpeg_block_drift": float(jpeg_drift),
                "object_inconsistency": float(object_inconsistency) if object_inconsistency is not None else None,
                "background_inconsistency": float(background_inconsistency) if background_inconsistency is not None else None,
                "parallax_err": float(parallax_err),
            },
        })
        report_str = _json.dumps(report_data, indent=2)

        header = (
            f"[VIDEO] decision={video_label} | score={video_prob:.1%} | "
            f"std={metrics.get('video_std', 0.0):.2f} | "
            f"fake_frames={metrics.get('n_fake_frames', 0)}/{metrics.get('n', 0)} | "
            f"chosen_frame={chosen_idx}"
        )

        html = header + "<br>" + (html or "")
        if enable_llm:
            explanation = f"{header}\n\n" + (explanation or "")
            if sora_flag:
                explanation += (
                    "\n\nTemporal identity drift, PRNU incoherence, face instability, and "
                    "motion/background inconsistencies are consistent with modern "
                    "generative video (e.g., Sora-like systems)."
                )
            if temporal_consistency_score > 0.65:
                explanation += (
                    "\n\nTemporal consistency anomalies detected (motion and appearance drift)."
                )
        else:
            explanation = ""

        return (
            html,
            heatmap_img,
            fft_panel_img,
            jitter_img,
            report_str,
            explanation,
            frame_table,
            gallery_items,
        )

    # Non-video: image file path or PIL
    html, heatmap_img, fft_panel_img, jitter_img, report_str, explanation = _predict_single_image(
        media_path,
        fast_mode=False,
        generate_explanation=enable_llm,
    )
    return html, heatmap_img, fft_panel_img, jitter_img, report_str, explanation, empty_table, empty_gallery

# ============================================================
#                        GRADIO UI
# ============================================================

import json, tempfile, os
import math
import gradio as gr

def verdict_color(label: str):
    return {
        "REAL": "#1eae63",
        "TAMPERED": "#f4b400",
        "FAKE": "#e03131",
        "INCONCLUSIVE": "#f4b400",
        "UNCERTAIN": "#f4b400",
    }.get(label, "#9ca3af")


def normalize_label_for_ui(label: str):
    if label in ("INCONCLUSIVE", "UNCERTAIN"):
        return "TAMPERED", "Low confidence"
    if label in ("RBR", "RETOUCHED_REAL"):
        return "TAMPERED", ""
    if label not in ("REAL", "TAMPERED", "FAKE"):
        return "TAMPERED", ""
    return label, ""


def status_chip(text: str, kind: str = ""):
    if kind:
        return f'<span class="chip chip-{kind}">{text}</span>'
    return f'<span class="chip">{text}</span>'


def metrics_strip(p_fake, certainty, temporal_score, sora_score, is_video=False):
    p_fake = float(p_fake or 0.0)
    certainty = float(certainty or 0.0)
    temporal_score = float(temporal_score or 0.0)
    sora_score = float(sora_score or 0.0)

    def _fmt(val, active=True):
        if not active:
            return "n/a"
        return f"{val * 100:.1f}%"

    temporal_active = bool(is_video)
    sora_active = True if sora_score is not None else bool(is_video)
    gen_label = "sora-like" if is_video else "gen-like"
    return f"""
    <div class="metrics-strip">
        <div class="metric-box">
            <div class="metric-label">p(fake)</div>
            <div class="metric-value">{_fmt(p_fake, True)}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">certainty</div>
            <div class="metric-value">{_fmt(certainty, True)}</div>
        </div>
        <div class="metric-box{' muted' if not temporal_active else ''}">
            <div class="metric-label">temporal</div>
            <div class="metric-value">{_fmt(temporal_score, temporal_active)}</div>
        </div>
        <div class="metric-box{' muted' if not sora_active else ''}">
            <div class="metric-label">{gen_label}</div>
            <div class="metric-value">{_fmt(sora_score, sora_active)}</div>
        </div>
    </div>
    """


def timeline_chart(frame_rows, sora_score=0.0, temporal_global=0.0):
    if not frame_rows:
        return ""

    p_vals = []
    t_vals = []
    for row in frame_rows:
        try:
            p_vals.append(float(row.get("final_prob", 0.0)))
            t_val = row.get("temporal_score", 0.0)
            t_vals.append(float(t_val) if t_val is not None else 0.0)
        except Exception:
            p_vals.append(0.0)
            t_vals.append(0.0)

    n = len(p_vals)
    if n == 0:
        return ""

    width = 640
    height = 160
    pad = 20
    step = (width - 2 * pad) / max(1, n - 1)

    def _xy(i, v):
        x = pad + i * step
        y = pad + (1.0 - float(np.clip(v, 0.0, 1.0))) * (height - 2 * pad)
        return x, y

    pf_points = " ".join(f"{_xy(i, v)[0]:.1f},{_xy(i, v)[1]:.1f}" for i, v in enumerate(p_vals))
    tf_points = " ".join(f"{_xy(i, v)[0]:.1f},{_xy(i, v)[1]:.1f}" for i, v in enumerate(t_vals))

    top_pf = sorted(range(n), key=lambda i: p_vals[i], reverse=True)[:2]
    top_tf = sorted(range(n), key=lambda i: t_vals[i], reverse=True)[:1]

    sora_y = _xy(0, sora_score)[1]
    temporal_global_y = _xy(0, temporal_global)[1]

    dots = []
    for i in top_pf:
        x, y = _xy(i, p_vals[i])
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" class="dot-worst"/>')
    for i in top_tf:
        x, y = _xy(i, t_vals[i])
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.0" class="dot-temporal"/>')

    svg = f"""
    <svg viewBox="0 0 {width} {height}" class="timeline-svg" preserveAspectRatio="none">
        <line x1="{pad}" y1="{sora_y:.1f}" x2="{width - pad}" y2="{sora_y:.1f}" class="line-sora"/>
        <line x1="{pad}" y1="{temporal_global_y:.1f}" x2="{width - pad}" y2="{temporal_global_y:.1f}" class="line-temporal-global"/>
        <polyline points="{pf_points}" class="line-pfake"/>
        <polyline points="{tf_points}" class="line-temporal"/>
        {''.join(dots)}
    </svg>
    """

    return f"""
    <div class="timeline-card">
        <div class="timeline-head">
            <div class="timeline-title">Frame timeline</div>
            <div class="timeline-legend">
                <span><i class="swatch pfake"></i>p(fake)</span>
                <span><i class="swatch temporal"></i>Temporal</span>
                <span><i class="swatch sora"></i>Sora</span>
            </div>
        </div>
        {svg}
    </div>
    """


def verdict_card(label: str, p_fake: float, certainty: float = 0.0, note: str = ""):
    p_fake = float(p_fake or 0.0)
    certainty = float(certainty or 0.0)

    color = verdict_color(label)
    note_html = f'<div class="verdict-note">{note}</div>' if note else ""
    return f"""
    <div class="verdict-wrap" data-label="{label}" style="--accent:{color};">
        <div class="verdict-kicker">Decision</div>
        <div class="verdict-title">{label}</div>
        <div class="verdict-sub">Primary label for this media</div>
        <div class="verdict-metrics">
            <div class="metric">
                <span>p(fake)</span>
                <strong>{p_fake*100:.1f}%</strong>
            </div>
            <div class="metric">
                <span>certainty</span>
                <strong>{certainty*100:.1f}%</strong>
            </div>
        </div>
        {note_html}
    </div>
    """


def prob_gauge(p: float, certainty: float = 0.0):
    p = float(p or 0.0)
    certainty = float(certainty or 0.0)
    return f"""
    <div class="prob-wrap">
      <div class="prob-kicker">Probability scale</div>
      <div class="prob-bar">
        <div class="prob-gradient"></div>
        <div class="prob-marker" style="left:{p*100:.1f}%"></div>
      </div>
      <div class="prob-labels">
        <span>Real</span>
        <span>Tampered</span>
        <span>Fake</span>
      </div>
      <div class="prob-meta">p(fake) {p*100:.1f}% - certainty {certainty*100:.1f}%</div>
    </div>
    """


def save_json_report(text):
    if not text:
        return None
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


def clear_all():
    return (
        None, "", "", "", "", "", "",    # file, verdict, gauge, metrics, timeline, status, explanation
        None, None, None,        # heatmap, fft, jitter
        "",                      # json
        [], [],                  # frame table, gallery
        None, None,              # image preview, video preview
    )

with gr.Blocks(
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;700&family=Lexend:wght@300;400;600;700&display=swap');
    :root {
        --bg: #050505;
        --ink: #f8fafc;
        --muted: #94a3b8;
        --card: #0f1115;
        --stroke: #1f2937;
        --stroke-soft: #2a3442;
        --accent: #22d3ee;
        --accent-2: #f97316;
        --real: #22c55e;
        --tampered: #f59e0b;
        --fake: #ef4444;
    }
    body {
        background:
            radial-gradient(900px 520px at 8% 10%, rgba(34, 211, 238, 0.18) 0%, transparent 60%),
            radial-gradient(900px 520px at 92% 0%, rgba(249, 115, 22, 0.16) 0%, transparent 60%),
            radial-gradient(900px 520px at 40% 85%, rgba(34, 197, 94, 0.14) 0%, transparent 60%),
            #050505;
        color: var(--ink);
    }
    .gradio-container {
        font-family: "Lexend", sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 18px 14px 32px;
        position: relative;
        z-index: 0;
        overflow: visible;
        color: var(--ink);
    }
    .gradio-container::before {
        content: "";
        position: absolute;
        top: -90px;
        right: -140px;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.35), transparent 70%);
        z-index: -1;
        pointer-events: none;
    }
    .gradio-container::after {
        content: "";
        position: absolute;
        bottom: -120px;
        left: -120px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(249, 115, 22, 0.25), transparent 70%);
        z-index: -1;
        pointer-events: none;
    }
    h1, h2, h3, .hero-title {
        font-family: "Bricolage Grotesque", sans-serif;
        letter-spacing: 0.4px;
    }
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        background: linear-gradient(120deg, rgba(15, 23, 42, 0.92), rgba(3, 7, 18, 0.95));
        border: 1px solid var(--stroke);
        border-radius: 20px;
        padding: 16px 20px;
        box-shadow: 0 18px 30px rgba(0, 0, 0, 0.45);
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: "";
        position: absolute;
        inset: -60% -10% auto auto;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(248, 113, 113, 0.35), transparent 70%);
        opacity: 0.6;
        z-index: 0;
    }
    .hero > div {
        position: relative;
        z-index: 1;
    }
    .hero-title {
        font-size: 34px;
        margin: 0;
    }
    .hero-sub {
        color: var(--muted);
        font-weight: 600;
        font-size: 13px;
        margin-top: 4px;
    }
    .hero-tag {
        font-size: 11px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #0b0b0b;
        border: 1px solid var(--stroke);
        padding: 6px 12px;
        border-radius: 999px;
        background: linear-gradient(120deg, #22d3ee, #f97316, #22c55e);
        box-shadow: 0 8px 16px rgba(34, 211, 238, 0.25);
        position: relative;
        z-index: 1;
    }
    #panel-input, #panel-results {
        background: linear-gradient(180deg, #0f1115 0%, #0b0d12 100%);
        border: 1px solid var(--stroke-soft);
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 16px 28px rgba(0, 0, 0, 0.45);
    }
    .section-title {
        font-size: 12px;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 12px;
    }
    #verdict, #gauge {
        width: 100%;
        flex: 1 1 0;
    }
    #verdict .verdict-wrap,
    #gauge .prob-wrap {
        height: 100%;
    }
    .verdict-wrap {
        background: linear-gradient(160deg, #0b0f14 0%, rgba(15, 23, 42, 0.9) 120%);
        border: 1px solid var(--stroke-soft);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 20px 32px rgba(0, 0, 0, 0.55);
        position: relative;
        overflow: hidden;
        animation: riseIn 0.45s ease both;
    }
    .verdict-wrap::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, var(--accent-soft), transparent);
        opacity: 0.6;
    }
    .verdict-wrap::after {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 8px;
        background: var(--accent);
    }
    .verdict-wrap > * {
        position: relative;
        z-index: 1;
    }
    .verdict-wrap[data-label="REAL"] {
        --accent: var(--real);
        --accent-soft: rgba(34, 197, 94, 0.2);
    }
    .verdict-wrap[data-label="TAMPERED"] {
        --accent: var(--tampered);
        --accent-soft: rgba(245, 158, 11, 0.2);
    }
    .verdict-wrap[data-label="FAKE"] {
        --accent: var(--fake);
        --accent-soft: rgba(239, 68, 68, 0.2);
    }
    .verdict-kicker {
        font-size: 11px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .verdict-title {
        font-family: "Bricolage Grotesque", sans-serif;
        font-size: 32px;
        margin: 4px 0 0;
    }
    .verdict-sub {
        margin-top: 4px;
        font-size: 13px;
        color: var(--muted);
    }
    .verdict-metrics {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 12px;
    }
    .metric {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(148, 163, 184, 0.35);
        border-radius: 12px;
        padding: 8px 10px;
    }
    .metric span {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--muted);
    }
    .metric strong {
        display: block;
        font-size: 16px;
        color: var(--ink);
        margin-top: 4px;
    }
    .verdict-note {
        margin-top: 10px;
        font-size: 12px;
        color: var(--muted);
    }
    .prob-wrap {
        background: linear-gradient(180deg, #0f1115 0%, #0b0d12 100%);
        border: 1px solid var(--stroke-soft);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 18px 30px rgba(0, 0, 0, 0.55);
        animation: riseIn 0.55s ease both;
    }
    .prob-kicker {
        font-size: 11px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 10px;
    }
    .prob-bar {
        position: relative;
        height: 12px;
        border-radius: 999px;
        background: #1f2937;
        overflow: visible;
        margin-bottom: 10px;
    }
    .prob-gradient {
        position: absolute;
        inset: 0;
        background: linear-gradient(90deg, var(--real), var(--tampered), var(--fake));
        border-radius: inherit;
    }
    .prob-marker {
        position: absolute;
        top: -6px;
        width: 3px;
        height: 24px;
        background: #f8fafc;
        transform: translateX(-50%);
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: var(--muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }
    .prob-meta {
        margin-top: 8px;
        font-size: 12px;
        color: var(--muted);
        font-weight: 600;
    }
    .metrics-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 8px 0 4px;
    }
    .metric-box {
        border: 1px solid var(--stroke-soft);
        border-radius: 14px;
        background: #0f1115;
        padding: 10px;
        box-shadow: 0 12px 18px rgba(0, 0, 0, 0.45);
        animation: riseIn 0.55s ease both;
    }
    .metrics-strip .metric-box:nth-child(1) { animation-delay: 0.05s; }
    .metrics-strip .metric-box:nth-child(2) { animation-delay: 0.10s; }
    .metrics-strip .metric-box:nth-child(3) { animation-delay: 0.15s; }
    .metrics-strip .metric-box:nth-child(4) { animation-delay: 0.20s; }
    .metrics-strip .metric-box:nth-child(1) {
        border-color: rgba(239, 68, 68, 0.5);
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(239, 68, 68, 0.08) 100%);
    }
    .metrics-strip .metric-box:nth-child(2) {
        border-color: rgba(34, 211, 238, 0.5);
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.2) 0%, rgba(34, 211, 238, 0.08) 100%);
    }
    .metrics-strip .metric-box:nth-child(3) {
        border-color: rgba(245, 158, 11, 0.5);
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.08) 100%);
    }
    .metrics-strip .metric-box:nth-child(4) {
        border-color: rgba(34, 197, 94, 0.5);
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.08) 100%);
    }
    .timeline-card {
        border: 1px solid var(--stroke-soft);
        border-radius: 16px;
        padding: 12px 14px;
        background: #0b0d12;
        box-shadow: 0 12px 18px rgba(0, 0, 0, 0.4);
        margin: 8px 0 12px;
    }
    .timeline-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 6px;
    }
    .timeline-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        color: var(--muted);
        font-weight: 700;
    }
    .timeline-legend {
        display: flex;
        gap: 10px;
        font-size: 10px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.18em;
    }
    .timeline-legend span {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .swatch {
        width: 10px;
        height: 10px;
        border-radius: 3px;
        display: inline-block;
    }
    .swatch.pfake { background: #ef4444; }
    .swatch.temporal { background: #f59e0b; }
    .swatch.sora { background: #22d3ee; }
    .timeline-svg {
        width: 100%;
        height: 160px;
    }
    .line-pfake {
        fill: none;
        stroke: #ef4444;
        stroke-width: 2.2;
    }
    .line-temporal {
        fill: none;
        stroke: #f59e0b;
        stroke-width: 2.0;
    }
    .line-sora {
        stroke: #22d3ee;
        stroke-width: 1.4;
        stroke-dasharray: 5 5;
        opacity: 0.8;
    }
    .line-temporal-global {
        stroke: #f59e0b;
        stroke-width: 1.2;
        stroke-dasharray: 3 4;
        opacity: 0.6;
    }
    .dot-worst {
        fill: #ef4444;
        stroke: #f8fafc;
        stroke-width: 1.2;
    }
    .dot-temporal {
        fill: #f59e0b;
        stroke: #f8fafc;
        stroke-width: 1.2;
    }
    .metric-box.muted {
        background: #111827;
        color: var(--muted);
        border-color: var(--stroke-soft);
        box-shadow: none;
    }
    .metric-label {
        font-size: 10px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .metric-value {
        font-family: "Bricolage Grotesque", sans-serif;
        font-size: 22px;
        margin-top: 4px;
    }
    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 6px;
    }
    .chip {
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid var(--stroke-soft);
        background: rgba(15, 23, 42, 0.75);
        font-size: 10px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 600;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.35);
    }
    .chip-warn {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.35) 0%, rgba(245, 158, 11, 0.15) 100%);
        border-color: rgba(245, 158, 11, 0.5);
        color: #fcd34d;
    }
    .chip-sora {
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.3) 0%, rgba(34, 197, 94, 0.12) 100%);
        border-color: rgba(34, 211, 238, 0.45);
        color: #67e8f9;
    }
    #status {
        margin-bottom: 6px;
    }
    #explanation {
        background: linear-gradient(180deg, #0f1115 0%, #0b0d12 100%);
        border: 1px solid var(--stroke-soft);
        border-radius: 14px;
        padding: 12px 14px;
        color: var(--muted);
        font-size: 13px;
        box-shadow: 0 10px 18px rgba(0, 0, 0, 0.45);
    }
    .gradio-container button {
        border-radius: 14px;
        font-weight: 700;
        padding: 10px 14px;
        border: 1px solid var(--stroke-soft);
        background: #0f1115;
        color: var(--ink);
        box-shadow: 0 8px 14px rgba(0, 0, 0, 0.4);
    }
    .gradio-container button.primary {
        background: linear-gradient(120deg, #22d3ee, #f97316, #22c55e);
        border: none;
        color: #0b0b0b;
        box-shadow: 0 14px 24px rgba(34, 211, 238, 0.35);
    }
    .gradio-container button.primary:hover {
        filter: brightness(0.95);
    }
    .gradio-container input,
    .gradio-container select,
    .gradio-container textarea {
        border-radius: 12px;
        border-color: var(--stroke-soft) !important;
        background: #0f1115;
        color: var(--ink);
    }
    .tabs {
        margin-top: 10px;
    }
    .tabitem {
        border-radius: 999px !important;
        padding: 6px 14px !important;
        font-weight: 700;
        border: 1px solid var(--stroke-soft) !important;
        background: #0f1115 !important;
        color: var(--ink) !important;
    }
    .tabitem.selected {
        background: linear-gradient(120deg, #22d3ee, #f97316, #22c55e) !important;
        border-color: rgba(34, 211, 238, 0.5) !important;
        color: #0b0b0b !important;
    }
    table {
        border-collapse: collapse;
        color: var(--ink);
        background: #0b0d12;
    }
    table thead th {
        background: #0f172a;
        color: var(--ink);
        font-weight: 700;
        border-bottom: 1px solid var(--stroke-soft) !important;
    }
    table td, table th {
        border-color: var(--stroke-soft) !important;
        border-width: 1px !important;
        color: var(--ink);
        background: #0b0d12;
    }
    img {
        border-radius: 14px;
        border: 1px solid var(--stroke-soft);
        background: #0b0b0b;
    }
    textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 12px;
        color: var(--ink);
    }
    @keyframes riseIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 900px) {
        #panel-input, #panel-results { padding: 12px; }
        .hero-title { font-size: 28px; }
        .hero { flex-direction: column; align-items: flex-start; }
        .metrics-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    """
) as demo:

    gr.HTML("""
    <div class="hero" id="hero">
        <div>
            <div class="hero-title">Deepfake Detector</div>
            <div class="hero-sub">REAL / TAMPERED / FAKE - image and video forensics</div>
        </div>
        <div class="hero-tag">Forensic triage</div>
    </div>
    """)

    with gr.Row(equal_height=True, elem_id="main-row"):

        # ================= LEFT =================
        with gr.Column(scale=1, elem_id="panel-input"):

            gr.HTML('<div class="section-title">Input</div>')
            file_in = gr.File(
                label="Upload image or video",
                type="filepath",
                file_types=[
                    ".jpg",".jpeg",".png",".webp",".bmp",".avif",
                    ".mp4",".mov",".avi",".mkv",".webm",".mpeg",
                ]
            )

            image_preview = gr.Image(visible=False)
            video_preview = gr.Video(visible=False)

            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.Button("Clear")

            with gr.Accordion("Advanced video settings", open=False):
                video_frames = gr.Slider(4, 32, value=12, step=1, label="Sample frames")
                video_agg = gr.Dropdown(
                    ["median"],
                    value="median",
                    label="Aggregation",
                    interactive=False,
                )
                strictness = gr.Dropdown(
                    ["conservative", "balanced", "aggressive"],
                    value="balanced",
                    label="Strictness",
                )
                with gr.Row():
                    scene_detect = gr.Checkbox(
                        value=VIDEO_SCENE_DETECT,
                        label="Scene detection",
                    )
                    adaptive_sample = gr.Checkbox(
                        value=VIDEO_ADAPTIVE_SAMPLE,
                        label="Adaptive sampling",
                    )
                with gr.Row():
                    temporal_weighting = gr.Checkbox(
                        value=True,
                        label="Temporal weighting",
                    )
                    llm_explain = gr.Checkbox(
                        value=LLM_ENABLED,
                        label="LLM explanation",
                    )

        # ================= RIGHT =================
        with gr.Column(scale=2, elem_id="panel-results"):

            gr.HTML('<div class="section-title">Results</div>')

            with gr.Row():
                verdict_html = gr.HTML(elem_id="verdict")
                gauge_html = gr.HTML(elem_id="gauge")
            metrics_html = gr.HTML(elem_id="metrics")
            timeline_html = gr.HTML(elem_id="timeline")
            status_md = gr.HTML(elem_id="status")
            explanation_md = gr.Markdown(elem_id="explanation")

            with gr.Tabs():

                with gr.Tab("Maps"):
                    with gr.Row():
                        heatmap_out = gr.Image(label="Suspicious regions")
                        fft_out = gr.Image(label="FFT / forensic")
                        jitter_out = gr.Image(label="Jitter")

                with gr.Tab("Video"):
                    frames_table = gr.Dataframe(
                        headers=["frame", "p_fake", "label"],
                        interactive=False,
                    )
                    gallery_out = gr.Gallery(columns=4)

                with gr.Tab("Report"):
                    json_out = gr.Textbox(lines=16)
                    download_btn = gr.DownloadButton("Download JSON")

    # -------- Preview logic --------
    def _preview(path):
        if path and _is_video_file(path):
            return (
                gr.update(visible=False),
                gr.update(value=path, visible=True),
            )
        return (
            gr.update(value=path, visible=True),
            gr.update(visible=False),
        )

    file_in.change(_preview, file_in, [image_preview, video_preview])

    # -------- Main action --------
    def run_all(
        path,
        vf,
        agg,
        strict,
        scene_on,
        adaptive_on,
        weighting_on,
        llm_on,
    ):
        (
            html,
            heatmap,
            fft,
            jitter,
            report,
            explanation,
            table,
            gallery,
        ) = predict(
            path,
            vf,
            agg,
            0.30,
            strict,
            scene_on,
            adaptive_on,
            weighting_on,
            llm_on,
        )

        label = "INCONCLUSIVE"
        p = 0.0
        certainty = 0.0
        status_items = [status_chip("Mode: Image")]
        sora_flag = False
        sora_likelihood = 0.0
        image_gen_score = 0.0
        temporal_consistency_score = 0.0
        is_video = False
        try:
            r = json.loads(report)
            label = r.get("video_label", r.get("prediction", label))
            p = r.get("video_prob", r.get("final_prob", 0.0))
            certainty = float(r.get("certainty", 0.0) or 0.0)
            if "video_total_sampled_frames" in r:
                is_video = True
                status_items = [
                    status_chip("Mode: Video"),
                    status_chip(f"Frames: {r.get('video_total_sampled_frames')}"),
                    status_chip(f"Aggregation: {agg}"),
                    status_chip(f"Strictness: {strict}"),
                ]
                sora_likelihood = float(r.get("sora_likelihood", 0.0) or 0.0)
                sora_flag = bool(r.get("sora_flag", False))
                temporal_consistency_score = float(r.get("temporal_consistency_score", 0.0) or 0.0)
            else:
                image_gen_score = float(r.get("image_gen_likelihood", 0.0) or 0.0)
        except Exception:
            pass
        if DISABLE_TAMPERED and label in ("TAMPERED", "RBR", "RETOUCHED_REAL"):
            label = "FAKE" if p >= FINAL_FAKE_THRESH else "REAL"
        if DISABLE_INCONCLUSIVE and label in ("INCONCLUSIVE", "UNCERTAIN"):
            label = "FAKE" if p >= FINAL_FAKE_THRESH else "REAL"

        label_ui, note = normalize_label_for_ui(label)
        if note:
            status_items.append(status_chip(note, "warn"))
        if sora_flag:
            status_items.append(status_chip(f"Sora-like {sora_likelihood*100:.0f}%", "sora"))
        if temporal_consistency_score > 0.65:
            status_items.append(status_chip(f"Temporal {temporal_consistency_score*100:.0f}%", "warn"))
        if not is_video and image_gen_score >= IMAGE_GEN_TAMPERED_THRESH:
            status_items.append(status_chip(f"Gen-like {image_gen_score*100:.0f}%", "sora"))
        if is_video:
            if scene_on:
                status_items.append(status_chip("Scene detect"))
            if adaptive_on:
                status_items.append(status_chip("Adaptive sample"))
            if weighting_on:
                status_items.append(status_chip("Weighted agg"))

        status = f"<div class='status-row'>{''.join(status_items)}</div>"
        timeline = ""
        if is_video:
            try:
                frame_rows = r.get("video_frame_probs", [])
                timeline = timeline_chart(frame_rows, sora_likelihood, temporal_consistency_score)
            except Exception:
                timeline = ""

        return (
            verdict_card(label_ui, p, certainty, note),
            prob_gauge(p, certainty),
            metrics_strip(
                p,
                certainty,
                temporal_consistency_score,
                sora_likelihood if is_video else image_gen_score,
                is_video,
            ),
            timeline,
            status,
            explanation,
            heatmap, fft, jitter,
            report,
            table, gallery,
            save_json_report(report),
        )

    analyze_btn.click(
        run_all,
        inputs=[
            file_in,
            video_frames,
            video_agg,
            strictness,
            scene_detect,
            adaptive_sample,
            temporal_weighting,
            llm_explain,
        ],
        outputs=[
            verdict_html,
            gauge_html,
            metrics_html,
            timeline_html,
            status_md,
            explanation_md,
            heatmap_out, fft_out, jitter_out,
            json_out,
            frames_table, gallery_out,
            download_btn,
        ],
        concurrency_limit=1,
    )

    clear_btn.click(
        clear_all,
        outputs=[
            file_in,
            verdict_html,
            gauge_html,
            metrics_html,
            timeline_html,
            status_md,
            explanation_md,
            heatmap_out, fft_out, jitter_out,
            json_out,
            frames_table, gallery_out,
            image_preview, video_preview,
        ]
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
    demo.queue(max_size=20)
    demo.launch(show_error=True)
# paste EVERYTHING from the code block here
