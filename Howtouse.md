

````markdown
# Deepfake Detector v5.0 (Experimental)

SigLIP + upgraded frequency forensics + adaptive fusion + CORAL risk calibration.

This repo contains:

- `app.py` — Gradio app / HF Space entrypoint (v5.0 detector)
- `train_freq_mlp_v5.py` — trains the upgraded frequency MLP (`freq_mlp.safetensors`)
- `train_fusion_head_only.py` — trains the adaptive fusion head (`fusion_head.safetensors`)
- `fit_coral_v5.py` — fits CORAL cutpoints for ordinal risk bands

The detector combines:

1. **SigLIP visual expert** (pretrained + fine-tuned)
2. **Frequency MLP forensic expert** (FFT + SRM + wavelets)
3. **Adaptive fusion head** (learned mix of SigLIP + freq)
4. **CORAL** (ordinal risk calibration into REAL → FAKE bands)

---

## 1. Repo structure (expected)

Typical layout:

```text
.
├─ app.py
├─ train_freq_mlp_v5.py
├─ train_fusion_head_only.py
├─ fit_coral_v5.py
├─ README.md
└─ models/
   ├─ best_model.safetensors      # SigLIP visual deepfake head
   ├─ freq_mlp.safetensors        # (output of train_freq_mlp_v5.py)
   ├─ fusion_head.safetensors     # (output of train_fusion_head_only.py)
   ├─ coral_cutpoints.json        # (output of fit_coral_v5.py)
   ├─ coral_temp.json             # (output of fit_coral_v5.py)
   └─ coral_bins.npy              # (output of fit_coral_v5.py)
````

On Hugging Face, those `.safetensors` + CORAL files should live in the model repo you point `MODEL_REPO` to (e.g. `joesound212985/siglip`).

---

## 2. Installation

Python 3.10+ recommended.

```bash
pip install torch torchvision safetensors pywavelets scikit-learn tqdm opencv-python
pip install open-clip-torch
pip install gradio matplotlib
pip install huggingface_hub  # if pulling models from HF
```

---

## 3. Data layout

You need two folders of images:

```text
/path/to/Real-img/Real-img   # all REAL images
/path/to/Fake-img/Image      # all FAKE images
```

Any common image formats are supported: `.jpg, .jpeg, .png, .webp, .heic, ...`.

---

## 4. Step 1 — SigLIP visual head (`best_model.safetensors`)

This repo assumes you already have:

```text
models/best_model.safetensors
```

This is your SigLIP + MLP binary classifier trained on deepfake vs real.
`app.py` and the fusion trainer treat this as fixed.

If you don’t have it yet, train SigLIP separately and save the result as `best_model.safetensors`.

---

## 5. Step 2 — Train the upgraded Frequency MLP

Script: `train_freq_mlp_v5.py`
Outputs: `freq_mlp.safetensors`

This trains a **forensic frequency head** on 24-dim handcrafted features:

* FFT energy bands + slope
* Wavelet band energies
* SRM residual stats

The FreqMLP architecture:

* `FeatureNormalizer`
* `ContrastScaler`
* `BandGating` (4 bands × 6 dims)
* 2 × residual MLP blocks
* Linear head + `TemperatureScaler`

### Example

```bash
python train_freq_mlp_v5.py \
  --real-dir "/path/to/Real-img/Real-img" \
  --fake-dir "/path/to/Fake-img/Image" \
  --limit 8000 \
  --epochs 80 \
  --batch-size 8 \
  --lr 1e-3 \
  --freq-out "models/freq_mlp.safetensors"
```

Key flags:

* `--real-dir` / `--fake-dir` — data paths
* `--limit` — max images per class (0 = use all)
* `--freq-out` — where to save `freq_mlp.safetensors`

---

## 6. Step 3 — Train the Adaptive Fusion Head

Script: `train_fusion_head_only.py`
Outputs: `fusion_head.safetensors`

This script:

1. Loads **SigLIP** from `best_model.safetensors`
2. Loads **FreqMLP** from `freq_mlp.safetensors`
3. For each image:

   * SigLIP → `z_sig`
   * FreqMLP → `z_freq`
4. Trains `AdaptiveFusionHead`:

   ```python
   features = [z_freq, z_sig, |z_freq - z_sig|]
   → tiny MLP → [w_freq, w_sig]
   → softmax → fused logit
   ```

### Example

```bash
python train_fusion_head_only.py \
  --real-dir "/path/to/Real-img/Real-img" \
  --fake-dir "/path/to/Fake-img/Image" \
  --best-model "models/best_model.safetensors" \
  --freq-mlp  "models/freq_mlp.safetensors" \
  --fusion-out "models/fusion_head.safetensors" \
  --batch-size 32 \
  --epochs 5
```

This produces a trained `fusion_head.safetensors` that learns when to trust SigLIP vs freq.

---

## 7. Step 4 — Fit CORAL risk calibration (optional but recommended)

Script: `fit_coral_v5.py`
Outputs:

* `coral_cutpoints.json`
* `coral_temp.json`
* `coral_bins.npy`

CORAL maps the fused logit into **5 ordinal risk levels**:

* REAL
* LEAN_REAL
* BORDERLINE
* LEAN_FAKE
* FAKE

The script:

* Runs `best_model` + `freq_mlp` + `fusion_head` on a calibration set
* Collects fused logits
* Chooses 4 cutpoints to split them into 5 bins
* Writes them to JSON / NPY

### Example

```bash
python fit_coral_v5.py \
  --real-dir "/path/to/Real-img/Real-img" \
  --fake-dir "/path/to/Fake-img/Image" \
  --best-model "models/best_model.safetensors" \
  --freq-mlp  "models/freq_mlp.safetensors" \
  --fusion-head "models/fusion_head.safetensors" \
  --out-prefix "models/coral"
```

You’ll get:

* `models/coral_cutpoints.json`
* `models/coral_temp.json`
* `models/coral_bins.npy`

In `app.py`, you can load these instead of using hardcoded cutpoints.

---

## 8. How `app.py` uses the weights

In the v5 `app.py`, the high-level flow is:

1. **Load models**

   ```python
   # SigLIP visual expert
   best_path = hf_hub_download(MODEL_REPO, "best_model.safetensors", ...)
   siglip = BinaryClassifier(...); siglip.load_state_dict(...)

   # Frequency expert
   freq_path = hf_hub_download(MODEL_REPO, "freq_mlp.safetensors", ...)
   freq_mlp = FreqMLP(); freq_mlp.load_state_dict(...)

   # Fusion expert
   fusion_path = hf_hub_download(MODEL_REPO, "fusion_head.safetensors", ...)
   fusion_head = AdaptiveFusionHead(); fusion_head.load_state_dict(...)

   # CORAL (optional)
   with open("coral_cutpoints.json") as f:
       cuts = json.load(f)
   CORAL = CoralCalibrator(cuts=cuts)
   ```

2. **Per image:**

   * Run **multicrop + flip TTA** through SigLIP and FreqMLP
   * Obtain `z_sig`, `z_freq`
   * Fusion head → `z_fused`
   * CORAL → risk distribution, risk index, smooth fake prob
   * Patch-grid analysis, JPEG residual, embedding anomaly, sharpness, etc.
   * Combine into final `p_enhanced` and label: REAL / FAKE / UNCERTAIN / INCONCLUSIVE.

3. **Gradio UI** shows:

   * Text explanation (probs, risk band, certainty)
   * Heatmap overlay (suspicious regions)
   * Jitter preview (stability check)

---

## 9. Deploying to Hugging Face Spaces

1. Push this repo to GitHub.

2. Create a HF Space (Gradio, Python).

3. In `app.py`, set:

   ```python
   MODEL_REPO = "your-username/siglip"  # or whatever repo holds the weights
   ```

4. Upload to that HF model repo:

   ```text
   best_model.safetensors
   freq_mlp.safetensors
   fusion_head.safetensors
   coral_cutpoints.json
   coral_temp.json
   coral_bins.npy
   ```

5. The Space will download these via `hf_hub_download` and run `app.py`.

---

## 10. Quick pipeline summary

1. Train SigLIP deepfake head → `best_model.safetensors` (outside this repo).
2. `train_freq_mlp_v5.py` → `freq_mlp.safetensors`
3. `train_fusion_head_only.py` → `fusion_head.safetensors`
4. `fit_coral_v5.py` → `coral_cutpoints.json`, `coral_temp.json`, `coral_bins.npy`
5. Deploy `app.py` + all weights → full v5 detector.

---

```

You can adjust paths (e.g. `models/` vs root) and your actual HF repo name, but this should be a solid starting README for GitHub.
```
