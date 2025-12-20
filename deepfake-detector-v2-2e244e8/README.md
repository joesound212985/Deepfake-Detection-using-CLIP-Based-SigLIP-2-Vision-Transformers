---
title: Deepfake Detector v2

colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "4.36.1"
app_file: app.py
pinned: false
---

# Deepfake Detector v2

This Space runs a deepfake detector built with:

- SigLIP visual encoder
- Frequency-domain features
- Forensic and diffusion-specific detectors
- Gradio UI (`app.py`)

Upload an image to see:

- Predicted label (REAL / FAKE)
- Risk band (GREEN / YELLOW / RED)
- Forensic scores and other debug metrics

Check out the configuration reference at:  
https://huggingface.co/docs/hub/spaces-config-reference
