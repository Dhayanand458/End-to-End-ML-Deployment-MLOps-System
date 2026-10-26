from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from utils import CLASS_NAMES, load_pca, softmax

st.title("What‑if explorer")

st.write(
    "Upload an image to compare **Production** and **Staging** models (if available). Uses ONNX if present, falls back to Torch."
)

img_file = st.file_uploader("Upload an RGB image", type=["png", "jpg", "jpeg"])


@st.cache_resource
def _load_onnx(stage: str):
    # Expect ONNX files mirrored under artifacts/runs/onnx/<stage>.onnx
    p = Path(f"artifacts/runs/onnx/{stage}.onnx")
    if p.exists():
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        return sess
    # Fallback: generic model.onnx
    p2 = Path("artifacts/runs/onnx/model.onnx")
    if p2.exists():
        return ort.InferenceSession(str(p2), providers=["CPUExecutionProvider"])
    return None


@st.cache_resource
def _torch_model():
    try:
        from driftradar_vision.models import build_backbone

        m = build_backbone("resnet18").eval()
        ckpt = Path("artifacts/runs/best.pt")
        if ckpt.exists():
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        return m
    except Exception:
        return None


TFM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

col1, col2 = st.columns([1, 1])

if img_file:
    img = Image.open(img_file).convert("RGB")
    col1.image(img, caption="Input", use_container_width=True)

    x = TFM(img).unsqueeze(0).numpy()  # NCHW float32

    prod = _load_onnx("production") or _load_onnx("staging")
    logits_prod = None
    if prod:
        out = prod.run(None, {"input": x})[0]
        logits_prod = out
    elif _torch_model() is not None:
        with torch.no_grad():
            logits_prod = _torch_model()(torch.from_numpy(x)).numpy()

    staging = _load_onnx("staging")
    logits_stg = None
    if staging:
        logits_stg = staging.run(None, {"input": x})[0]

    if logits_prod is None:
        st.warning("No model available. Export ONNX or run training first.")
    else:
        probs_p = softmax(logits_prod)
        topk_p = probs_p[0].argsort()[-5:][::-1]
        st.markdown("### Production predictions")
        st.bar_chart({CLASS_NAMES[i]: float(probs_p[0, i]) for i in topk_p})

        if logits_stg is not None:
            probs_s = softmax(logits_stg)
            topk_s = probs_s[0].argsort()[-5:][::-1]
            st.markdown("### Staging predictions")
            st.bar_chart({CLASS_NAMES[i]: float(probs_s[0, i]) for i in topk_s})

        # Optional: embedding z‑score vs reference
        emb = load_pca()
        if emb is not None:
            ref_z, pca = emb
            try:
                import torchvision.models as tvm

                m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
                backbone = torch.nn.Sequential(*list(m.children())[:-1]).eval()
                with torch.no_grad():
                    t = TFM(img).unsqueeze(0)
                    z = backbone(t).view(1, -1).numpy()
                    zproj = pca.transform(z)
                    mu = ref_z.mean(axis=0)
                    sd = ref_z.std(axis=0) + 1e-8
                    zscore = float(np.mean(np.abs((zproj - mu) / sd)))
                    st.info(f"Embedding z‑score vs reference: {zscore:.2f}")
            except Exception as e:
                st.caption(f"Embedding not available: {e}")
else:
    st.info("Upload an image to begin.")
