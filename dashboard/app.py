from pathlib import Path

import streamlit as st
from utils import get_latest_paths, load_metadata

st.set_page_config(
    page_title="DriftRadar‑Vision",
    page_icon="🛰️",
    layout="wide",
)

meta = load_metadata()

st.sidebar.title("DriftRadar‑Vision")
st.sidebar.caption("Vision drift • GE • Evidently • MLflow")

# quick status chips
lp = get_latest_paths()
(
    st.sidebar.success("Artifacts detected")
    if lp["ev_summary"].exists()
    else st.sidebar.warning("Artifacts not found yet")
)

st.sidebar.write("**Locations**")
st.sidebar.code(str(Path.cwd()), language="bash")

st.title("🛰️ DriftRadar‑Vision Dashboard")
st.caption("Overview, data quality, drift trends, model registry, and what‑if explorer.")

st.info("Use the left sidebar to navigate pages. This landing page just introduces the app.")

st.markdown(
    """
    **How it works**

    *Artifacts are read from your repo's `artifacts/` folder.* On Spaces, push small JSON/HTML/PNG files — they will be rendered here.

    - **GE**: Data Docs and a compact summary are shown in *Data Quality*.
    - **Evidently**: Drift HTML/JSON + a compact `summary.json` feed *Drift Trends*.
    - **Registry**: Reads **MLflow** file store (`mlruns/`) if present to display model versions.
    - **What‑if**: Upload an image; run **Production** and **Staging** (ONNX/Torch), compare confidences, compute optional embedding z‑score.
    """
)

st.divider()

st.write(
    "**Tip:** If this is empty, run your pipelines to generate artifacts or copy sample files into `artifacts/`.\nUse the repo README for exact commands."
)
