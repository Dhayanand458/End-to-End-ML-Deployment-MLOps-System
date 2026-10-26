from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load as joblib_load

ART = Path("artifacts")
EV_DIR = ART / "reports" / "evidently"
GE_DIR = ART / "reports" / "ge"
EMB_DIR = ART / "reference" / "embeddings"
RUNS_DIR = ART / "runs"
MLRUNS = Path("mlruns")


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def get_latest_paths() -> dict[str, Path]:
    ev_dates = sorted([p for p in EV_DIR.glob("*/summary.json")])
    ge_sum = GE_DIR / "_latest" / "ge_summary.json"
    latest_ev = ev_dates[-1] if ev_dates else EV_DIR / "_latest" / "summary.json"
    pred_drift = Path(str(latest_ev).replace("summary.json", "prediction_drift.json"))
    return {
        "ev_summary": latest_ev,
        "ev_html": Path(str(latest_ev).replace("summary.json", "report.html")),
        "ev_json": Path(str(latest_ev).replace("summary.json", "report.json")),
        "pred_drift": pred_drift,
        "ge_summary": ge_sum,
    }


@st.cache_data(show_spinner=False)
def list_evidently_history(max_days: int = 60) -> list[Path]:
    items = sorted([p for p in EV_DIR.glob("*/summary.json")])
    return items[-max_days:]


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    path = Path("dashboard/metadata.json")
    return load_json(path)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_pca() -> tuple[np.ndarray, object] | None:
    z = EMB_DIR / "ref_pca.npy"
    p = EMB_DIR / "pca.joblib"
    if z.exists() and p.exists():
        return np.load(z), joblib_load(p)
    return None


@st.cache_data(show_spinner=False)
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
