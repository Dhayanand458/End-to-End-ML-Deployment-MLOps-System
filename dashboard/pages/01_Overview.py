import streamlit as st
from components import stat_card
from utils import get_latest_paths, load_json

st.title("Overview")

paths = get_latest_paths()
ev = load_json(paths["ev_summary"]) if paths["ev_summary"].exists() else {}
pd = load_json(paths["pred_drift"]) if paths["pred_drift"].exists() else {}
ge = load_json(paths["ge_summary"]) if paths["ge_summary"].exists() else {}

c1, c2, c3, c4 = st.columns(4)
stat_card(
    "Production drift",
    "✅ No" if not ev.get("is_drifted") else "⚠️ Yes",
    "From Evidently summary",
    key="drift",
)
stat_card(
    "Embedding drift (top‑5 PSI)",
    f"{ev.get('embedding_top5_psi_mean', 0.0):.3f}",
    f"threshold {ev.get('psi_threshold', 0.2)}",
    key="psi",
)
stat_card(
    "Conf. drop (probe)",
    f"{pd.get('confidence_drop_pct', 0.0):.1f}%",
    "mean confidence drop",
    key="conf",
)
stat_card(
    "GE blockers",
    "0" if ge.get("blockers") in (0, None) else str(ge.get("blockers")),
    "should be 0 for promotion",
    key="ge",
)

st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Latest Evidently report")
    if paths["ev_html"].exists():
        st.link_button("Open report.html", str(paths["ev_html"]))
    st.caption(str(paths["ev_html"]))
with col2:
    st.subheader("GE summary")
    st.json(ge, expanded=False)
