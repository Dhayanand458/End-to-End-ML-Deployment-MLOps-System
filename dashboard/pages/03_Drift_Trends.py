import pandas as pd
import plotly.express as px
import streamlit as st
from utils import get_latest_paths, list_evidently_history, load_json

st.title("Drift Trends (Evidently)")

items = list_evidently_history(90)
rows = []
for p in items:
    d = p.parent.name
    s = load_json(p)
    pdj = p.with_name("prediction_drift.json")
    pred = load_json(pdj) if pdj.exists() else {}
    rows.append(
        {
            "date": d,
            "is_drifted": bool(s.get("is_drifted", False)),
            "emb_psi": float(s.get("embedding_top5_psi_mean", 0.0)),
            "psi_thresh": float(s.get("psi_threshold", 0.2)),
            "conf_drop": float(pred.get("confidence_drop_pct", 0.0)),
        }
    )

if rows:
    df = pd.DataFrame(rows).sort_values("date")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Embedding drift (PSI)")
        fig = px.line(df, x="date", y="emb_psi", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Confidence drop (%)")
        fig2 = px.line(df, x="date", y="conf_drop", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No Evidently summaries yet. Run drift_check.py.")

st.divider()
paths = get_latest_paths()
st.write("Latest summary.json:")
if paths["ev_summary"].exists():
    st.code(paths["ev_summary"].read_text()[:4000])
