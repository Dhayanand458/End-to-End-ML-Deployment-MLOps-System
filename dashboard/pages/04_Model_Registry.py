import pandas as pd
import streamlit as st
from mlflow.tracking import MlflowClient
from utils import MLRUNS

st.title("Model Registry (MLflow)")

if not MLRUNS.exists():
    st.info(
        "No local MLflow store found (mlruns/). Push mlruns/ or mirror metadata.json for Space."
    )
else:
    client = MlflowClient(tracking_uri="file:./mlruns")
    names = sorted(set([m.name for m in client.search_registered_models()]))
    if not names:
        st.info("No registered models yet.")
    else:
        name = st.selectbox("Model name", names, index=0)
        vers = client.search_model_versions(f"name='{name}'")
        rows = []
        for v in vers:
            run = client.get_run(v.run_id)
            rows.append(
                {
                    "version": int(v.version),
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "best_val_acc1": run.data.metrics.get("best_val_acc1"),
                    "final_val_ece": run.data.metrics.get("final_val_ece"),
                }
            )
        df = pd.DataFrame(rows).sort_values("version")
        st.dataframe(df, use_container_width=True, hide_index=True)
