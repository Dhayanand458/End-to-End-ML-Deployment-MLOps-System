from __future__ import annotations

import pandas as pd
import streamlit as st


def stat_card(label: str, value: str, help: str | None = None, key: str | None = None):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"<h3 style='margin-top:0'>{value}</h3>", unsafe_allow_html=True)
        if help:
            st.caption(help)


def kv_table(title: str, kv: dict[str, str]):
    st.subheader(title)
    df = pd.DataFrame({"key": list(kv.keys()), "value": list(kv.values())})
    st.dataframe(df, hide_index=True, use_container_width=True)
