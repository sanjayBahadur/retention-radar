"""Small shared helpers for the Streamlit pages."""

import json

import pandas as pd
import streamlit as st

from retention_radar.config import get_project_root

ROOT = get_project_root()
OUTPUTS = ROOT / "data" / "outputs"
GREEN = "#234e3b"
AMBER = "#865715"
STATES = ["New", "Champion", "Regular", "At-Risk", "Dormant", "Lapsed"]
STATE_COLORS = dict(zip(STATES, ["#789880", "#234e3b", "#617e67", "#b18b45", "#93633d", "#a6ada2"]))


def page(title: str, description: str) -> None:
    """Give every page the same quiet layout and clear opening."""
    st.set_page_config(page_title=f"{title} | Retention Radar", layout="wide")
    st.markdown(
        """<style>
        .block-container {max-width:1120px;padding-top:2.8rem;}
        h1,h2,h3 {font-family:Georgia,serif;font-weight:400;letter-spacing:-.025em;}
        [data-testid="stMetricValue"] {font-family:Georgia,serif;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.caption("RETENTION RADAR · A RETAIL DATA CASE STUDY")
    st.title(title)
    st.write(description)
    st.sidebar.caption(
        "Historical analysis. Financial values are modeled scenarios, not realized campaign returns."
    )


def read_output(name: str):
    """Read a saved artifact, with an actionable message when it is missing."""
    path = OUTPUTS / name
    if not path.exists():
        st.error(f"The analysis output {name} is missing. Follow the rebuild steps in the README.")
        st.stop()
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_parquet(path)


def chart(figure) -> None:
    """Use one chart style across the app."""
    figure.update_layout(
        font=dict(family="Arial, sans-serif", color="#252d28"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=30, b=24),
        legend=dict(orientation="h", y=-0.18),
    )
    figure.update_xaxes(gridcolor="#e2e4da")
    figure.update_yaxes(gridcolor="#e2e4da")
    st.plotly_chart(figure, width="stretch")
