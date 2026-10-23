"""Streamlit app entry point — two pages, one job each.

Usage:
    streamlit run dashboard.py

  - Research & Analysis (views/research.py): what the bot looked at,
    calculated and decided this cycle — the reasoning.
  - Portfolio Bot (views/portfolio_bot.py): what it's holding and how
    that's going — the result.

Uses st.navigation rather than a pages/ folder purely so the sidebar shows
real page names instead of filenames.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Trading Bot", layout="wide")

# Tighten Streamlit's default spacing so each page fits one screen instead
# of needing a scroll for what's really a handful of numbers and a chart.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 100%; }
      [data-testid="stMetric"] { padding: 0.4rem 0.6rem; }
      [data-testid="stMetricLabel"] { font-size: 0.8rem; }
      [data-testid="stMetricValue"] { font-size: 1.3rem; }
      div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.25rem; }
      h1 { font-size: 1.6rem; margin-bottom: 0.1rem; }
      h3 { margin-top: 0.3rem; margin-bottom: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.navigation([
    st.Page("views/portfolio_bot.py", title="Portfolio Bot", icon=":material/smart_toy:", default=True),
    st.Page("views/research.py", title="Research & Analysis", icon=":material/science:"),
    st.Page("views/scenario_lab.py", title="Scenario Lab", icon=":material/experiment:"),
]).run()
