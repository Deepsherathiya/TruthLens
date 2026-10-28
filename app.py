"""
TruthLens: NLP-powered news headline analysis.

Run:  streamlit run app.py

Pages live in views/, shared UI pieces in ui/, and all analysis logic in core/
(the UI only calls core.analyze.analyze() and reads the result dict).
"""

from pathlib import Path

import streamlit as st

from ui import state, theme
from views import analyze, batch, insights, method, performance

MARK = Path(__file__).parent / "assets" / "mark.svg"

st.set_page_config(page_title="TruthLens", page_icon=str(MARK), layout="wide",
                   initial_sidebar_state="collapsed")
theme.apply()
state.init()
st.logo(str(MARK), size="medium")

state.PAGES.update({
    "analyze": st.Page(analyze.render, title="Analyze", url_path="analyze", default=True),
    "insights": st.Page(insights.render, title="Text insights", url_path="insights"),
    "batch": st.Page(batch.render, title="Batch", url_path="batch"),
    "performance": st.Page(performance.render, title="Model performance", url_path="performance"),
    "method": st.Page(method.render, title="Method", url_path="method"),
})
st.navigation(list(state.PAGES.values()), position="top").run()
