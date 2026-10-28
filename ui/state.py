"""
Model loading, cached evidence checks and per-session state for the UI.

Session state keys:
  headline  text in the headline box          online   source-check toggle
  last      most recent analysis (dict)        history  list of past analyses
  batch     latest batch result                example  example picker value
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core import style_model
from core.analyze import analyze
from core.schema import FAKE
from core.sources import NEWS

# Training datasets, by the "source" name train_model.py gives them.
DATASETS = {
    "ISOT": ("US news (ISOT)", "2016–17"),
    "IFND": ("Indian news (IFND)", "2013–21"),
}

# st.Page objects by name, filled in by app.py so pages can link to each other.
PAGES: dict = {}

EXAMPLES = {
    "Senate passes tax bill": "Senate passes tax bill, Trump says he will sign it",
    "Brexit talks": "EU leaders agree to start next phase of Brexit talks",
    "UK votes to leave EU": "UK votes to leave the European Union",
    "“DESTROYED on live TV”": "WATCH: Hillary Just DESTROYED This GOP Moron On Live TV",
    "“Caught lying again”": "BREAKING: Obama Caught Lying AGAIN, And It's Worse Than You Think",
    "RBI holds repo rate": "RBI keeps repo rate unchanged at 6.5 per cent",
    "“WhatsApp calls monitored”": "Government will monitor all WhatsApp calls and messages from tomorrow",
}


def init() -> None:
    st.session_state.setdefault("headline", "")
    st.session_state.setdefault("online", True)
    st.session_state.setdefault("last", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("batch", None)
    # Streamlit forgets a widget's value when its page isn't shown. Re-assigning
    # keeps the headline text and toggle as the user moves between pages.
    for key in ("headline", "online"):
        st.session_state[key] = st.session_state[key]


@st.cache_resource(show_spinner=False)
def _load_pipeline(path: str, modified: float):
    # `modified` is part of the cache key, so retraining loads the new model.
    return style_model.load_pipeline(Path(path))


def get_pipeline():
    """The trained pipeline, or None if it hasn't been trained yet."""
    path = style_model.model_path()
    if not path.exists():
        return None  # not cached: the app picks up the model as soon as it exists
    return _load_pipeline(str(path), path.stat().st_mtime)


def get_metrics() -> dict | None:
    try:
        return style_model.load_metrics()
    except (OSError, ValueError):
        return None


def training_data_summary() -> str:
    """e.g. "87,239 real and fake headlines (US news, 2016–17; Indian news, 2013–21)"."""
    m = get_metrics()
    if m is None:
        return "real and fake headlines"
    d = m["dataset"]
    sources = list(d.get("source_counts") or (["ISOT"] if "ISOT" in d["name"] else []))
    eras = "; ".join(f"{DATASETS[s][0].split(' (')[0]}, {DATASETS[s][1]}" for s in sources if s in DATASETS)
    return f"{d['final_rows']:,} real and fake headlines" + (f" ({eras})" if eras else "")


def factcheck_key() -> str | None:
    """Google Fact Check API key from the environment, .env, or .streamlit/secrets.toml."""
    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GOOGLE_FACTCHECK_API_KEY")
    except Exception:  # no secrets file
        return None


class _Uncacheable(Exception):
    def __init__(self, result):
        self.result = result


@st.cache_data(ttl=3600, max_entries=200, show_spinner=False)
def _cached_sources(text: str, api_key: str | None) -> dict:
    result = analyze(text, mode="evidence", api_key=api_key)
    # Don't keep a failed or partial search for an hour: raising skips the cache.
    # GDELT is busy so often that its failure alone doesn't count.
    failed = [c for c in result["sources_checked"] if not c["ok"] and not c["skipped"] and c["name"] != NEWS]
    if result["error"] or failed:
        raise _Uncacheable(result)
    return result


def run_sources(text: str) -> dict:
    """Evidence check for a headline; repeated checks within an hour are served from cache."""
    try:
        return _cached_sources(text.strip(), factcheck_key())
    except _Uncacheable as e:
        return e.result


def run_style(text: str, pipeline) -> dict | None:
    return analyze(text, mode="style", pipeline=pipeline) if pipeline is not None else None


def remember(text: str, style: dict | None, sources: dict | None) -> None:
    now = datetime.now()
    st.session_state.last = {"text": text, "time": now.strftime("%H:%M:%S"), "style": style, "sources": sources}
    st.session_state.history.insert(0, {
        "Time": now.strftime("%H:%M:%S"),
        "Headline": text.strip(),
        "Writing style": (style or {}).get("verdict") or "–",
        "P(fake)": (style or {}).get("probabilities", {}).get(FAKE),
        "Evidence check": (sources or {}).get("verdict") or ("Off" if sources is None else "Unavailable"),
    })


def history_frame() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.history)
