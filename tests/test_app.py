"""
UI tests: the Streamlit app is run headlessly with streamlit.testing.
The online evidence check is switched off, so no network is used.
"""

from pathlib import Path

import joblib
from streamlit.testing.v1 import AppTest

from core.schema import FAKE, REAL, UNCERTAIN
from core.sources import NEWS
from ui import components, state

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run_headline(at: AppTest, text: str) -> AppTest:
    at.text_area(key="headline").set_value(text)
    at.toggle(key="online").set_value(False)
    return next(b for b in at.button if b.label == "Analyze").click().run()


def test_analyze_history_and_clear(model_dir):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at = _run_headline(at, "Senate says trade talks with China will continue")
    assert not at.exception
    assert len(at.session_state.history) == 1
    assert at.session_state.history[0]["Evidence check"] == "Off"
    next(b for b in at.button if b.label == "Clear history").click().run()
    assert at.session_state.history == []


def test_empty_input_warning_goes_away_on_next_analyze(model_dir):
    # Regression: the warning was drawn before the next click cleared it, so it
    # stayed on screen next to the new report.
    at = AppTest.from_file(APP, default_timeout=60).run()
    at = _run_headline(at, "   ")
    assert at.session_state.input_error and at.session_state.history == []
    at = _run_headline(at, "Senate says trade talks with China will continue")
    assert not at.exception
    assert at.session_state.input_error is None and len(at.session_state.history) == 1
    assert not any("Please enter" in h.proto.body for h in at.get("html"))


def test_app_works_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TRUTHLENS_MODEL_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at = _run_headline(at, "Senate says trade talks with China will continue")
    assert not at.exception
    assert at.session_state.last["style"] is None  # card shows "Not trained yet"


def test_model_is_picked_up_after_training(tmp_path, monkeypatch, pipeline):
    # Regression: a missing model used to be cached, so training while the
    # app was open still showed "No trained model found" until a restart.
    monkeypatch.setenv("TRUTHLENS_MODEL_DIR", str(tmp_path))
    assert state.get_pipeline() is None
    joblib.dump(pipeline, tmp_path / "news_pipeline.joblib")
    assert state.get_pipeline() is not None


def test_html_from_sources_is_escaped():
    evil = {"source": NEWS, "publisher": "<b>x</b>", "title": "<script>alert(1)</script>",
            "sentence": "Senate <img src=x onerror=alert(1)> passes", "url": "javascript:alert(1)",
            "date": None, "rating": None, "stance": "supports", "score": 0.9, "matched": ["Senate"]}
    html = components.clippings([evil])
    assert "<script>" not in html and "<img" not in html and "<b>x</b>" not in html
    assert "javascript:" not in html                     # unsafe link not rendered
    assert "<mark>Senate</mark>" in html


def test_rule_based_uncertain_shows_no_probabilities():
    # Regression: "Hitler was a dictator" (too short) showed "Uncertain" over a 99% fake needle.
    r = {"verdict": UNCERTAIN, "probabilities": {FAKE: 0.99, REAL: 0.01}, "stats": {"known_word_ratio": 1.0}}
    html = components.style_card(r, model_ready=True)
    assert "P(fake)" not in html and "tl-meter" not in html
    r["probabilities"] = {FAKE: 0.52, REAL: 0.48}  # a real 50/50 keeps the meter
    assert "tl-meter" in components.style_card(r, model_ready=True)
