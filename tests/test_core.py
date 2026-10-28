"""
Basic tests. They train a tiny model on a handful of hand-written headlines
(a test fixture only, not used by the app) so they run without the dataset.

Run:  python -m pytest
"""

import pandas as pd
import pytest

from core.analyze import analyze
from core.preprocessing import MAX_INPUT_CHARS, clean_title
from core.schema import FAKE, REAL, UNCERTAIN
from core.style_model import ModelNotTrainedError, load_pipeline, predict_batch
from train_model import build_models, prepare

FIXTURE = pd.DataFrame({
    "title": [
        "Senate passes budget bill, says leader", "EU and China hold trade talks",
        "Court rules on tax case, minister says", "PM says talks with EU will continue",
        "Senate panel says vote set for Tuesday", "China says trade deal close",
        "WATCH: This Liberal Just Got DESTROYED", "You Won't Believe What Hillary Just Did",
        "This GOP Idiot Just Said The Dumbest Thing", "BREAKING: He Is Lying And This Proves It",
        "Just Watch This Moron Melt Down [VIDEO]", "This Is The Most Insane Thing He Ever Did",
    ],
    "label": ["real"] * 6 + ["fake"] * 6,
})


@pytest.fixture(scope="module")
def pipeline():
    df, _ = prepare(FIXTURE)
    model = build_models()["Logistic Regression"]
    model.fit(df["clean"], df["label"])
    return model


def test_cleaning_removes_source_artifacts():
    assert clean_title(" BREAKING: WATCH: Trump SLAMS U.S. Judge [VIDEO]") == "trump slams us judge"
    assert clean_title("Factbox: Merkel says “not” now (Dec 28)") == 'merkel says "not" now'


def test_cleaning_keeps_negations():
    assert "not" in clean_title("Government will NOT approve the plan").split()


@pytest.mark.parametrize("bad", ["", "   ", None, "x" * (MAX_INPUT_CHARS + 1), "[VIDEO]"])
def test_invalid_input_returns_error(pipeline, bad):
    r = analyze(bad, pipeline=pipeline)
    assert r["error"] and r["verdict"] is None


def test_valid_prediction_and_class_mapping(pipeline):
    real = analyze("Senate says trade talks with China will continue", pipeline=pipeline)
    fake = analyze("This Moron Just Got DESTROYED, You Won't Believe It", pipeline=pipeline)
    assert real["verdict"] == REAL and fake["verdict"] == FAKE
    # Probabilities follow the model's classes_ order and sum to 1
    for r in (real, fake):
        assert set(r["probabilities"]) == {REAL, FAKE}
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert fake["probabilities"][FAKE] > fake["probabilities"][REAL]


def test_explanation_sign_points_to_fake(pipeline):
    r = analyze("This moron just did it", pipeline=pipeline)
    top = r["word_contributions"][0]
    assert top["score"] > 0  # positive = pushes toward Fake


def test_unknown_words_are_uncertain(pipeline):
    r = analyze("zebra quantum flibbertigibbet", pipeline=pipeline)
    assert r["verdict"] == UNCERTAIN and r["word_contributions"] == [] and r["warnings"]


def test_batch_keeps_row_alignment(pipeline):
    texts = ["Senate says talks continue", "", "This moron just lied", "   "]
    out = predict_batch(pipeline, texts)
    assert len(out) == len(texts)
    assert [bool(r["error"]) for r in out] == [False, True, False, True]
    assert [r["input"] for r in out] == texts


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotTrainedError):
        load_pipeline(tmp_path / "missing.joblib")


def test_prepare_drops_duplicates_and_conflicts():
    df = pd.DataFrame({
        "title": ["Same title", "same TITLE", "Clash", "clash", "Unique"],
        "label": ["real", "real", "real", "fake", "fake"],
    })
    out, report = prepare(df)
    assert sorted(out["clean"]) == ["same title", "unique"]
    assert report["dropped_conflicting"] == 2 and report["dropped_duplicates"] == 1
