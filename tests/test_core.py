"""
Tests for cleaning, the style model, batch alignment and training.

Run:  pytest
"""

import json

import pandas as pd
import pytest

from core.analyze import analyze
from core.preprocessing import MAX_INPUT_CHARS, clean_title, normalise, text_stats, validate_input
from core.schema import FAKE, REAL, UNCERTAIN
from core.style_model import ModelNotTrainedError, load_pipeline, model_path, predict_batch
from train_model import (
    load_isot, load_labeled_csv, main, misclassified_examples, prepare,
)


# ---------------------------------------------------------------- cleaning

def test_cleaning_removes_source_artifacts():
    assert clean_title(" BREAKING: WATCH: Trump SLAMS U.S. Judge [VIDEO]") == "trump slams us judge"
    assert clean_title("Factbox: Merkel says “not” now (Dec 28)") == 'merkel says "not" now'
    assert clean_title("WATCH - Senator speaks") == "senator speaks"


def test_cleaning_leaves_hyphenated_words_alone():
    # "Watch-list" is a word, not a "WATCH:" source tag.
    assert clean_title("Watch-list grows for FBI") == "watch-list grows for fbi"
    assert clean_title("Report-card shows gains") == "report-card shows gains"


def test_cleaning_keeps_negations():
    assert "not" in clean_title("Government will NOT approve the plan").split()


def test_normalise_keeps_case():
    assert normalise("BREAKING: Paris talks with the U.S.") == "Paris talks with the US"


@pytest.mark.parametrize("bad", ["", "   ", None, 12345, "x" * (MAX_INPUT_CHARS + 1), "[VIDEO]"])
def test_invalid_input_returns_error(pipeline, bad):
    r = analyze(bad, pipeline=pipeline)
    assert r["error"] and r["verdict"] is None


def test_validate_input_accepts_normal_text():
    assert validate_input("Senate passes the budget, 51-49!") is None


def test_text_stats():
    s = text_stats("  WATCH: Hillary Just DESTROYED the UK GOP!  ")
    assert s["words"] == 7 and s["characters"] == 41  # surrounding spaces not counted
    assert s["all_caps_words"] == 3          # WATCH, DESTROYED, GOP; not the 2-letter "UK"
    assert s["exclamations"] == 1


# ---------------------------------------------------------------- style model

def test_valid_prediction_and_class_mapping(pipeline):
    real = analyze("Senate says trade talks with China will continue", pipeline=pipeline)
    fake = analyze("This Moron Just Got DESTROYED, You Won't Believe It", pipeline=pipeline)
    assert real["verdict"] == REAL and fake["verdict"] == FAKE
    # Probabilities follow the model's classes_ order and sum to 1
    for r in (real, fake):
        assert set(r["probabilities"]) == {REAL, FAKE}
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-6
    assert fake["probabilities"][FAKE] > fake["probabilities"][REAL]


def test_result_has_insights(pipeline):
    r = analyze("Senate says trade talks will continue", pipeline=pipeline)
    assert r["stats"]["words"] == 6 and 0 < r["stats"]["known_word_ratio"] <= 1
    assert r["tfidf_terms"] and all(t["weight"] > 0 for t in r["tfidf_terms"])


def test_explanation_sign_points_to_fake(pipeline):
    r = analyze("This moron just did it", pipeline=pipeline)
    top = r["word_contributions"][0]
    assert top["score"] > 0  # positive = pushes toward Fake


def test_unknown_words_are_uncertain(pipeline):
    r = analyze("zebra quantum flibbertigibbet", pipeline=pipeline)
    assert r["verdict"] == UNCERTAIN and r["word_contributions"] == [] and r["warnings"]


def test_batch_keeps_row_alignment(pipeline):
    texts = ["Senate says talks continue", "", "This moron just lied", "   "]
    progress = []
    out = predict_batch(pipeline, texts, on_progress=lambda done, total: progress.append((done, total)))
    assert len(out) == len(texts)
    assert [bool(r["error"]) for r in out] == [False, True, False, True]
    assert [r["input"] for r in out] == texts
    assert progress[-1] == (4, 4)


def test_missing_model_file(tmp_path):
    with pytest.raises(ModelNotTrainedError):
        load_pipeline(tmp_path / "missing.joblib")


def test_model_dir_can_be_overridden(model_dir):
    assert model_path() == model_dir / "news_pipeline.joblib"
    assert load_pipeline() is not None


# ---------------------------------------------------------------- training

def test_prepare_drops_duplicates_and_conflicts():
    df = pd.DataFrame({
        "title": ["Same title", "same TITLE", "Clash", "clash", "Unique"],
        "label": ["real", "real", "real", "fake", "fake"],
    })
    out, report = prepare(df)
    assert sorted(out["clean"]) == ["same title", "unique"]
    assert report["dropped_conflicting"] == 2 and report["dropped_duplicates"] == 1


def test_missing_title_column_exits(tmp_path):
    for name in ("Fake.csv", "True.csv"):
        pd.DataFrame({"headline": ["a b c"]}).to_csv(tmp_path / name, index=False)
    with pytest.raises(SystemExit, match="no 'title' column"):
        load_isot(tmp_path / "Fake.csv", tmp_path / "True.csv")


def test_missing_dataset_file_exits(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        load_isot(tmp_path / "Fake.csv", tmp_path / "True.csv")


def test_labeled_csv_maps_labels_explicitly(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame({"text": ["a b", "c d", "e f", "g h"], "y": [0, 1, 1, 7]}).to_csv(path, index=False)
    df = load_labeled_csv(path, "text", "y", fake_values=["0"], real_values=["1"])
    assert list(df["label"]) == ["fake", "real", "real"]  # label 7 is skipped, not guessed
    with pytest.raises(SystemExit, match="missing column"):
        load_labeled_csv(path, "headline", "y", ["0"], ["1"])


def test_no_misclassified_examples_does_not_crash(pipeline):
    df, _ = prepare(pd.DataFrame({"title": ["Senate says talks continue", "This moron just lied"],
                                  "label": ["real", "fake"]}))
    pred = pipeline.predict(df["clean"])
    assert misclassified_examples(pipeline, pred, df["clean"], df["label"], df["title"]) == []


def test_training_end_to_end(tmp_path):
    from tests.conftest import FIXTURE
    fake, real = tmp_path / "Fake.csv", tmp_path / "True.csv"
    FIXTURE[FIXTURE.label == "fake"][["title"]].to_csv(fake, index=False)
    FIXTURE[FIXTURE.label == "real"][["title"]].to_csv(real, index=False)
    out = tmp_path / "models"
    main(["--fake", str(fake), "--real", str(real), "--out", str(out)])
    metrics = json.loads((out / "metrics.json").read_text())
    assert (out / "news_pipeline.joblib").exists()
    assert metrics["split"]["random_state"] == 42 and metrics["positive_class"] == "fake"
    assert set(metrics["models"]) == {"Logistic Regression", "Multinomial Naive Bayes"}
