"""
Shared test fixtures. A tiny model is trained on a handful of hand-written
headlines (a test fixture only, not used by the app), so the tests run
without the dataset and without internet access.
"""

import joblib
import pandas as pd
import pytest

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


@pytest.fixture(scope="session")
def pipeline():
    df, _ = prepare(FIXTURE)
    model = build_models()["Logistic Regression"]
    model.fit(df["clean"], df["label"])
    return model


@pytest.fixture
def model_dir(tmp_path, pipeline, monkeypatch):
    """A models/ folder with the fixture model, used via $TRUTHLENS_MODEL_DIR."""
    joblib.dump(pipeline, tmp_path / "news_pipeline.joblib")
    monkeypatch.setenv("TRUTHLENS_MODEL_DIR", str(tmp_path))
    return tmp_path
