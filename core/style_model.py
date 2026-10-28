"""
V1 "style" analysis: TF-IDF + Logistic Regression prediction with
word-level explanations.

Only load model files produced by this project's own train_model.py:
joblib files can execute code when loaded, so never load untrusted ones.
"""

import json
from pathlib import Path

import joblib
import numpy as np

from core.preprocessing import clean_title
from core.schema import MODEL_LABEL_TO_VERDICT, UNCERTAIN

MODEL_PATH = Path("models/news_pipeline.joblib")
METRICS_PATH = Path("models/metrics.json")

# If the fake probability falls in this band, we say "Uncertain"
# instead of forcing a Real/Fake call.
UNCERTAIN_BAND = (0.40, 0.60)
TOP_N_WORDS = 8


class ModelNotTrainedError(FileNotFoundError):
    pass


def load_pipeline(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise ModelNotTrainedError(
            f"Model file not found at {path}. Run `python train_model.py` first."
        )
    return joblib.load(path)


def load_metrics(path: Path = METRICS_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def explain(pipeline, cleaned: str, top_n: int = TOP_N_WORDS) -> list[dict]:
    """
    Contribution of each term actually present in the headline:
        score = tfidf_value * coefficient
    Positive score pushes toward "fake", negative toward "real".
    Words the model never saw in training get no score (they had no influence).
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    vec = tfidf.transform([cleaned])
    if vec.nnz == 0:
        return []
    # coef_ points toward classes_[1]; flip it if that class isn't "fake".
    sign = 1.0 if clf.classes_[1] == "fake" else -1.0
    coef = clf.coef_[0] * sign
    vocab = tfidf.get_feature_names_out()
    scores = vec.data * coef[vec.indices]
    terms = [
        {"term": str(vocab[i]), "score": round(float(s), 4)}
        for i, s in zip(vec.indices, scores)
    ]
    terms.sort(key=lambda t: abs(t["score"]), reverse=True)
    return terms[:top_n]


def known_word_ratio(pipeline, cleaned: str) -> float:
    """Share of single words in the headline that exist in the model vocabulary."""
    tfidf = pipeline.named_steps["tfidf"]
    words = [w for w in tfidf.build_analyzer()(cleaned) if " " not in w]
    if not words:
        return 0.0
    return sum(w in tfidf.vocabulary_ for w in words) / len(words)


def predict_style(pipeline, text: str, result: dict) -> dict:
    """Fill a schema result dict (see core/schema.py) with the V1 prediction."""
    cleaned = clean_title(text)
    result["cleaned"] = cleaned

    proba = pipeline.predict_proba([cleaned])[0]
    classes = list(pipeline.classes_)  # read the order from the model, never assume it
    probs = {MODEL_LABEL_TO_VERDICT[c]: round(float(p), 4) for c, p in zip(classes, proba)}
    result["probabilities"] = probs

    fake_p = float(proba[classes.index("fake")])
    predicted = MODEL_LABEL_TO_VERDICT[classes[int(np.argmax(proba))]]
    result["confidence"] = round(float(max(proba)), 4)

    ratio = known_word_ratio(pipeline, cleaned)
    if ratio == 0:
        result["verdict"] = UNCERTAIN
        result["warnings"].append(
            "None of these words appear in the training data, so the model has nothing to go on."
        )
    elif UNCERTAIN_BAND[0] <= fake_p <= UNCERTAIN_BAND[1]:
        result["verdict"] = UNCERTAIN
        result["warnings"].append("The model's probabilities are close to 50/50.")
    else:
        result["verdict"] = predicted
        if ratio < 0.5:
            result["warnings"].append(
                f"Only {ratio:.0%} of the words were seen during training; treat this result with extra caution."
            )

    result["word_contributions"] = explain(pipeline, cleaned)
    return result


def predict_batch(pipeline, texts: list[str]) -> list[dict]:
    """Predict many headlines; output list is aligned 1:1 with the input."""
    from core.analyze import analyze  # local import avoids a circular import
    return [analyze(t, mode="style", pipeline=pipeline) for t in texts]
