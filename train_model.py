"""
Train the TruthLens V1 headline classifier on the ISOT dataset.

Usage:
    python train_model.py
    python train_model.py --fake data/Fake.csv --real data/True.csv

Outputs:
    models/news_pipeline.joblib   fitted TF-IDF + Logistic Regression pipeline
    models/metrics.json           held-out test metrics, model comparison, etc.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from core.preprocessing import clean_title

SEED = 42
TEST_SIZE = 0.2
POSITIVE = "fake"            # precision/recall are reported for the "fake" class
LABEL_ORDER = ["real", "fake"]  # row/column order of the confusion matrix


def load_isot(fake_path: Path, real_path: Path) -> pd.DataFrame:
    """Load both CSVs, label them explicitly, and keep only the title."""
    frames = []
    for path, label in [(fake_path, "fake"), (real_path, "real")]:
        if not path.exists():
            sys.exit(
                f"Dataset file not found: {path}\n"
                "Download the ISOT Fake News dataset (Fake.csv and True.csv) "
                "and place both files in the data/ folder."
            )
        df = pd.read_csv(path)
        if "title" not in df.columns:
            sys.exit(f"{path} has no 'title' column. Columns found: {list(df.columns)}")
        frames.append(pd.DataFrame({"title": df["title"], "label": label}))
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean titles and remove rows that would cause leakage."""
    report = {"raw_rows": len(df)}
    df = df.copy()
    df["clean"] = df["title"].map(clean_title)

    empty = df["clean"] == ""
    report["dropped_empty"] = int(empty.sum())
    df = df[~empty]

    # The same cleaned headline in both classes is contradictory: drop all copies.
    labels_per_title = df.groupby("clean")["label"].nunique()
    conflicting = labels_per_title[labels_per_title > 1].index
    report["dropped_conflicting"] = int(df["clean"].isin(conflicting).sum())
    df = df[~df["clean"].isin(conflicting)]

    # Duplicates could land in both train and test and inflate the score.
    before = len(df)
    df = df.drop_duplicates(subset="clean")
    report["dropped_duplicates"] = before - len(df)

    report["final_rows"] = len(df)
    report["class_counts"] = df["label"].value_counts().to_dict()
    return df.reset_index(drop=True), report


def make_vectorizer() -> TfidfVectorizer:
    # Text is already lowercased by clean_title(). No stop-word list, so
    # negations ("not", "never", "no") stay in.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", MultinomialNB()),
        ]),
    }


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, pos_label=POSITIVE, average="binary"
    )
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=LABEL_ORDER).tolist(),
    }, pred


def top_terms(model: Pipeline, n: int = 20) -> dict:
    """Most influential terms per class from the Logistic Regression weights."""
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    clf = model.named_steps["clf"]
    coef = clf.coef_[0]
    # coef_ > 0 pushes toward clf.classes_[1]; check the order, don't assume it.
    toward_second = clf.classes_[1]
    order = np.argsort(coef)
    neg = [(vocab[i], round(float(coef[i]), 3)) for i in order[:n]]
    pos = [(vocab[i], round(float(coef[i]), 3)) for i in order[::-1][:n]]
    first = clf.classes_[0]
    return {toward_second: pos, first: neg}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--fake", type=Path, default=Path("data/Fake.csv"))
    parser.add_argument("--real", type=Path, default=Path("data/True.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    print("Loading dataset...")
    df, data_report = prepare(load_isot(args.fake, args.real))
    for k, v in data_report.items():
        print(f"  {k}: {v}")

    # Split BEFORE fitting TF-IDF; the vectorizer is fit inside the pipeline
    # on training data only.
    X_train, X_test, y_train, y_test, title_train, title_test = train_test_split(
        df["clean"], df["label"], df["title"],
        test_size=TEST_SIZE, stratify=df["label"], random_state=SEED,
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    results, fitted, preds = {}, {}, {}
    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results[name], preds[name] = evaluate(model, X_test, y_test)
        fitted[name] = model
        m = results[name]
        print(f"  accuracy={m['accuracy']} precision={m['precision']} "
              f"recall={m['recall']} f1={m['f1']}")

    best = fitted["Logistic Regression"]

    # A few wrong predictions, for error analysis in the dashboard.
    wrong = np.where(preds["Logistic Regression"] != y_test.to_numpy())[0]
    proba = best.predict_proba(X_test.iloc[wrong])
    fake_col = list(best.classes_).index("fake")
    misclassified = [
        {
            "title": str(title_test.iloc[i]).strip(),
            "true": y_test.iloc[i],
            "predicted": preds["Logistic Regression"][i],
            "fake_probability": round(float(proba[j, fake_col]), 3),
        }
        for j, i in enumerate(wrong[:25])
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, args.out / "news_pipeline.joblib")

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "name": "ISOT Fake News (titles only)",
            "files": [str(args.fake), str(args.real)],
            **data_report,
        },
        "split": {
            "test_size": TEST_SIZE, "stratified": True, "random_state": SEED,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "test_class_counts": y_test.value_counts().to_dict(),
        },
        "positive_class": POSITIVE,
        "label_order": LABEL_ORDER,
        "selected_model": "Logistic Regression",
        "models": results,
        "top_terms": top_terms(best),
        "misclassified_examples": misclassified,
    }
    with open(args.out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {args.out / 'news_pipeline.joblib'} and {args.out / 'metrics.json'}")
    print("Top 'fake' terms:", [t for t, _ in metrics["top_terms"]["fake"][:12]])
    print("Top 'real' terms:", [t for t, _ in metrics["top_terms"]["real"][:12]])


if __name__ == "__main__":
    main()
