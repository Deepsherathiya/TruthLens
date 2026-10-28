"""
Train the TruthLens V1 headline classifier.

Usage (ISOT dataset, labels come from the file names):
    python train_model.py
    python train_model.py --fake data/Fake.csv --real data/True.csv --title-col title

Usage (any single CSV with a text column and a label column):
    python train_model.py --csv data/news.csv --text-col headline --label-col label \
        --fake-values fake,0 --real-values real,1

Outputs (in models/, or --out):
    news_pipeline.joblib   fitted TF-IDF + Logistic Regression pipeline
    metrics.json           held-out test metrics, model comparison, etc.
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
from core.style_model import model_dir

ROOT = Path(__file__).resolve().parent
SEED = 42
TEST_SIZE = 0.2
POSITIVE = "fake"            # precision/recall are reported for the "fake" class
LABEL_ORDER = ["real", "fake"]  # row/column order of the confusion matrix
MAX_MISCLASSIFIED = 25


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV or exit with a plain-language message."""
    if not path.exists():
        sys.exit(
            f"Dataset file not found: {path}\n"
            "Download the ISOT Fake News dataset (Fake.csv and True.csv) "
            "and place both files in the data/ folder, or pass --csv for your own file."
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        sys.exit(f"{path} is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        sys.exit(f"Could not read {path} as CSV: {e}")


def load_isot(fake_path: Path, real_path: Path, title_col: str = "title") -> pd.DataFrame:
    """Load both CSVs, label them explicitly, and keep only the title."""
    frames = []
    for path, label in [(fake_path, "fake"), (real_path, "real")]:
        df = read_csv(path)
        if title_col not in df.columns:
            sys.exit(f"{path} has no '{title_col}' column. Columns found: {list(df.columns)}")
        frames.append(pd.DataFrame({"title": df[title_col], "label": label}))
    return pd.concat(frames, ignore_index=True)


def _label_text(value) -> str:
    # 1.0 -> "1" so numeric labels read from a column with blanks still match.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip().lower()


def load_labeled_csv(path: Path, text_col: str, label_col: str,
                     fake_values: list[str], real_values: list[str]) -> pd.DataFrame:
    """Load one CSV and map its labels to "fake"/"real" with an explicit mapping."""
    df = read_csv(path)
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing column(s) {missing}. Columns found: {list(df.columns)}")

    mapping = {v.strip().lower(): "fake" for v in fake_values}
    mapping.update({v.strip().lower(): "real" for v in real_values})
    raw = df[label_col].map(_label_text)
    label = raw.map(mapping)
    unmapped = label.isna()
    if unmapped.any():
        examples = sorted(raw[unmapped].unique())[:10]
        print(f"  Skipping {int(unmapped.sum())} row(s) whose label is not in the mapping: {examples}")
    if label.nunique() < 2:
        sys.exit("Need rows of both classes. Check --fake-values / --real-values "
                 f"against the labels in '{label_col}': {sorted(raw.unique())[:10]}")
    return pd.DataFrame({"title": df.loc[~unmapped, text_col], "label": label[~unmapped]})


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
    report["class_counts"] = {k: int(v) for k, v in df["label"].value_counts().items()}
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


def evaluate(model: Pipeline, X_test, y_test) -> tuple[dict, np.ndarray]:
    pred = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, pos_label=POSITIVE, average="binary", zero_division=0
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


def misclassified_examples(model: Pipeline, pred, X_test, y_test, titles) -> list[dict]:
    """A few wrong predictions, for error analysis in the dashboard."""
    wrong = np.where(pred != y_test.to_numpy())[0][:MAX_MISCLASSIFIED]
    if len(wrong) == 0:  # a perfect test score leaves nothing to show
        return []
    proba = model.predict_proba(X_test.iloc[wrong])
    fake_col = list(model.classes_).index("fake")
    return [
        {
            "title": str(titles.iloc[i]).strip(),
            "true": str(y_test.iloc[i]),
            "predicted": str(pred[i]),
            "fake_probability": round(float(proba[j, fake_col]), 3),
        }
        for j, i in enumerate(wrong)
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    isot = parser.add_argument_group("ISOT dataset (default)")
    isot.add_argument("--fake", type=Path, default=ROOT / "data" / "Fake.csv")
    isot.add_argument("--real", type=Path, default=ROOT / "data" / "True.csv")
    isot.add_argument("--title-col", default="title", help="headline column in both files")
    single = parser.add_argument_group("single labelled CSV (instead of ISOT)")
    single.add_argument("--csv", type=Path, help="one CSV with a text and a label column")
    single.add_argument("--text-col", default="title")
    single.add_argument("--label-col", default="label")
    single.add_argument("--fake-values", default="fake,0",
                        help="comma-separated label values meaning fake (default: fake,0)")
    single.add_argument("--real-values", default="real,1",
                        help="comma-separated label values meaning real (default: real,1)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: models/)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out or model_dir()

    print("Loading dataset...")
    if args.csv:
        fake_values = args.fake_values.split(",")
        real_values = args.real_values.split(",")
        raw = load_labeled_csv(args.csv, args.text_col, args.label_col, fake_values, real_values)
        dataset = {
            "name": f"{args.csv.name} (column '{args.text_col}')",
            "files": [str(args.csv)],
            "label_mapping": {"fake": fake_values, "real": real_values},
        }
    else:
        raw = load_isot(args.fake, args.real, args.title_col)
        dataset = {
            "name": "ISOT Fake News (titles only)",
            "files": [str(args.fake), str(args.real)],
            "label_mapping": {"fake": [args.fake.name], "real": [args.real.name]},
        }
    df, data_report = prepare(raw)
    for k, v in data_report.items():
        print(f"  {k}: {v}")
    if df["label"].value_counts().min() < 2:
        sys.exit("Each class needs at least 2 headlines after cleaning to split train/test.")

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
    misclassified = misclassified_examples(best, preds["Logistic Regression"], X_test, y_test, title_test)

    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out / "news_pipeline.joblib")

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": {**dataset, **data_report},
        "split": {
            "test_size": TEST_SIZE, "stratified": True, "random_state": SEED,
            "train_rows": len(X_train), "test_rows": len(X_test),
            "train_class_counts": {k: int(v) for k, v in y_train.value_counts().items()},
            "test_class_counts": {k: int(v) for k, v in y_test.value_counts().items()},
        },
        "positive_class": POSITIVE,
        "label_order": LABEL_ORDER,
        "selected_model": "Logistic Regression",
        "models": results,
        "top_terms": top_terms(best),
        "misclassified_examples": misclassified,
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved {out / 'news_pipeline.joblib'} and {out / 'metrics.json'}")
    print("Top 'fake' terms:", [t for t, _ in metrics["top_terms"]["fake"][:12]])
    print("Top 'real' terms:", [t for t, _ in metrics["top_terms"]["real"][:12]])


if __name__ == "__main__":
    main()
