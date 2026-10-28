# TruthLens — NLP-Powered News Analysis (V1)

TruthLens classifies a news **headline** as **Likely Real**, **Likely Fake**, or **Uncertain**
using TF-IDF features and a Logistic Regression model trained on the ISOT Fake News dataset.
It shows model-estimated probabilities and which words pushed the prediction each way.

> **It is not a fact checker.** The model learns writing patterns from 2016–2017 news headlines.
> A prediction is a statistical guess about style, not proof that a story is true or false.

## Features
- Single-headline analysis with probabilities, an "Uncertain" zone, and word-level explanations
- Example headlines for quick demos, session history
- Batch CSV analysis with downloadable results (rows stay aligned with the input)
- Model performance page: accuracy/precision/recall/F1, confusion matrix, LR vs Naive Bayes,
  top words per class, and misclassified examples

## Project structure
```
TruthLens/
├── app.py                 # Streamlit demo UI
├── train_model.py         # load → clean → split → train → evaluate → save
├── core/
│   ├── preprocessing.py   # clean_title(), input validation (shared by training & prediction)
│   ├── schema.py          # the result dict every mode returns (V1 + planned V2)
│   ├── style_model.py     # V1 prediction + word contributions
│   └── analyze.py         # analyze(text, mode) — the one function the UI calls
├── tests/test_core.py
├── data/                  # put Fake.csv and True.csv here (not included)
└── models/                # created by train_model.py
```

## Setup
Python 3.10+.

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset
Download the **ISOT Fake News Dataset** (University of Victoria; also mirrored on Kaggle as
"Fake and Real News Dataset"). Place `Fake.csv` and `True.csv` in `data/`.
Only the `title` column is used. Labels come from the file name: `Fake.csv` → fake, `True.csv` → real.
The dataset is not included in this repo.

## Train
```
python train_model.py
```
This prints the cleaning report and test metrics and writes `models/news_pipeline.joblib` and
`models/metrics.json`. Other paths: `python train_model.py --fake path/Fake.csv --real path/True.csv`.

## Run the app
```
streamlit run app.py
```

## Run tests
```
python -m pytest
```
Tests use a tiny hand-written fixture, so they run without the dataset.

## How it works
1. **Cleaning** (`core/preprocessing.py`): lowercase, remove bracket tags like `[VIDEO]`,
   source prefixes like `BREAKING:` / `Factbox:`, curly quotes, and normalise `U.S.` → `us`.
   These patterns identify *which website* a headline came from, not whether it is true, so
   leaving them in inflates accuracy. Negation words are kept (no stop-word removal).
2. **Leakage control**: duplicate headlines are removed, and headlines that appear in both classes
   are dropped, **before** a stratified 80/20 split (seed 42). TF-IDF is fit on training data only.
3. **Model**: `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` +
   `LogisticRegression`, wrapped in one sklearn `Pipeline`. Multinomial Naive Bayes is trained on the
   same split for comparison.
4. **Explanation**: for each word present in the headline, `tfidf_value × coefficient`.
   Positive pushes toward Fake, negative toward Real. Unseen words get no score.
5. **Uncertain**: if P(Fake) is between 0.40 and 0.60, or none of the words were seen in training.

## Limitations
- **Style, not truth.** The remaining strongest signals are writing style: Reuters headlines drop
  words like "the"/"is" and use "says"; the fake sites use more pronouns and emotional words.
  A real headline written casually may be called fake, and vice versa.
- **Old, narrow data.** Mostly US politics/world news from 2016–2017. New names and topics are
  unseen words, so predictions on current news are weak (the app warns when few words are known).
- **Source bias.** All "real" items come from Reuters, so the model partly learns "Reuters style".
- Reported metrics apply only to this dataset's held-out test split.

## Troubleshooting
- *"No trained model found"* → run `python train_model.py` first.
- *"Dataset file not found"* → check `data/Fake.csv` and `data/True.csv` exist.
- Model file errors after upgrading scikit-learn → retrain.
- Only load `.joblib` files produced by this project; joblib can run code when loading.
