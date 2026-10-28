# TruthLens — NLP-Powered News Analysis

TruthLens reads a news **headline** in two independent ways:

1. **Writing style (offline).** A TF-IDF + Logistic Regression model trained on the ISOT Fake
   News dataset estimates whether the wording resembles real or fake headlines:
   **Likely Real**, **Likely Fake** or **Uncertain**, with model-estimated probabilities and the
   words that pushed the prediction each way.
2. **Source check (online, optional).** The headline's key terms are searched in Wikipedia,
   Wikipedia's daily news archive, recent world news (GDELT) and, with a free API key, published
   fact-checks (Google Fact Check Tools). Result: **Corroborated**, **Disputed** or
   **Not enough evidence**, with every matching passage shown and linked.

> **It is not a fact checker.** The style model learns writing patterns from 2016–2017 headlines.
> The source check matches words, names and numbers; it does not understand meaning. A match
> means other sources say something similar, and no match does not prove a story false.

## Features
- **Analyze**: one headline, both signals side by side, a probability meter with an "Uncertain"
  zone, the matching source passages with matched words highlighted, example headlines, and a
  session history you can download as CSV or clear
- **Text insights**: word/character counts, average word length, all-caps words, share of words
  the model knows, word contributions (diverging chart) and the headline's top TF-IDF terms
- **Batch**: CSV upload (≤ 5 MB, first 2,000 rows), choose the headline column, output rows stay
  aligned with the input, skipped rows explain why, download results
- **Model performance**: held-out accuracy / precision / recall / F1, confusion matrix,
  Logistic Regression vs Naive Bayes, data-cleaning ledger, top words per class, misclassified examples
- **Method**: how both signals work, what is sent online, and the limits
- Light and dark themes (follows the browser setting)

## Technology
Python 3.10+, scikit-learn, pandas, NumPy, Streamlit, Altair, requests, joblib, pytest.
No LLM, no GPU, no paid service. The source check uses free public APIs only.

## Project structure
```
TruthLens/
├── app.py                  # Streamlit entry point: theme + page navigation
├── train_model.py          # load → clean → split → train → evaluate → save
├── core/                   # all analysis logic (no Streamlit code)
│   ├── preprocessing.py    # clean_title(), normalise(), validation, text statistics
│   ├── schema.py           # the result dict every mode returns
│   ├── style_model.py      # V1: prediction, word contributions, TF-IDF terms
│   ├── evidence.py         # V2: key terms, matching, stance, verdict
│   ├── sources.py          # V2: Wikipedia / news archive / GDELT / Fact Check API calls
│   └── analyze.py          # analyze(text, mode) — the one function the UI calls
├── ui/                     # theme (colours, fonts, CSS), HTML components, charts, session state
├── views/                  # one module per page: analyze, insights, batch, performance, method
├── assets/mark.svg         # logo
├── .streamlit/config.toml  # theme, 5 MB upload limit, no usage statistics
├── tests/                  # pytest (no dataset or internet needed)
├── data/                   # put Fake.csv and True.csv here (not included)
└── models/                 # created by train_model.py
```

## Setup
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
"Fake and Real News Dataset") and place `Fake.csv` and `True.csv` in `data/`. Only the `title`
column is used. Labels come from the file name: `Fake.csv` → fake, `True.csv` → real.
The dataset is not included in this repo.

**Other datasets.** Any CSV with a text column and a label column works. Map its labels explicitly:
```
python train_model.py --csv data/news.csv --text-col headline --label-col label \
    --fake-values FAKE,0 --real-values REAL,1
```
Rows whose label is in neither list are skipped and reported, never guessed. For ISOT files with a
different headline column name use `--title-col`.

## Train
```
python train_model.py
```
This prints the cleaning report and test metrics and writes `models/news_pipeline.joblib` and
`models/metrics.json`. If the app is already open, just reload the page: it picks up the new model.
Other locations: `--fake path/Fake.csv --real path/True.csv --out other/folder`
(the app reads other folders via the `TRUTHLENS_MODEL_DIR` environment variable).

## Run the app
```
streamlit run app.py
```

## Source check settings (optional)
- **Google Fact Check Tools.** Get a free API key (Google Cloud console → enable "Fact Check Tools
  API" → create an API key), then either `export GOOGLE_FACTCHECK_API_KEY=...` or create
  `.streamlit/secrets.toml` with `GOOGLE_FACTCHECK_API_KEY = "..."` (git-ignored).
  Without a key the other sources still work.
- **User-Agent.** Wikimedia asks API clients to identify themselves. Set
  `TRUTHLENS_USER_AGENT="TruthLens/2.0 (your-email-or-repo-url)"` if you deploy the app.
- Turn the **Cross-check online sources** switch off to keep everything on your computer.

## Run tests
```
pytest
```
47 tests cover cleaning, class mapping, probabilities, explanations, batch alignment, training
(missing/wrong columns, label mapping, end-to-end), the source check with fake sources (no
internet), HTML escaping, and the app itself (history, clearing, missing model).

## How it works
**Style model**
1. **Cleaning** (`core/preprocessing.py`): lowercase, remove bracket tags like `[VIDEO]`, source
   prefixes like `BREAKING:` / `Factbox:`, curly quotes, and normalise `U.S.` → `us`. These patterns
   identify *which website* a headline came from, not whether it is true, so leaving them in
   inflates accuracy. Negation words are kept (no stop-word removal).
2. **Leakage control**: duplicate headlines, and headlines that appear in both classes, are removed
   **before** a stratified 80/20 split (seed 42). TF-IDF is fit on training data only.
3. **Model**: `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` +
   `LogisticRegression` in one sklearn `Pipeline`. Multinomial Naive Bayes is trained on the same
   split for comparison. Precision/recall treat **fake** as the positive class.
4. **Explanation**: for each word present in the headline, `tfidf_value × coefficient`.
   Positive pushes toward Fake, negative toward Real. Unseen words get no score.
5. **Uncertain**: if P(fake) is between 0.40 and 0.60, or none of the words were seen in training.

**Source check** (`core/evidence.py`, `core/sources.py`)
1. **Key terms**: stop-words and reporting words ("says", "breaking") are dropped; numbers weigh 2,
   names/acronyms 1.5, other words 1; a light suffix rule matches "passes" with "passed".
2. **Search**: Wikipedia article intros; Wikipedia *Current events* daily pages (each item cites the
   outlet, e.g. bbc.com, which becomes the publisher); GDELT news from the last ~3 months; Google
   Fact Check Tools if a key is set. Sources are queried in parallel; one failing never breaks the others.
3. **Matching**: a passage is a *close match* if it contains ≥ 75% of the weighted key terms and
   ≥ 2/3 of the names and numbers. A negation ("not", "never", "denies") just before a matched word
   in only one of the two texts makes it *opposite wording*. Weaker overlaps are *related*.
4. **Verdict**: fact-checkers' ratings of a matching claim decide first. Otherwise **Corroborated**
   needs close matches from ≥ 2 independent publishers (or one Wikipedia passage covering ≥ 80%),
   **Disputed** needs ≥ 2 publishers with opposite wording, and anything else is
   **Not enough evidence**. Results are cached for an hour.

## Example workflow
1. `python train_model.py` → note the test metrics it prints.
2. `streamlit run app.py` → on **Analyze**, pick the example "UK votes to leave EU" and press
   **Analyze headline**.
3. Read the style card (P(fake), uncertain zone) and the source card (which sources answered, which
   publishers matched), then scroll through the highlighted passages.
4. Open **Text insights** to see which words moved the model, and **Model performance** for the
   held-out metrics.
5. On **Batch**, download the sample CSV, upload it, and download the results.

## Limitations and ethics
- **Style, not truth.** The strongest signals are writing style: Reuters headlines drop words like
  "the"/"is" and use "says"; the fake sites use more pronouns and emotional words. A sober fake or
  a sensational real headline will fool it.
- **Source bias.** All "real" training items come from Reuters, so the model partly learns
  "Reuters style". Reported metrics apply only to this dataset's held-out test split.
- **Old, narrow data.** Mostly US politics/world news from 2016–2017. New names are unknown words
  (the app warns when few words are known).
- **Word matching, not understanding.** The source check can miss paraphrases ("approves" vs
  "passes") and contradictions that use no negation word, and a passage can share words with a
  claim without saying the same thing. Always read the passages it shows.
- **Coverage gaps.** Very recent, local or niche stories may simply not be in these sources yet.
  "Not enough evidence" is not a debunk.
- **Privacy.** The style model runs locally. With the source check on, the headline's key terms
  are sent to Wikipedia, GDELT and (with a key) Google. Session history stays in the browser
  session only.

## Troubleshooting
- *"Not trained yet"* on the Analyze page → run `python train_model.py`, then reload the page.
- *"Dataset file not found"* → check `data/Fake.csv` and `data/True.csv` exist.
- *Recent news: busy / timed out* → GDELT allows one request every 5 seconds and is often busy;
  the other sources still answer. Try again shortly.
- *Wikipedia: busy* → Wikimedia rate limit; wait a few seconds. Setting `TRUTHLENS_USER_AGENT`
  with contact details helps.
- *Fact checks: not configured* → add `GOOGLE_FACTCHECK_API_KEY` (see above).
- Model file errors after upgrading scikit-learn → retrain.
- Only load `.joblib` files produced by this project; joblib can run code when loading.
