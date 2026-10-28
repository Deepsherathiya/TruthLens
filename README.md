# TruthLens — NLP-Powered News Analysis

TruthLens has two ways of looking at a claim or headline:

| | **V2 — Evidence Check** | **V1 — Style Analysis** |
|---|---|---|
| Question it answers | *What does the evidence say?* | *Does this read like fake news?* |
| How | Google Fact Check API + Wikipedia → sentence embeddings → NLI model | TF-IDF + Logistic Regression trained on ISOT headlines |
| Output | Supported / Refuted / Not enough evidence, with sources | Likely Real / Likely Fake / Uncertain, with word weights |

> **Neither is a guarantee.** V1 judges writing style only. V2 is evidence-based but automated:
> it can retrieve the wrong article or misread a sentence. Always open the sources it shows.

## V2 — Evidence Check
1. **Fact Check API**: search existing fact-checks (PolitiFact, Snopes, AFP, Alt News, BOOM, ...).
   If the full claim finds nothing, retry with keywords (the API needs *all* words to match).
   A fact-check is only used if it's about the *same* claim: similar meaning (MiniLM embeddings ≥ 0.60)
   **and** the logically valid direction holds (NLI entailment ≥ 0.50):
   - rated False → our claim must *imply* the fact-checked claim
   - rated True → the fact-checked claim must imply ours

   This drops look-alikes such as *"Video shows Chandrayaan-3 landing"* (False: the video was fake),
   which says nothing about whether Chandrayaan-3 landed.
2. **Wikipedia** (public API, no key): top 3 articles for the claim, split into sentences.
3. **Ranking**: `all-MiniLM-L6-v2` keeps the 15 sentences most similar to the claim.
4. **NLI**: `cross-encoder/nli-deberta-v3-small` labels each as entails / contradicts / neutral.
5. **Verdict**: matching fact-checks decide first. Otherwise the **most relevant** sentence
   (similarity ≥ 0.55) that the NLI model is confident about (≥ 0.80) decides. Using relevance order
   avoids NLI's habit of calling unrelated sentences "contradictions"
   (e.g. *"Chandrayaan-3 was launched on 14 July"* vs *"... landed"*).
   If an almost equally relevant sentence disagrees, the app shows a warning.

Simple yes/no questions are rewritten as statements first ("Is the Earth flat?" → "the Earth is flat").

**Results** (`python check_demo_claims.py`): 10/10 on the demo claims, which were used to tune the
thresholds, so treat that as a best case. On 10 *unseen* claims it got 8/10. Both misses were retrieval
errors: a replica-tower sentence beat the real Eiffel Tower sentence, and "Boiling water reactor" was
retrieved for a claim about boiling water.

**Setup:** needs internet, plus a free Google Fact Check Tools API key. Copy `.env.example` to `.env`
and paste the key after `GOOGLE_FACTCHECK_API_KEY=`. `.env` is git-ignored. Without a key, V2 still runs
on Wikipedia alone. The two models (~270 MB) download from Hugging Face on first use.

## V1 — Style Analysis
Classifies a news **headline** as **Likely Real**, **Likely Fake**, or **Uncertain**
using TF-IDF features and a Logistic Regression model trained on the ISOT Fake News dataset.
It shows model-estimated probabilities and which words pushed the prediction each way.
The model learns writing patterns from 2016–2017 news headlines, so a prediction is a statistical
guess about style, not proof that a story is true or false.

## Features
- Evidence check with fact-check and Wikipedia evidence cards, plus the V1 result for comparison
- Single-headline analysis with probabilities, an "Uncertain" zone, and word-level explanations
- Example headlines for quick demos, session history
- Batch CSV analysis with downloadable results (rows stay aligned with the input)
- Model performance page: accuracy/precision/recall/F1, confusion matrix, LR vs Naive Bayes,
  top words per class, and misclassified examples

## Project structure
```
TruthLens/
├── app.py                 # Streamlit demo UI
├── train_model.py         # V1: load → clean → split → train → evaluate → save
├── check_demo_claims.py   # V2: run the demo claims live and compare with expected verdicts
├── core/
│   ├── preprocessing.py   # clean_title(), question → statement, input validation
│   ├── schema.py          # the result dict every mode returns
│   ├── analyze.py         # analyze(text, mode) — the one function the UI calls
│   ├── style_model.py     # V1 prediction + word contributions
│   ├── retrieval.py       # V2 Fact Check API + Wikipedia
│   ├── nlp_models.py      # V2 MiniLM embeddings + DeBERTa NLI
│   └── evidence.py        # V2 matching rules + verdict
├── tests/                 # offline tests (no internet or models needed)
├── .env.example           # copy to .env and add your Fact Check API key
├── data/                  # put Fake.csv and True.csv here (not included)
└── models/                # created by train_model.py
```

## Setup
Python 3.10+.

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
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
### V2
- **Retrieval is the weak link.** If Wikipedia search returns the wrong article, the NLI step reasons
  correctly about the wrong text.
- **Recent or local events** usually get "Not enough evidence": Wikipedia lags, and there's no news search yet.
- **Numbers and dates**: small NLI models are weak at arithmetic and exact figures.
- **Thresholds were tuned on 10 claims**, which is a small set. Real accuracy is lower than on the demo claims.
- English only. Each check takes about 10–20 s on CPU (the first one ~30–45 s while models load).

### V1
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
