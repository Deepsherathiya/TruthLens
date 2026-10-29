# TruthLens — NLP-Powered News Analysis

TruthLens looks at a news **headline or claim** in two independent ways:

| | **Evidence check (V2)** | **Writing style (V1)** |
|---|---|---|
| Question it answers | *What does the evidence say?* | *Does this read like fake news?* |
| How | Fact-checks + Wikipedia + news archives → sentence embeddings → NLI model | TF-IDF + Logistic Regression trained on ISOT headlines |
| Output | Supported / Refuted / Not enough evidence, with sources | Likely Real / Likely Fake / Uncertain, with word weights |

> **Neither is a guarantee.** V1 judges writing style only. V2 is evidence-based but automated:
> it can retrieve the wrong article or misread a sentence. Always open the sources it shows.

## Features
- **Analyze**: both signals side by side; a probability meter with an "Uncertain" zone; the evidence
  passages (deciding one first, claim words highlighted, links to the sources); example headlines and
  demo claims; a session history you can download as CSV or clear
- **Text insights**: word/character counts, average word length, all-caps words, share of words the
  model knows, word contributions (diverging chart) and the headline's top TF-IDF terms
- **Batch**: CSV upload (≤ 5 MB, first 2,000 rows), choose the headline column, output rows stay
  aligned with the input, skipped rows explain why, download results
- **Model performance**: held-out accuracy / precision / recall / F1, confusion matrix,
  Logistic Regression vs Naive Bayes, data-cleaning ledger, top words per class, misclassified examples
- **Method**: how both signals work, what is sent online, and the limits
- Light and dark themes (follows the browser setting)

## Project structure
```
TruthLens/
├── app.py                  # Streamlit entry point: theme + page navigation
├── train_model.py          # V1: load → clean → split → train → evaluate → save
├── check_demo_claims.py    # V2: run the demo claims live and compare with expected verdicts
├── core/                   # all analysis logic (no Streamlit code)
│   ├── preprocessing.py    # clean_title(), question → statement, validation, text statistics
│   ├── schema.py           # the result dict every mode returns
│   ├── analyze.py          # analyze(text, mode) — the one function the UI calls
│   ├── style_model.py      # V1: prediction, word contributions, TF-IDF terms
│   ├── evidence.py         # V2: picks the engine; word-matching fallback engine
│   ├── nli_evidence.py     # V2: NLI engine — fact-check matching, ranking, verdict rule
│   ├── nlp_models.py       # V2: MiniLM embeddings + DeBERTa NLI
│   ├── retrieval.py        # V2: Fact Check API + full Wikipedia articles
│   └── sources.py          # V2: news archive, GDELT, (word engine) Wikipedia intros + fact-checks
├── ui/                     # theme (colours, fonts, CSS), HTML components, charts, session state
├── views/                  # one module per page: analyze, insights, batch, performance, method
├── assets/mark.svg         # logo
├── .streamlit/config.toml  # theme, 5 MB upload limit, no usage statistics
├── .env.example            # copy to .env and add your Fact Check API key
├── tests/                  # pytest (no dataset, internet or model downloads needed)
├── data/                   # put Fake.csv, True.csv (and IFND.csv) here (not included)
└── models/                 # created by train_model.py
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
The two NLI models (~270 MB) download from Hugging Face on first use. If PyTorch can't be installed,
remove the last three lines of `requirements.txt`: the app still runs and judges evidence by word
matching instead (see below).

**Fact Check API key (recommended):** copy `.env.example` to `.env` and paste a free Google Fact
Check Tools API key after `GOOGLE_FACTCHECK_API_KEY=` (Google Cloud console → enable "Fact Check
Tools API" → create an API key). `.env` is git-ignored. Without a key the other sources still work.

## Dataset (V1)
Download the **ISOT Fake News Dataset** (University of Victoria; also mirrored on Kaggle as
"Fake and Real News Dataset") and place `Fake.csv` and `True.csv` in `data/`. Only the `title`
column is used. Labels come from the file name: `Fake.csv` → fake, `True.csv` → real.

**Indian context (optional, recommended).** Also download the **Indian Fake News Dataset (IFND)**
(Kaggle: "IFND dataset", 56,714 Indian headlines from 2013–21) and save it as `data/IFND.csv`.
`python train_model.py` then trains on both (`--no-ifnd` to skip it). The `Statement` column is used,
with `Label` TRUE → real and Fake → fake. Two IFND problems are fixed while loading, because either
would let the model score well for the wrong reason:
- **Broken punctuation.** The file mixes three encodings, and the junk differs by class (`Ûª` only
  in real rows, `â€™` only in fake rows). Junk between two letters becomes an apostrophe, any other
  junk a space.
- **Fact-check wording.** About 81% of the "fake" rows are fact-checkers' headlines
  (*"Fact Check: No, this photo is not…"*), against 5% of real rows. Words such as "fact check",
  "false", "viral", "hoax", "claim" and a leading "No," are removed from every IFND row.

The datasets are not included in this repo.

**Other datasets.** Any CSV with a text column and a label column works. Map its labels explicitly:
```
python train_model.py --csv data/news.csv --text-col headline --label-col label \
    --fake-values FAKE,0 --real-values REAL,1
```
Rows whose label is in neither list are skipped and reported, never guessed. For ISOT files with a
different headline column name use `--title-col`.

## Train, run, test
```
python train_model.py          # writes models/news_pipeline.joblib and models/metrics.json
streamlit run app.py
pytest                         # offline: no dataset, internet or model downloads
python check_demo_claims.py    # live check of the evidence engine on the 10 demo claims
```
If the app is already open when you train, just reload the page: it picks up the new model.

## V2 — Evidence check
1. **Fact Check API**: search existing fact-checks (PolitiFact, Snopes, AFP, Alt News, BOOM, ...).
   If the full claim finds nothing, retry with keywords (the API needs *all* words to match).
   A fact-check is only used if it's about the *same* claim: similar meaning (MiniLM embeddings ≥ 0.60)
   **and** the logically valid direction holds (NLI entailment ≥ 0.50):
   - rated False → our claim must *imply* the fact-checked claim
   - rated True → the fact-checked claim must imply ours

   This drops look-alikes such as *"Video shows Chandrayaan-3 landing"* (False: the video was fake),
   which says nothing about whether Chandrayaan-3 landed.
2. **Candidate sentences** (no keys needed):
   - **Wikipedia**: top 3 articles for the claim, split into sentences. For a person, one more
     sentence comes from the article's categories: *"Narendra Modi is alive."* ("Living people") or
     *"Ratan Tata died in 2024."* ("2024 deaths"). No article says "Modi did not commit suicide", so
     without it death hoaxes could only get *Not enough evidence*. A death claim with only a surname
     (*"modi died"*) is checked about the best-known person with that name (the biography with the
     most incoming links whose title ends in it: Narendra Modi), and the app says so. Otherwise
     Wikipedia's search found Modi Group businessmen who did die and the claim came out *Supported*.
   - **News archive**: Wikipedia's daily *Current events* pages, back to the early 2000s. Each item
     cites the outlet that reported it (Reuters, BBC, …), which is shown as the publisher.
   - **Recent news**: GDELT, worldwide news from about the last 3 months (often busy; optional).
3. **Ranking**: `all-MiniLM-L6-v2` keeps the 15 sentences most similar to the claim.
4. **NLI**: `cross-encoder/nli-deberta-v3-small` labels each as entails / contradicts / neutral.
5. **Verdict**: matching fact-checks decide first. Otherwise the **most relevant** sentence
   (similarity ≥ 0.55) that the NLI model is confident about (≥ 0.80) decides. Using relevance order
   avoids NLI's habit of calling unrelated sentences "contradictions"
   (e.g. *"Chandrayaan-3 was launched on 14 July"* vs *"... landed"*). The deciding sentence must
   also be **on topic**: it must contain ≥ 50% of the claim's key words and ≥ 50% of its names and
   numbers. Without this, *"Minister Arvind Sawant resigned from the cabinet"* "supported"
   *"Narendra Modi resigned"*. A **refuting** sentence must also share a key word besides the names,
   because contradiction is the NLI model's weak spot: *"Modi is often accused of having narcissistic
   traits"* (97% contradiction) was refuting *"modi is a dictator"*. Supporting sentences don't need
   this, so paraphrases still count (*"two wind-related fatalities"* supports *"killed two people"*).
   If an almost equally relevant sentence disagrees, the app shows a warning.

Simple yes/no questions are rewritten as statements first ("Is the Earth flat?" → "the Earth is flat").

**Word-matching fallback.** If PyTorch isn't installed or the models fail to load (or you set
`TRUTHLENS_ENGINE=words`), each passage is judged by how many of the claim's key words it contains
(numbers weigh 2, names 1.5, other words 1). A *close match* needs ≥ 75% of them and ≥ 2/3 of the
names and numbers; a negation ("not", "never") just before a matched word in only one text makes it
*opposite wording*. **Supported** then needs close matches from ≥ 2 independent publishers (or one
Wikipedia passage covering ≥ 80%), and **Refuted** needs ≥ 2 publishers with opposite wording.

**Results** (`python check_demo_claims.py`): 10/10 on the demo claims with a Fact Check API key; the
demo claims were used to tune the thresholds, so treat that as a best case. On 10 *unseen* claims it
got 8/10. Both misses were retrieval errors: a replica-tower sentence beat the real Eiffel Tower
sentence, and "Boiling water reactor" was retrieved for a claim about boiling water.
Without an API key it gets 7/10 on the demo claims: the three hoaxes (2000-rupee GPS chip, UNESCO
anthem, 5G coronavirus) are only refuted by published fact-checks, so they come back as
*Not enough evidence*.

## V1 — Style model
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
5. **Uncertain**: if P(fake) is between 0.40 and 0.60, none of the words were seen in training, or
   the text has fewer than 4 words (a two-word claim like "modi died" has no writing style to judge).

## Example workflow
1. `python train_model.py` → note the test metrics it prints.
2. `streamlit run app.py` → on **Analyze**, pick a demo claim (e.g. the Great Wall of China one) or the
   headline "UK votes to leave EU", and press **Analyze**.
3. Compare the style card (P(fake), uncertain zone) with the evidence card (verdict, which sources
   answered), then read the deciding passage and open its source.
4. Open **Text insights** to see which words moved the style model, and **Model performance** for the
   held-out metrics.
5. On **Batch**, download the sample CSV, upload it, and download the results.

## Limitations and ethics
### V2
- **Retrieval is the weak link.** If the wrong article is found, the NLI step reasons correctly about
  the wrong text.
- **Recent or local events** may get "Not enough evidence": Wikipedia lags, and the free news sources
  don't cover everything. "Not enough evidence" is not a debunk.
- **Numbers and dates**: small NLI models are weak at arithmetic and exact figures.
- **Thresholds were tuned on 10 claims**, which is a small set. Real accuracy is lower than on the demo claims.
- English only. Each check takes about 10–20 s on CPU (the first one ~30–45 s while models load).
- **Word-matching fallback** can miss paraphrases ("approves" vs "passes") and contradictions that
  use no negation word.

### V1
- **Style, not truth.** The strongest signals are writing style: Reuters headlines drop words like
  "the"/"is" and use "says"; the fake sites use more pronouns and emotional words. A sober fake or a
  sensational real headline will fool it.
- **Source bias.** All ISOT "real" items come from Reuters, and most IFND real items from a few
  Indian papers (The Tribune, The Statesman, ThePrint), so the model partly learns their house style.
  Reported metrics apply only to these datasets' held-out test splits.
- **Fact-check style.** Even after cleaning, IFND's fake side is mostly fact-checkers' wording about a
  claim, so words like "photo", "video", "old" and "did" lean fake.
- **Old data.** US news from 2016–17 and Indian news from 2013–21. New names are unknown words
  (the app warns when few words are known).

With IFND (87,239 headlines after cleaning), Logistic Regression scores 93.3% accuracy / 91.2% F1
overall: 93.1% / 89.5% on Indian test headlines and 93.5% / 92.8% on US ones. Naive Bayes scores
92.6% / 90.5%. Each model's main setting is chosen on a validation slice of the training data
(Logistic Regression C = 8, Naive Bayes alpha = 0.1); the test set is used once. Accuracy drops on
very short text (54.5% on the 11 test headlines of 1–3 words, 88.4% at 4–5 words), so text with
fewer than 4 words gets *Uncertain*.

### Privacy
The style model and the NLI models run locally. With **Check evidence online** on, the claim (or its
key words) is sent to Google Fact Check, Wikipedia and GDELT. Session history stays in the browser
session only.

## Troubleshooting
- *"Not trained yet"* on the Analyze page → run `python train_model.py`, then reload the page.
- *"Dataset file not found"* → check `data/Fake.csv` and `data/True.csv` exist.
- *Judged by: word matching* → PyTorch or sentence-transformers isn't installed (see Setup).
- *Recent news: busy / timed out* → GDELT allows one request every 5 seconds and is often busy;
  the other sources still answer.
- *Fact checks: not configured* → add the key to `.env` (see Setup).
- Model file errors after upgrading scikit-learn → retrain.
- Only load `.joblib` files produced by this project; joblib can run code when loading.
