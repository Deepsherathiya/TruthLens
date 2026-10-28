# TruthLens: An NLP-Powered Fake News Detection and Headline Credibility Analysis System

## 1. Project Overview

Build **TruthLens**, a lightweight, polished, beginner-friendly NLP lab project that classifies a news headline as **Likely Real** or **Likely Fake** based on patterns learned from a labeled dataset.

The project should be practical and demo-ready without requiring a paid API, GPU, LLM, or complex deep-learning model. It must be easy for a B.Tech student to explain in an NLP lab viva.

### Important limitation and wording

TruthLens is a **machine-learning text classifier, not a factual verification engine**. Its output must be presented as a model prediction, not as proof that a headline is true or false. A headline can be misleading even if it sounds credible, and sensational language does not automatically mean a story is false.

Use wording such as:
- “Likely Real”
- “Likely Fake”
- “Model-estimated probability”
- “Prediction based on learned patterns”

Avoid wording such as:
- “Verified true”
- “Confirmed fake”
- “100% real”
- “Fact checked” unless an actual evidence-verification feature is implemented.

Do not fabricate model performance, dataset statistics, predictions, or probability values.

---

## 2. Goals

1. Accept a user-provided news headline.
2. Preprocess the text consistently with the training pipeline.
3. Predict the likely class using a trained NLP/ML model.
4. Display class probabilities when supported by the selected model.
5. Provide basic linguistic/text insights.
6. Explain predictions using learned feature contributions where feasible.
7. Store prediction history for the current session.
8. Support batch classification from a CSV file.
9. Evaluate the model using a held-out test set.
10. Present results in a clean, modern, responsive dashboard.
11. Include clear setup instructions and a complete README.
12. Keep implementation straightforward, modular, and easy to explain.

---

## 3. Recommended Technology Stack

Use the following by default:

- **Language:** Python 3.10 or later (prefer a version compatible with all dependencies)
- **UI:** Streamlit
- **ML/NLP:** scikit-learn
- **Data processing:** pandas, NumPy
- **Charts:** Plotly or Matplotlib
- **Model serialization:** joblib
- **Optional text utilities:** NLTK or spaCy only if genuinely needed

Do not use an external LLM API or require paid credentials. Do not require a GPU. Do not add a database for the first version.

Use a scikit-learn `Pipeline` containing a TF-IDF vectorizer and Logistic Regression classifier. This ensures the same vectorization steps are applied during training and inference. Use `predict_proba` only if supported by the chosen classifier.

---

## 4. Required Features

### 4.1 Headline Input and Validation

- Provide a text area for entering or pasting a headline.
- Include an **Analyze Headline** button.
- Include a **Clear** action if practical.
- Reject empty or whitespace-only inputs.
- Enforce a reasonable maximum input length and show a friendly message.
- Handle prediction errors gracefully.
- Do not crash if the model file is missing; show clear instructions to train/download the model first.
- Show a disclaimer near the result: predictions are model-based and are not factual verification.

### 4.2 NLP Preprocessing

Use consistent, conservative preprocessing. TF-IDF can handle much of the text processing itself.

- Trim surrounding whitespace.
- Normalize repeated whitespace.
- Preserve meaningful punctuation and negations where possible.
- Do not blindly remove words such as “not”, “never”, “no”, or “without”.
- Avoid aggressive stemming/lemmatization unless justified.
- Keep the same preprocessing behavior at training and inference time.
- Avoid training/inference mismatch.

### 4.3 Classification

- Predict one of two labels: `Likely Real` or `Likely Fake`.
- Train at least one baseline model: TF-IDF + Logistic Regression.
- If time permits, compare against Multinomial Naive Bayes using the same split.
- Use a fixed random seed for reproducibility.
- Keep class label mapping explicit and documented. Never assume that class index 0 means fake or real without inspecting the model's `classes_`.
- Do not train a model every time the app reruns.
- Save the fitted pipeline to a model file such as `models/news_pipeline.joblib`.

### 4.4 Prediction Probabilities / Score

- Display estimated probabilities for both classes if the model supports them.
- Identify the predicted class correctly using the fitted model's class ordering.
- Label scores as **model-estimated probabilities**, not factual truth probabilities.
- Do not invent sample scores. All displayed live scores must come from the model.
- Avoid implying that a high score means a claim has been fact-checked.

### 4.5 Text Insights

For the entered headline, display:
- Word count
- Character count
- Average word length (if useful)
- Top relevant TF-IDF terms or other meaningful text features, where available

Do not present generic statistics as evidence of truthfulness. If showing words associated with a class, explain that they reflect dataset/model associations.

### 4.6 Prediction Explanation

For Logistic Regression:
- Use learned feature coefficients and the transformed headline to estimate which present features contribute toward each class.
- Respect the actual class order from the model.
- Show a short list of terms contributing toward the predicted class and, where useful, terms pushing toward the alternative class.
- If a token is not in the vocabulary, do not pretend it influenced the prediction.
- Use clear explanatory text: these are statistical associations, not causal proof or evidence of deception.
- If robust explanations would add too much complexity, implement a simple and accurate coefficient-based explanation rather than a misleading explanation.

### 4.7 Prediction History

- Keep history in Streamlit session state initially; no database is required.
- Record headline, predicted label, model-estimated probability for predicted class, and timestamp.
- Display history in a table.
- Provide a clear-history action.
- Provide CSV export for history if practical.
- Make sure history is session-scoped and do not imply it persists after the session ends.

### 4.8 Batch CSV Analysis

- Allow users to upload a CSV file containing a headline/title column.
- Let users select the headline column if several columns are present.
- Validate the selected column and skip or clearly report empty rows.
- Add columns to the output for predicted label and class probabilities if available.
- Allow downloading the results as CSV.
- Do not silently discard failed rows or misalign predictions with input rows.
- Limit file size or row count sensibly to avoid performance issues.

### 4.9 Model Evaluation Dashboard

Create an evaluation section that reports actual held-out test-set results:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Optional class distribution and comparison of models

Requirements:
- Use a stratified train/test split where appropriate.
- Split before fitting TF-IDF; never fit vectorization on the full dataset before splitting.
- Avoid data leakage and check for duplicate or near-duplicate headlines/articles when feasible.
- Use a fixed random seed.
- Clearly show which dataset and split produced the metrics.
- Do not hardcode or fabricate performance numbers.
- If the model has not been trained/evaluated, show an honest setup message rather than fake charts.
- For binary classification, state which class is treated as the positive class when reporting precision/recall.
- Use a confusion matrix with clear axis labels and consistent class ordering.

### 4.10 Optional Model Comparison

If practical, compare:
1. TF-IDF + Logistic Regression
2. TF-IDF + Multinomial Naive Bayes

Use the same train/test split and preprocessing for fair comparison. Report actual metrics and make it clear that model selection should consider precision/recall/F1, not accuracy alone.

### 4.11 UI / UX

Create a professional, modern, responsive Streamlit dashboard with a coherent visual identity.

Brand:
- Name: **TruthLens**
- Subtitle: **NLP-Powered News Analysis**
- Style: clean, modern, trustworthy; dark navy/blue accents are acceptable
- Avoid clutter, excessive animations, and unnecessary UI complexity.

Suggested navigation:
1. **Home** — headline input and prediction result
2. **Text Insights** — text statistics and model explanation
3. **Batch Analysis** — CSV upload, classification, download
4. **Model Performance** — evaluation metrics and confusion matrix

UI requirements:
- Clear heading and short project description.
- Input and result should be visually distinct.
- Show a helpful empty state before a prediction is made.
- Use readable charts and tables.
- Use labels and colors consistently; do not rely on color alone to convey the class.
- Show a visible disclaimer explaining model limitations.
- Display errors and setup instructions in plain language.
- Ensure the app is usable on common laptop screen sizes.

---

## 5. Dataset Strategy

The app must not depend on a made-up dataset or silently assume a dataset exists.

Support a configurable dataset path, for example:
`data/news_dataset.csv`

Allow dataset column names to be configured or selected. Typical source columns might include `title`, `text`, `label`, or `subject`, but these vary by dataset.

### Dataset options

Potential starting points:
- **ISOT Fake News Dataset:** commonly used for real/fake news classification, often containing full article text and titles. Verify the dataset source, license/terms, columns, and labels before use.
- **LIAR dataset:** contains short political statements with multiple truthfulness labels; it is not a simple headline-only binary dataset. Do not use it without documenting a defensible mapping from its original labels.

### Dataset requirements

- Include clear instructions on obtaining a suitable public dataset; do not claim it is bundled unless it actually is.
- Do not scrape websites without permission or ignore terms of service.
- Inspect class distribution and missing values.
- Confirm the dataset actually has the required text and label fields.
- Normalize labels explicitly and document the mapping.
- If only article bodies are available, clearly explain that this is not equivalent to headline-only classification.
- Prefer headline/title text for the intended task.
- If using full articles, make the UI/report accurately describe the model as article-text classification rather than headline-only classification.
- Avoid leakage from publisher/source names, article metadata, duplicated stories, and collection artifacts where possible.
- Do not report universal accuracy or claim the system works for all news domains.

---

## 6. Suggested Project Structure

Use a simple modular layout:

```text
TruthLens/
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── explainability.py
│   └── evaluation.py
└── tests/
    └── test_prediction.py
```

Adjust the structure if needed, but keep it simple. Do not create lots of empty abstractions or overengineer the project.

### File responsibilities

- `app.py`: Streamlit pages/navigation, input forms, results, history, batch upload and charts.
- `train_model.py`: load/validate dataset, split data, train models, evaluate, save pipeline and metrics.
- `predict.py`: load saved pipeline safely and expose reusable prediction functions.
- `utils/preprocessing.py`: conservative, shared text normalization helpers if needed.
- `utils/explainability.py`: coefficient-based feature contribution logic.
- `utils/evaluation.py`: metric calculation and confusion-matrix helpers.
- `tests/test_prediction.py`: basic tests for empty input, valid prediction, class mapping and batch alignment.
- `README.md`: setup, dataset instructions, training, running, limitations and troubleshooting.
- `requirements.txt`: only dependencies actually used.

Save the fitted pipeline and evaluation metadata in a documented format. Ensure model artifacts and datasets are excluded from Git if they are large or their license does not permit redistribution.

---

## 7. Training and Inference Workflow

### Training (`train_model.py`)

1. Load CSV dataset from a configurable path.
2. Validate required columns and report available columns if configuration is wrong.
3. Drop or report rows with missing text/labels.
4. Normalize text labels with explicit mappings.
5. Inspect and report class counts.
6. Split the data into training and test sets, stratifying by label where appropriate.
7. Check duplicates and leakage risks.
8. Create a scikit-learn pipeline using `TfidfVectorizer` and `LogisticRegression`.
9. Fit only on training data.
10. Evaluate on held-out test data.
11. Optionally train a Multinomial Naive Bayes baseline using the same split.
12. Save the fitted pipeline and evaluation results.
13. Print a clear training summary, including dataset path, row counts, split details, class mapping and actual metrics.
14. Fail gracefully with useful messages when the dataset is absent or invalid.

### Inference (`predict.py`)

Expose reusable functions for:
- Loading the saved pipeline.
- Validating and normalizing a headline.
- Returning predicted label.
- Returning class probabilities if supported.
- Returning useful text statistics.
- Returning feature-contribution explanations where supported.

Do not reload/retrain the model for every headline unnecessarily. In Streamlit, use resource caching appropriately when loading the model.

---

## 8. Privacy, Security, and Reliability

- Process text locally by default; do not transmit user headlines to third-party APIs.
- Do not log sensitive user-entered content outside session history.
- Validate uploaded file type and size.
- Handle malformed CSVs, missing columns, empty files, invalid text, and model-loading errors.
- Avoid unsafe deserialization from untrusted model files. Only load the model artifact generated by this project's own trusted training workflow.
- Never execute uploaded files.
- Keep dependencies minimal.
- Use readable, commented code suitable for a student project.
- Add basic tests for core functionality.
- Do not include secrets or API keys.

---

## 9. Testing Checklist

At minimum, test:
- Empty headline.
- Whitespace-only headline.
- Normal headline.
- Very long input.
- Punctuation and numbers.
- Negation words such as “not” and “never”.
- Unknown/unseen vocabulary.
- Missing model artifact.
- Missing or incorrect dataset columns.
- CSV with blank headline rows.
- CSV with multiple candidate text columns.
- CSV output preserves row alignment.
- Both predicted classes map to the correct human-readable label.
- Probability display follows `classes_` order.
- Evaluation charts use the same class order as metrics.
- Session history clears correctly.
- Model metrics are generated from actual test predictions.

---

## 10. README Requirements

Write a complete `README.md` that includes:
- Project name and summary.
- Features.
- Technology stack.
- Project folder structure.
- Python version requirements.
- Virtual environment setup for Windows and macOS/Linux.
- Dependency installation.
- Dataset acquisition and placement instructions.
- How to configure dataset column names/labels.
- How to train the model.
- How to launch Streamlit.
- How to run tests.
- Troubleshooting.
- Limitations and ethical considerations.
- Example workflow.

Do not state that a dataset is included unless one is actually provided. Do not include fabricated screenshots, metrics, or model results.

---

## 11. Suggested Implementation Phases

Build incrementally and make each phase runnable before proceeding.

### Phase 1 — Working classifier
- Dataset loader and validation.
- TF-IDF + Logistic Regression pipeline.
- Training/test split and real evaluation metrics.
- Save/load model.
- Simple prediction interface.

### Phase 2 — Polished single-headline UI
- Brand identity and layout.
- Input validation.
- Predicted class and class probabilities.
- Text statistics.
- Disclaimer.

### Phase 3 — Useful supporting features
- Session history.
- CSV batch classification and export.
- Model performance dashboard.

### Phase 4 — Explainability and quality
- Coefficient-based word contributions.
- Model comparison.
- Unit tests.
- Final README and UI polish.

Prioritize correctness over visual decoration. The app should remain useful if optional features are unavailable.

---

## 12. Acceptance Criteria

The project is complete when:

1. It installs using the documented dependencies.
2. It trains from a real, user-supplied labeled dataset.
3. It saves and reloads the fitted pipeline.
4. A valid headline receives a prediction from the actual model.
5. The class mapping and displayed probabilities are correct.
6. The UI clearly explains that the output is not fact verification.
7. Evaluation metrics are calculated from a held-out test set, not hardcoded.
8. CSV batch predictions preserve input rows and can be downloaded.
9. History works within the current session.
10. Missing files and invalid input produce helpful messages rather than crashes.
11. Tests cover important prediction and data-handling behavior.
12. The README allows another student to set up and run the project.

---

## 13. Instructions for Claude Code

You are working as a coding agent inside the user's project workspace.

1. First inspect the existing workspace and its files. Do not overwrite unrelated files.
2. If the workspace is empty, create the project structure above.
3. Implement the project in phases, prioritizing a working, correct classifier.
4. Do not assume a dataset exists. If none is present, build dataset validation and provide exact instructions for adding one; do not fabricate training data or performance results.
5. Do not use paid APIs, API keys, GPU dependencies, or an LLM.
6. Keep code readable, modular and beginner-friendly, with comments where they improve understanding.
7. Use actual fitted model outputs for predictions and metrics.
8. Test the core functions. If dependencies are unavailable, explain what could not be tested.
9. Run the app or tests when possible and fix errors you can reproduce.
10. At the end, summarize:
   - files created/modified,
   - installation commands,
   - dataset requirements,
   - training command,
   - app launch command,
   - test command,
   - known limitations and any steps requiring user action.
11. Do not claim that the app works or tests pass unless you have actually verified that.
12. If a design choice is ambiguous, prefer the simplest reliable option that meets this specification.

**Build a complete, runnable student project—not merely a mockup or static dashboard.**
