"""
The single output format every analysis mode returns.

The UI only reads this dict, so a new mode never changes how the UI reads
results. Two modes exist:

  "style"    - V1: TF-IDF + Logistic Regression on the wording alone.
  "evidence" - V2: looks for the claim in fact-checks, Wikipedia and news, and
               judges each passage with an NLI model (or, if PyTorch isn't
               installed, with word matching).

{
  "mode": "style" | "evidence",
  "input": original text,
  "cleaned": text after cleaning (style) / the claim that was checked (evidence),
  "verdict": one of VERDICTS (None if there was an error),
  "confidence": float 0..1 (model-estimated, not "truth"; None if not applicable),
  "probabilities": {"Likely Real": p, "Likely Fake": p}          # style only
  "word_contributions": [{"term": str, "score": float}]          # style only
                       # +score pushes toward Fake, -score toward Real
  "tfidf_terms": [{"term": str, "weight": float}]                # style only
  "stats": {"words", "characters", "avg_word_length", ...}       # both modes
  "engine": "nli" | "words",  # how evidence stances were judged # evidence only
  "summary": str,             # one-sentence reason for the verdict
  "keywords": [str],          # search terms sent to the sources
  "evidence": [{"source", "publisher", "title", "sentence", "url", "date",
                "stance", "score", "rating", "matched",
                "similarity" (NLI only: relevance to the claim),
                "deciding" (True on the item that decided the verdict)}]
  "sources_checked": [{"name", "ok", "skipped", "found", "message"}]
  "warnings": [str],
  "error": str | None,
}
"""

REAL = "Likely Real"
FAKE = "Likely Fake"
UNCERTAIN = "Uncertain"

# V2 verdicts.
SUPPORTED = "Supported"
REFUTED = "Refuted"
NOT_ENOUGH = "Not enough evidence"

VERDICTS = [REAL, FAKE, UNCERTAIN, SUPPORTED, REFUTED, NOT_ENOUGH]

# Stance of one piece of evidence toward the claim.
SUPPORTS = "supports"
REFUTES = "refutes"
NEUTRAL = "neutral"

# Labels stored inside the trained model. Kept explicit (never "0 = fake").
MODEL_LABEL_TO_VERDICT = {"fake": FAKE, "real": REAL}

DISCLAIMER = (
    "This is a machine-learning prediction based on writing patterns learned "
    "from older news headlines (US 2016-17, Indian 2013-21). It is not fact verification."
)

EVIDENCE_DISCLAIMER = (
    "The evidence check is automated: retrieval can miss the right article or pick the "
    "wrong one, and the models can misread a sentence. Always open the sources."
)


def empty_result(mode: str, text: str) -> dict:
    return {
        "mode": mode,
        "input": text,
        "cleaned": "",
        "verdict": None,
        "confidence": None,
        "probabilities": {},
        "word_contributions": [],
        "tfidf_terms": [],
        "stats": {},
        "engine": None,
        "summary": "",
        "keywords": [],
        "evidence": [],
        "sources_checked": [],
        "warnings": [],
        "error": None,
    }
