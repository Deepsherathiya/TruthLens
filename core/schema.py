"""
The single output format every analysis mode returns.

The UI only reads this dict, so a new mode never changes how the UI reads
results. Two modes exist:

  "style"    - V1: TF-IDF + Logistic Regression on the wording alone.
  "evidence" - V2: searches Wikipedia, recent news and fact-checkers for
               independent sources that match the claim.

{
  "mode": "style" | "evidence",
  "input": original text,
  "cleaned": text after clean_title(),
  "verdict": one of VERDICTS (None if there was an error),
  "confidence": float 0..1 (style: model-estimated; evidence: None),
  "probabilities": {"Likely Real": p, "Likely Fake": p}          # style only
  "word_contributions": [{"term": str, "score": float}]          # style only
                       # +score pushes toward Fake, -score toward Real
  "tfidf_terms": [{"term": str, "weight": float}]                # style only
  "stats": {"words", "characters", "avg_word_length", ...}       # both modes
  "summary": str,                                                # evidence only
  "keywords": [str],          # search terms sent to the sources # evidence only
  "evidence": [{"source", "publisher", "title", "sentence", "url", "date",
                "stance", "score", "rating", "matched"}]          # evidence only
  "sources_checked": [{"name", "ok", "skipped", "found", "message"}]
  "warnings": [str],
  "error": str | None,
}
"""

REAL = "Likely Real"
FAKE = "Likely Fake"
UNCERTAIN = "Uncertain"

# V2 verdicts. Deliberately cautious wording: "Corroborated" means independent
# sources report the same thing, not that the claim is proven true.
CORROBORATED = "Corroborated"
DISPUTED = "Disputed"
NOT_ENOUGH = "Not enough evidence"

VERDICTS = [REAL, FAKE, UNCERTAIN, CORROBORATED, DISPUTED, NOT_ENOUGH]

# Stance of one piece of evidence toward the claim.
SUPPORTS = "supports"
CONTRADICTS = "contradicts"
RELATED = "related"

# Labels stored inside the trained model. Kept explicit (never "0 = fake").
MODEL_LABEL_TO_VERDICT = {"fake": FAKE, "real": REAL}

DISCLAIMER = (
    "This is a machine-learning prediction based on writing patterns learned "
    "from a 2016-2017 news dataset. It is not fact verification."
)

EVIDENCE_DISCLAIMER = (
    "The source check matches key words, names and numbers against Wikipedia, "
    "recent news and fact-checkers. A match means other sources say something "
    "similar, not that the claim is proven true; no match does not prove it false."
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
        "summary": "",
        "keywords": [],
        "evidence": [],
        "sources_checked": [],
        "warnings": [],
        "error": None,
    }
