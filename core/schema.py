"""
The single output format every analysis mode returns.

The UI only reads this dict, so V2 (evidence mode) can be added later
without changing the UI code that renders V1.

{
  "mode": "style" | "evidence",
  "input": original text,
  "cleaned": text after clean_title(),
  "verdict": one of VERDICTS,
  "confidence": float 0..1 (model-estimated, not "truth"),
  "probabilities": {"Likely Real": p, "Likely Fake": p}   # V1 only
  "word_contributions": [{"term": str, "score": float}]    # +score pushes to Fake,
                                                            # -score pushes to Real
  "evidence": [{"sentence", "source", "url", "stance", "score"}]  # V2 only
  "warnings": [str],
  "error": str | None,
}
"""

REAL = "Likely Real"
FAKE = "Likely Fake"
UNCERTAIN = "Uncertain"

# V2 verdicts, defined now so the UI can already handle them.
SUPPORTED = "Supported"
REFUTED = "Refuted"
NOT_ENOUGH = "Not enough evidence"

VERDICTS = [REAL, FAKE, UNCERTAIN, SUPPORTED, REFUTED, NOT_ENOUGH]

# Labels stored inside the trained model. Kept explicit (never "0 = fake").
MODEL_LABEL_TO_VERDICT = {"fake": FAKE, "real": REAL}

DISCLAIMER = (
    "This is a machine-learning prediction based on writing patterns learned "
    "from a 2016-2017 news dataset. It is not fact verification."
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
        "evidence": [],
        "warnings": [],
        "error": None,
    }
