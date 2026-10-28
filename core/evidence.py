"""
V2 "evidence" analysis: find evidence for a claim and decide
Supported / Refuted / Not enough evidence.

Pipeline:
  1. Fact Check API  -> existing fact-checks; keep only those about the *same*
                        claim (similar meaning AND the NLI model says one
                        statement implies the other).
  2. Wikipedia       -> sentences from the top articles.
  3. Ranking         -> keep the Wikipedia sentences most similar to the claim.
  4. NLI             -> does each sentence entail or contradict the claim?
  5. Verdict         -> matching fact-checks decide first (human experts);
                        otherwise the MOST RELEVANT sentence the NLI model is
                        confident about decides.

Why "most relevant confident sentence" instead of "strongest anywhere":
the NLI model often says "contradiction" for sentences that are merely about
something else (e.g. "Chandrayaan-3 was launched on 14 July" vs "... landed").
Those sentences are less similar to the claim than the ones that really
address it, so going in similarity order avoids most of these errors.
"""

import re

from core import retrieval
from core.nlp_models import nli, rank_by_similarity
from core.schema import NOT_ENOUGH, REFUTED, SUPPORTED

# Thresholds, tuned on the demo claims (check_demo_claims.py). See README.
FACTCHECK_SIM = 0.60     # min similarity between our claim and a fact-checked claim
FACTCHECK_NLI = 0.50     # min entailment (either direction) to treat them as the same claim
CANDIDATES = 15          # Wikipedia sentences sent to the NLI model
RELEVANCE = 0.55         # min similarity for a sentence to be allowed to decide
STANCE_PROB = 0.80       # min NLI probability to call a sentence supporting/refuting
CONFLICT_WINDOW = 0.05   # warn if an opposite sentence is almost as relevant as the decider
SHOW_EVIDENCE = 6

_STOPWORDS = set("""a an the is are was were be been being of in on at to for from by with and or
that this these those it its as into than then there their they he she his her has have had do does
did not no new last week claim claims said says""".split())

_FALSE_WORDS = ["false", "fake", "incorrect", "misleading", "pants on fire", "hoax",
                "not true", "untrue", "wrong", "fabricated", "baseless", "inaccurate",
                "no evidence", "scam", "misattributed", "distorted", "unproven", "myth"]
_MIXED_WORDS = ["half true", "half-true", "mixture", "mixed", "partly", "partially",
                "missing context", "needs context", "unverified"]
_TRUE_WORDS = ["true", "correct", "accurate", "verified", "confirmed", "real"]


def rating_to_stance(rating: str) -> str:
    """Map a fact-checker's free-text rating to supports / refutes / neutral."""
    r = (rating or "").lower()
    # Order matters: "mostly false" and "not true" contain "true".
    if any(w in r for w in _MIXED_WORDS):
        return "neutral"
    if any(w in r for w in _FALSE_WORDS):
        return "refutes"
    if any(w in r for w in _TRUE_WORDS):
        return "supports"
    return "neutral"


def keyword_queries(claim: str) -> list[str]:
    """
    The Fact Check API only returns results containing ALL query words, so a
    full sentence often finds nothing. Fallbacks: content words only, then the
    same with one word left out at a time ("GPS" was missing from the
    2000-rupee fact-check, for example).
    """
    words = [w for w in re.findall(r"[A-Za-z0-9]+", claim) if w.lower() not in _STOPWORDS]
    queries = [" ".join(words)]
    if len(words) > 3:
        queries += [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    return queries


def factcheck_evidence(claim: str, warnings: list) -> list[dict]:
    reviews, error = retrieval.search_factchecks(claim)
    for q in keyword_queries(claim):
        if reviews or error:
            break
        reviews, error = retrieval.search_factchecks(q)
    if error:
        warnings.append(error)
    if not reviews:
        return []

    ranked = [(i, s) for i, s in rank_by_similarity(claim, [r["claim_text"] for r in reviews], len(reviews))
              if s >= FACTCHECK_SIM]
    if not ranked:
        return []
    texts = [reviews[i]["claim_text"] for i, _ in ranked]
    forward = nli(texts, claim)                       # fact-checked claim -> our claim
    backward = [nli([claim], t)[0] for t in texts]    # our claim -> fact-checked claim
    items, seen = [], set()
    for (idx, sim), f, b in zip(ranked, forward, backward):
        r = reviews[idx]
        stance = rating_to_stance(r["rating"])
        # A verdict only carries over in the logically valid direction:
        #  - rated False: our claim must IMPLY the fact-checked claim
        #    (if ours implies something false, ours is false too).
        #  - rated True: the fact-checked claim must imply ours.
        # This drops related-but-different fact-checks, e.g. "Video shows
        # Chandrayaan-3 landing" (False, the video was fake) does not refute
        # "Chandrayaan-3 landed".
        needed = b["entailment"] if stance == "refutes" else f["entailment"]
        if stance == "neutral" or needed < FACTCHECK_NLI or r["url"] in seen:
            continue
        seen.add(r["url"])
        items.append({
            "sentence": f'"{r["claim_text"]}" — rated "{r["rating"]}"',
            "source": f"Fact-check · {r['publisher']}",
            "url": r["url"],
            "stance": stance,
            "score": round(needed, 3),
        })
    return items


def wikipedia_evidence(claim: str, warnings: list) -> list[dict]:
    sentences, error = retrieval.search_wikipedia(claim)
    if error:
        warnings.append(error)
    if not sentences:
        return []
    ranked = [(i, s) for i, s in rank_by_similarity(claim, [x["sentence"] for x in sentences], CANDIDATES)
              if s >= RELEVANCE]
    if not ranked:
        return []
    probs = nli([sentences[i]["sentence"] for i, _ in ranked], claim)
    items = []
    for (i, sim), p in zip(ranked, probs):
        if p["entailment"] >= STANCE_PROB:
            stance, score = "supports", p["entailment"]
        elif p["contradiction"] >= STANCE_PROB:
            stance, score = "refutes", p["contradiction"]
        else:
            stance, score = "neutral", p["neutral"]
        items.append({
            "sentence": sentences[i]["sentence"],
            "source": f"Wikipedia · {sentences[i]['title']}",
            "url": sentences[i]["url"],
            "stance": stance,
            "score": round(score, 3),
            "similarity": round(sim, 3),
        })
    return items


def decide(fc_items: list[dict], wiki_items: list[dict]) -> tuple[str, float, str | None]:
    """Return (verdict, confidence, optional warning)."""
    fc_sup = [e for e in fc_items if e["stance"] == "supports"]
    fc_ref = [e for e in fc_items if e["stance"] == "refutes"]
    if len(fc_ref) > len(fc_sup):
        return REFUTED, max(e["score"] for e in fc_ref), None
    if len(fc_sup) > len(fc_ref):
        return SUPPORTED, max(e["score"] for e in fc_sup), None
    # No matching fact-checks (or they disagree): use Wikipedia.

    # Most relevant first; only sentences relevant enough may decide.
    decisive = [e for e in sorted(wiki_items, key=lambda e: -e["similarity"])
                if e["similarity"] >= RELEVANCE and e["stance"] != "neutral"]
    if not decisive:
        return NOT_ENOUGH, 0.0, None
    top = decisive[0]
    verdict = SUPPORTED if top["stance"] == "supports" else REFUTED
    note = None
    if any(e["stance"] != top["stance"] and top["similarity"] - e["similarity"] < CONFLICT_WINDOW
           for e in decisive[1:]):
        note = "Another highly relevant sentence points the other way; check the evidence below."
    return verdict, top["score"], note


def check_claim(claim: str, result: dict) -> dict:
    """Fill a schema result dict (see core/schema.py) with the V2 verdict and evidence."""
    warnings = result["warnings"]
    fc_items = factcheck_evidence(claim, warnings)
    wiki_items = wikipedia_evidence(claim, warnings)

    verdict, confidence, note = decide(fc_items, wiki_items)
    if note:
        warnings.append(note)
    result["verdict"] = verdict
    result["confidence"] = round(confidence, 4)

    # Show decisive evidence first, then the most relevant neutral sentences.
    wiki_items.sort(key=lambda e: (e["stance"] == "neutral", -e["score"]))
    result["evidence"] = fc_items + wiki_items[: SHOW_EVIDENCE]
    return result
