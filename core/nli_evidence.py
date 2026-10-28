"""
V2 "evidence" analysis with NLI models: find evidence for a claim and decide
Supported / Refuted / Not enough evidence. Used when PyTorch is installed;
otherwise core/evidence.py falls back to word matching.

Pipeline:
  1. Fact Check API  -> existing fact-checks; keep only those about the *same*
                        claim (similar meaning AND the NLI model says one
                        statement implies the other).
  2. Candidates      -> sentences from the top Wikipedia articles, plus news
                        sentences from Wikipedia's Current events archive and
                        GDELT (fetched by core/evidence.py).
  3. Ranking         -> keep the candidate sentences most similar to the claim.
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
from core.schema import NEUTRAL, NOT_ENOUGH, REFUTED, REFUTES, SUPPORTED, SUPPORTS
from core.sources import FACT_CHECK, WIKIPEDIA

# Thresholds, tuned on the demo claims (check_demo_claims.py). See README.
FACTCHECK_SIM = 0.60     # min similarity between our claim and a fact-checked claim
FACTCHECK_NLI = 0.50     # min entailment (either direction) to treat them as the same claim
CANDIDATES = 15          # candidate sentences sent to the NLI model
RELEVANCE = 0.55         # min similarity for a sentence to be allowed to decide
STANCE_PROB = 0.80       # min NLI probability to call a sentence supporting/refuting
CONFLICT_WINDOW = 0.05   # warn if an opposite sentence is almost as relevant as the decider
SHOW_EVIDENCE = 6

_STOPWORDS = set("""a an the is are was were be been being of in on at to for from by with and or
that this these those it its as into than then there their they he she his her has have had do does
did not no new last week claim claims said says""".split())

_FALSE_WORDS = ["false", "fake", "incorrect", "misleading", "pants on fire", "hoax",
                "not true", "untrue", "wrong", "fabricated", "baseless", "inaccurate",
                "no evidence", "scam", "misattributed", "distorted", "unproven", "myth",
                "altered", "doctored", "debunked", "satire"]
_MIXED_WORDS = ["half true", "half-true", "mixture", "mixed", "partly", "partially",
                "missing context", "needs context", "unverified", "exaggerat"]
_TRUE_WORDS = ["true", "correct", "accurate", "verified", "confirmed", "real"]


def rating_to_stance(rating: str) -> str:
    """Map a fact-checker's free-text rating to supports / refutes / neutral."""
    r = (rating or "").lower()
    # Order matters: "mostly false" and "not true" contain "true".
    if any(w in r for w in _MIXED_WORDS):
        return NEUTRAL
    if any(w in r for w in _FALSE_WORDS):
        return REFUTES
    if any(w in r for w in _TRUE_WORDS):
        return SUPPORTS
    return NEUTRAL


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


def factcheck_evidence(claim: str, warnings: list, api_key: str | None = None) -> tuple[list[dict], dict]:
    """(evidence items, source status) from published fact-checks of the same claim."""
    status = {"name": FACT_CHECK, "ok": True, "skipped": False, "found": 0, "message": ""}
    reviews, error = retrieval.search_factchecks(claim, api_key)
    for q in keyword_queries(claim):
        if reviews or error:
            break
        reviews, error = retrieval.search_factchecks(q, api_key)
    if error:
        no_key = "No Fact Check API key" in error
        status.update(ok=False, skipped=no_key,
                      message="add a free Google Fact Check API key to include published fact-checks"
                      if no_key else error)
        if not no_key:
            warnings.append(error)
    status["found"] = len(reviews)
    if not reviews:
        return [], status

    ranked = [(i, s) for i, s in rank_by_similarity(claim, [r["claim_text"] for r in reviews], len(reviews))
              if s >= FACTCHECK_SIM]
    if not ranked:
        return [], status
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
        needed = b["entailment"] if stance == REFUTES else f["entailment"]
        if stance == NEUTRAL or needed < FACTCHECK_NLI or r["url"] in seen:
            continue
        seen.add(r["url"])
        items.append({
            "source": FACT_CHECK,
            "publisher": r["publisher"],
            "title": r["title"] or r["claim_text"],
            "sentence": r["claim_text"],
            "url": r["url"],
            "date": None,
            "rating": r["rating"],
            "stance": stance,
            "score": round(needed, 3),
        })
    return items, status


def wikipedia_candidates(claim: str) -> tuple[list[dict], dict]:
    """(candidate sentences, source status) from the top Wikipedia articles."""
    sentences, error = retrieval.search_wikipedia(claim)
    status = {"name": WIKIPEDIA, "ok": error is None, "skipped": False,
              "found": len({s["title"] for s in sentences}), "message": error or ""}
    candidates = [{"sentence": s["sentence"], "source": WIKIPEDIA, "publisher": "wikipedia.org",
                   "title": s["title"], "url": s["url"], "date": None} for s in sentences]
    return candidates, status


def sentence_evidence(claim: str, candidates: list[dict]) -> list[dict]:
    """Rank candidate sentences by similarity to the claim, then judge the best with NLI."""
    if not candidates:
        return []
    ranked = [(i, s) for i, s in rank_by_similarity(claim, [c["sentence"] for c in candidates], CANDIDATES)
              if s >= RELEVANCE]
    if not ranked:
        return []
    probs = nli([candidates[i]["sentence"] for i, _ in ranked], claim)
    items = []
    for (i, sim), p in zip(ranked, probs):
        if p["entailment"] >= STANCE_PROB:
            stance, score = SUPPORTS, p["entailment"]
        elif p["contradiction"] >= STANCE_PROB:
            stance, score = REFUTES, p["contradiction"]
        else:
            stance, score = NEUTRAL, p["neutral"]
        items.append({**candidates[i], "rating": None, "stance": stance,
                      "score": round(score, 3), "similarity": round(sim, 3)})
    return items


def decide(fc_items: list[dict], wiki_items: list[dict]) -> tuple[str, float, str | None]:
    """
    Return (verdict, confidence, optional warning).
    The evidence item that decided gets "deciding": True, so the UI can show it first.
    """
    fc_sup = [e for e in fc_items if e["stance"] == SUPPORTS]
    fc_ref = [e for e in fc_items if e["stance"] == REFUTES]
    if len(fc_ref) != len(fc_sup):
        verdict, winners = (REFUTED, fc_ref) if len(fc_ref) > len(fc_sup) else (SUPPORTED, fc_sup)
        best = max(winners, key=lambda e: e["score"])
        best["deciding"] = True
        return verdict, best["score"], None
    # No matching fact-checks (or they disagree): use the candidate sentences.

    # Most relevant first; only sentences relevant enough may decide.
    decisive = [e for e in sorted(wiki_items, key=lambda e: -e["similarity"])
                if e["similarity"] >= RELEVANCE and e["stance"] != NEUTRAL]
    if not decisive:
        return NOT_ENOUGH, 0.0, None
    top = decisive[0]
    top["deciding"] = True
    verdict = SUPPORTED if top["stance"] == SUPPORTS else REFUTED
    note = None
    if any(e["stance"] != top["stance"] and top["similarity"] - e["similarity"] < CONFLICT_WINDOW
           for e in decisive[1:]):
        note = "Another highly relevant sentence points the other way; check the evidence below."
    return verdict, top["score"], note


def summarize(verdict: str, evidence: list[dict]) -> str:
    """One sentence saying what decided the verdict."""
    top = next((e for e in evidence if e.get("deciding")), None)
    if top is None:
        return ("No sentence was both relevant enough to the claim and clearly supporting or "
                "contradicting it.")
    if top["source"] == FACT_CHECK:
        return f'A published fact-check by {top["publisher"]} rated a matching claim "{top["rating"]}".'
    verb = "supports" if verdict == SUPPORTED else "contradicts"
    return (f"The most relevant sentence the NLI model is confident about ({top['publisher']}) "
            f"{verb} the claim.")


def check_claim(claim: str, result: dict, api_key: str | None = None, more_candidates=None) -> dict:
    """
    Fill a schema result dict (see core/schema.py) with the V2 verdict and evidence.
    more_candidates() returns extra candidate sentences (e.g. news); it is called
    after the fact-check and Wikipedia searches, so it can run in the background.
    """
    warnings = result["warnings"]
    fc_items, fc_status = factcheck_evidence(claim, warnings, api_key)
    candidates, wiki_status = wikipedia_candidates(claim)
    if wiki_status["message"]:
        warnings.append(wiki_status["message"])
    result["sources_checked"] += [wiki_status, fc_status]
    if more_candidates:
        candidates += more_candidates()
    wiki_items = sentence_evidence(claim, candidates)

    verdict, confidence, note = decide(fc_items, wiki_items)
    if note:
        warnings.append(note)
    result["verdict"] = verdict
    result["confidence"] = round(confidence, 4)

    # Show evidence in the order it was weighed: the deciding item first, then
    # fact-checks, then sentences by relevance (NOT by NLI confidence, which
    # made barely-relevant "contradictions" look like the main evidence).
    wiki_items.sort(key=lambda e: -e["similarity"])
    evidence = fc_items + wiki_items[:SHOW_EVIDENCE]
    evidence.sort(key=lambda e: not e.get("deciding", False))  # stable sort keeps the rest in order
    result["evidence"] = evidence
    result["summary"] = summarize(verdict, evidence)
    return result
