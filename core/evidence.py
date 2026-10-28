"""
V2 "evidence" analysis: look for the claim in fact-checks, Wikipedia and news.

check_sources() picks one of two engines for judging the evidence:

  "nli"   (core/nli_evidence.py) when PyTorch and sentence-transformers are
          installed: sentence embeddings rank candidate sentences and an NLI
          model decides whether each one supports or contradicts the claim.
  "words" (this file) otherwise, or if the models fail to load: word matching
          that needs no ML model, so every step can be checked by hand:
            1. parse_claim(): the claim's key terms (content words, names,
               numbers) and whether it is negated ("not", "never", "denies").
            2. match(): what share of those key terms a passage contains.
            3. stance(): supports / refutes / neutral for each passage.
            4. decide(): combine the stances into one verdict.

Both engines search the news archive and recent news (core/sources.py), so
news headlines get evidence even when Wikipedia has nothing yet.
Set TRUTHLENS_ENGINE=words to force word matching (e.g. on a slow machine).

Known limits of word matching: a paraphrase ("approves" vs "passes") can be
missed, and a contradiction that doesn't use a negation word goes unnoticed.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from core import nlp_models, sources
from core.nli_evidence import rating_to_stance
from core.preprocessing import normalise, question_to_statement
from core.schema import NEUTRAL, NOT_ENOUGH, REFUTED, REFUTES, SUPPORTED, SUPPORTS

# Thresholds: share of the claim's (weighted) key terms found in a passage.
# STRONG_MATCH is deliberately high: with a short claim like "Paris is the
# capital of Germany", a passage missing just "capital" must not count.
STRONG_MATCH = 0.75      # close enough to treat as the same claim
RELATED_MATCH = 0.35     # same topic: shown, but doesn't count toward the verdict
FACT_CHECK_MATCH = 0.6   # fact-checks rephrase claims, so slightly looser
WIKI_ALONE = 0.8         # a Wikipedia passage this close can support the claim on its own
ENTITY_COVERAGE = 2 / 3  # share of names/numbers a strong match must contain
MIN_MATCHED_TERMS = 2
MIN_INDEPENDENT = 2      # different publishers needed to support or refute
NEGATION_WINDOW = 2      # a negation counts only up to this many words before a matched term
MAX_EVIDENCE = 12

WEIGHT_NUMBER = 2.0
WEIGHT_NAME = 1.5
WEIGHT_WORD = 1.0

# "without" is left out: in news text it is usually a preposition
# ("leave without a deal"), not a denial of the main statement.
NEGATIONS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor",
             "deny", "denies", "denied", "denying"}
# Words about how something was reported rather than what happened.
REPORTING_WORDS = {"say", "says", "said", "saying", "report", "reports", "reported",
                   "reportedly", "according", "claim", "claims", "claimed", "tell",
                   "tells", "told", "new", "news", "just", "watch", "breaking",
                   "video", "latest", "live", "update", "exclusive", "via"}
STOP_WORDS = (ENGLISH_STOP_WORDS | REPORTING_WORDS) - NEGATIONS

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

# Searched by both engines; the NLI engine searches Wikipedia and fact-checks itself.
NEWS_FETCHERS = {sources.NEWS_ARCHIVE: sources.search_news_archive, sources.NEWS: sources.search_news}
SOURCE_ORDER = [sources.WIKIPEDIA, sources.NEWS_ARCHIVE, sources.NEWS, sources.FACT_CHECK]


def tokens(text: str) -> list[str]:
    text = (text or "").replace("’", "'").replace("n't", " not").replace("N'T", " NOT")
    return [t[:-2] if t.lower().endswith("'s") else t for t in _TOKEN.findall(text)]


def stem(word: str) -> str:
    """Very light suffix stripping so "passes"/"passed"/"pass" compare equal."""
    w = word.lower()
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if not w.endswith(suffix) or len(w) - len(suffix) < 3:
            continue
        if suffix == "es" and not w.endswith(("ses", "xes", "zes", "ches", "shes")):
            continue
        if suffix == "s" and w.endswith(("ss", "us", "is")):
            continue
        return w[: len(w) - len(suffix)] + replacement
    return w


def parse_claim(text: str) -> dict:
    """
    Key terms of a claim, each with a weight:
      numbers 2.0, names/acronyms 1.5, other content words 1.0.
    Names are capitalised words after the first one, unless the whole
    headline is in Title Case (then capitals say nothing about names).
    """
    words = tokens(normalise(text))
    later = words[1:]
    title_case = bool(later) and sum(w[0].isupper() for w in later) / len(later) > 0.6
    # In an ALL-CAPS headline short capitalised words aren't acronyms.
    shouting = bool(words) and sum(w.isupper() for w in words) / len(words) > 0.5

    terms, entities, originals = {}, set(), {}
    for i, w in enumerate(words):
        low = w.lower()
        is_acronym = not shouting and w.isupper() and w.isalpha() and 2 <= len(w) <= 4
        if low in STOP_WORDS and not is_acronym:
            continue
        has_digit = any(c.isdigit() for c in w)  # keeps "21", "5G", "G7"
        if low in NEGATIONS or (len(w) < 3 and not has_digit and not is_acronym):
            continue
        if has_digit:
            weight = WEIGHT_NUMBER
        elif is_acronym or (i > 0 and w[0].isupper() and not title_case):
            weight = WEIGHT_NAME
        else:
            weight = WEIGHT_WORD
        s = stem(w)
        originals.setdefault(s, w)
        terms[s] = max(terms.get(s, 0.0), weight)
        if weight > WEIGHT_WORD:
            entities.add(s)

    # Search keywords: names and numbers first, then longer (more specific) words.
    ranked = sorted(terms, key=lambda s: (s not in entities, -len(originals[s])))
    return {
        "terms": terms,
        "entities": entities,
        "negated": any(w.lower() in NEGATIONS for w in words),
        "keywords": [originals[s] for s in ranked],
    }


def match(claim: dict, passage: str) -> dict:
    """How well one passage covers the claim's key terms."""
    words = tokens(passage)
    stems = [stem(w) for w in words]
    found = [s for s in claim["terms"] if s in stems]
    total = sum(claim["terms"].values())
    entities = claim["entities"]
    # A negation only counts just before a matched key term ("did not pass",
    # "never left"). A "no" elsewhere ("leave with no deal") negates something else.
    negation_at = [i for i, w in enumerate(words) if w.lower() in NEGATIONS]
    hits = [i for i, s in enumerate(stems) if s in claim["terms"]]
    return {
        "score": round(sum(claim["terms"][s] for s in found) / total, 3) if total else 0.0,
        "n_matched": len(found),
        "entity_coverage": len(entities & set(stems)) / len(entities) if entities else 1.0,
        "negated": any(0 < h - n <= NEGATION_WINDOW for n in negation_at for h in hits),
        # The passage's own words that matched, so the UI can highlight them.
        "matched_words": sorted({w for w in words if stem(w) in claim["terms"]}, key=str.lower),
    }


def stance(claim: dict, m: dict, fact_check_rating: str | None = None) -> str | None:
    """supports / refutes / neutral, or None if the passage is off-topic."""
    strong = m["n_matched"] >= MIN_MATCHED_TERMS and m["entity_coverage"] >= ENTITY_COVERAGE
    same_polarity = m["negated"] == claim["negated"]
    if fact_check_rating is not None:
        if strong and m["score"] >= FACT_CHECK_MATCH:
            rated = rating_to_stance(fact_check_rating)
            # A fact-check rating a claim "False" refutes our claim, unless
            # our claim is the negated version of it.
            if rated == REFUTES:
                return REFUTES if same_polarity else SUPPORTS
            if rated == SUPPORTS:
                return SUPPORTS if same_polarity else REFUTES
            return NEUTRAL
    elif strong and m["score"] >= STRONG_MATCH:
        return SUPPORTS if same_polarity else REFUTES
    return NEUTRAL if m["score"] >= RELATED_MATCH else None


def _trim(text: str, limit: int = 280) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def assess(claim: dict, item: dict) -> dict | None:
    """Score a raw item from core/sources.py using its best-matching passage."""
    scored = [(p, match(claim, p)) for p in item["passages"] if p]
    if not scored:
        return None
    passage, m = max(scored, key=lambda pm: pm[1]["score"])
    is_fact_check = item["source"] == sources.FACT_CHECK
    st = stance(claim, m, (item.get("rating") or "") if is_fact_check else None)
    if st is None:
        return None
    return {
        "source": item["source"],
        "publisher": item["publisher"],
        "title": item["title"],
        "sentence": _trim(passage),
        "url": item["url"],
        "date": item["date"],
        "rating": item.get("rating"),
        "stance": st,
        "score": m["score"],
        "matched": m["matched_words"],
    }


def _publishers(evidence: list[dict], st: str) -> set[str]:
    """
    Publishers whose stance is `st`. Each publisher gets one stance (from its
    closest match), so one site can't count as both supporting and refuting.
    """
    closest = {}
    for e in evidence:
        if e["stance"] == NEUTRAL and st != NEUTRAL:
            continue
        best = closest.get(e["publisher"])
        if best is None or e["score"] > best["score"]:
            closest[e["publisher"]] = e
    return {p for p, e in closest.items() if e["stance"] == st}


def _names(publishers: set[str]) -> str:
    names = sorted(publishers)
    if len(names) > 3:
        return ", ".join(names[:3]) + f" and {len(names) - 3} more"
    return " and ".join([", ".join(names[:-1]), names[-1]]) if len(names) > 1 else names[0]


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def decide(evidence: list[dict]) -> tuple[str, str]:
    """Word-matching verdict: combine individual stances into (verdict, one-sentence reason)."""
    # 1. Published fact-checks of a matching claim outweigh everything else.
    checks = [e for e in evidence if e["source"] == sources.FACT_CHECK and e["stance"] != NEUTRAL]
    if checks:
        false_n = sum(e["stance"] == REFUTES for e in checks)
        true_n = len(checks) - false_n
        if false_n > true_n:
            return REFUTED, f"{_count(false_n, 'published fact-check')} rated a matching claim as false or misleading."
        if true_n > false_n:
            return SUPPORTED, f"{_count(true_n, 'published fact-check')} rated a matching claim as accurate."
        return NOT_ENOUGH, "Published fact-checks of matching claims disagree with each other."

    # 2. Otherwise count independent publishers that closely match.
    support = _publishers(evidence, SUPPORTS)
    against = _publishers(evidence, REFUTES)
    wiki_close = any(
        e["source"] == sources.WIKIPEDIA and e["stance"] == SUPPORTS and e["score"] >= WIKI_ALONE
        for e in evidence
    )
    if (len(support) >= MIN_INDEPENDENT or wiki_close) and len(support) > len(against):
        return SUPPORTED, f"Closely matched by {_names(support)}."
    if len(against) >= MIN_INDEPENDENT and len(against) > len(support):
        return REFUTED, f"{_names(against)} match the wording but state the opposite."
    if support and against:
        return NOT_ENOUGH, f"Sources conflict: {_names(support)} match it, {_names(against)} state the opposite."
    if support:
        return NOT_ENOUGH, (f"Only {_names(support)} closely matches; at least "
                            f"{MIN_INDEPENDENT} independent sources are needed.")
    if against:
        return NOT_ENOUGH, (f"Only {_names(against)} appears to contradict it; at least "
                            f"{MIN_INDEPENDENT} independent sources are needed.")
    related = _publishers(evidence, NEUTRAL)
    if related:
        return NOT_ENOUGH, (f"Related coverage from {_count(len(related), 'source')}, "
                            "but nothing closely matches the claim.")
    return NOT_ENOUGH, "No matching coverage was found in the sources checked."


def default_fetchers(api_key: str | None = None) -> dict:
    fetchers = {sources.WIKIPEDIA: sources.search_wikipedia, **NEWS_FETCHERS}
    if api_key:
        fetchers[sources.FACT_CHECK] = lambda kw: sources.search_fact_checks(kw, api_key)
    return fetchers


def fetch_all(fetchers: dict, keywords: list[str]) -> tuple[list[dict], list[dict]]:
    """Query all sources at the same time; each can fail on its own. Returns (items, statuses)."""
    raw, checked = [], []
    with ThreadPoolExecutor(max_workers=max(1, len(fetchers))) as pool:
        futures = {name: pool.submit(fn, keywords) for name, fn in fetchers.items()}
        for name, future in futures.items():
            status = {"name": name, "ok": True, "skipped": False, "found": 0, "message": ""}
            try:
                items = future.result()
                raw.extend(items)
                status["found"] = len(items)
            except sources.SourceError as e:
                status.update(ok=False, message=str(e))
            except Exception as e:  # a bug in one source must not hide the others
                status.update(ok=False, message=f"failed unexpectedly ({type(e).__name__})")
            checked.append(status)
    return raw, checked


def _check_words(claim: dict, result: dict, api_key: str | None, fetchers: dict | None) -> dict:
    fetchers = fetchers or default_fetchers(api_key)
    raw, checked = fetch_all(fetchers, claim["keywords"])
    if sources.FACT_CHECK not in fetchers:
        checked.append({"name": sources.FACT_CHECK, "ok": False, "skipped": True, "found": 0,
                        "message": "add a free Google Fact Check API key to include published fact-checks"})
    result["sources_checked"] = checked
    if not any(c["ok"] for c in checked):
        return result

    evidence, seen = [], set()
    for item in raw:
        if item.get("url") in seen:
            continue
        seen.add(item.get("url"))
        assessed = assess(claim, item)
        if assessed:
            evidence.append(assessed)
    # Fact-checks first, then close matches, then related coverage.
    evidence.sort(key=lambda e: (e["source"] != sources.FACT_CHECK, e["stance"] == NEUTRAL, -e["score"]))

    result["verdict"], result["summary"] = decide(evidence)
    result["evidence"] = evidence[:MAX_EVIDENCE]
    return result


def _check_nli(claim_text: str, claim: dict, result: dict, api_key: str | None) -> dict:
    from core import nli_evidence

    # News sources are fetched in the background while the NLI pipeline
    # searches fact-checks and Wikipedia; their sentences join the candidates.
    with ThreadPoolExecutor(max_workers=1) as pool:
        news = pool.submit(fetch_all, NEWS_FETCHERS, claim["keywords"])

        def news_candidates() -> list[dict]:
            raw, statuses = news.result()
            result["sources_checked"] += statuses
            return [{"sentence": p, "source": it["source"], "publisher": it["publisher"],
                     "title": it["title"], "url": it["url"], "date": it["date"]}
                    for it in raw for p in it["passages"]]

        nli_evidence.check_claim(claim_text, result, api_key=api_key, more_candidates=news_candidates)

    for e in result["evidence"]:
        e["matched"] = match(claim, e["sentence"])["matched_words"]
    return result


def check_sources(text: str, result: dict, api_key: str | None = None,
                  fetchers: dict | None = None, engine: str | None = None) -> dict:
    """
    Fill a schema result dict (see core/schema.py) with the V2 evidence check.
    engine: "nli", "words" or None (NLI if installed; fake `fetchers` imply "words").
    """
    # The NLI model compares statements: "Is the Earth flat?" -> "the Earth is flat".
    claim_text, changed = question_to_statement(text)
    if changed:
        result["warnings"].append(f'Question rewritten as a statement: "{claim_text}"')
    claim = parse_claim(claim_text)
    result["cleaned"] = normalise(claim_text)
    result["keywords"] = claim["keywords"][:6]
    if len(claim["terms"]) < MIN_MATCHED_TERMS:
        result["error"] = "Not enough distinctive words to search for. Try a more specific headline."
        return result

    engine = engine or os.environ.get("TRUTHLENS_ENGINE") or (
        "nli" if fetchers is None and nlp_models.available() else "words")
    result["engine"] = engine
    if engine == "nli":
        warnings_before = list(result["warnings"])
        try:
            _check_nli(claim_text, claim, result, api_key)
        except Exception as e:  # model download failed, out of memory, ...
            result.update(verdict=None, confidence=None, summary="", evidence=[], sources_checked=[],
                          warnings=warnings_before + [f"The NLI model could not run ({type(e).__name__}); "
                                                      "used word matching instead."])
            result["engine"] = engine = "words"
    if engine == "words":
        _check_words(claim, result, api_key, fetchers)

    checked = result["sources_checked"]
    checked.sort(key=lambda c: SOURCE_ORDER.index(c["name"]) if c["name"] in SOURCE_ORDER else 99)
    failed = [c for c in checked if not c["ok"] and not c["skipped"]]
    if failed and len(failed) == sum(not c["skipped"] for c in checked):
        result.update(verdict=None, confidence=None, evidence=[], error=(
            "Could not reach any online source ("
            + "; ".join(f"{c['name']}: {c['message']}" for c in failed) + ")."))
        return result
    for c in failed:
        message = f"{c['name']} {c['message']}."
        if not any(c["message"] in w for w in result["warnings"]):
            result["warnings"].append(message)
    return result
