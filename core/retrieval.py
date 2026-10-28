"""
V2 evidence retrieval from two free sources:

1. Google Fact Check Tools API  - existing fact-checks by PolitiFact, Alt News, BOOM, ...
   Needs GOOGLE_FACTCHECK_API_KEY in .env.
2. Wikipedia (public MediaWiki API) - no key needed. We search with the claim,
   take the top articles, and split their text into sentences.

Both functions return plain lists of dicts and never raise on network errors;
they return an "error" string instead so the app can keep going with the other source.
"""

import os
import re
from functools import lru_cache

import requests

# core.sources reads .env and holds the shared User-Agent and Wikipedia lock.
from core.sources import USER_AGENT, WIKI_LOCK

TIMEOUT = 10
# Wikimedia asks API users to identify themselves with a descriptive User-Agent.
HEADERS = {"User-Agent": USER_AGENT}

FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
WIKI_API = "https://en.wikipedia.org/w/api.php"

WIKI_PAGES = 3
MAX_SENTENCES_PER_PAGE = 400
# Everything after these headings is lists of links, not prose.
_STOP_SECTIONS = re.compile(r"^==+\s*(See also|References|Notes|External links|Further reading|Bibliography|Sources)\s*==+", re.M)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def factcheck_key() -> str:
    return os.getenv("GOOGLE_FACTCHECK_API_KEY", "").strip()


@lru_cache(maxsize=256)
def _factcheck_raw(query: str, key: str) -> tuple:
    params = {"query": query, "languageCode": "en", "pageSize": 10, "key": key}
    r = requests.get(FACTCHECK_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return tuple(r.json().get("claims", []))


def search_factchecks(claim: str, api_key: str | None = None) -> tuple[list[dict], str | None]:
    """Return (reviews, error). One dict per published review of a matching claim."""
    key = api_key or factcheck_key()
    if not key:
        return [], "No Fact Check API key set in .env, so existing fact-checks were not searched."
    try:
        claims = _factcheck_raw(claim, key)
    except requests.RequestException as e:
        # Don't show the URL: it contains the API key.
        return [], f"Fact Check API request failed ({type(e).__name__})."
    reviews = []
    for c in claims:
        for rev in c.get("claimReview", []):
            reviews.append({
                "claim_text": c.get("text", ""),
                "claimant": c.get("claimant", ""),
                "publisher": rev.get("publisher", {}).get("name") or rev.get("publisher", {}).get("site", ""),
                "url": rev.get("url", ""),
                "title": rev.get("title", ""),
                "rating": rev.get("textualRating", ""),
            })
    return reviews, None


@lru_cache(maxsize=256)
def _wiki_search(query: str) -> tuple:
    params = {"action": "query", "list": "search", "srsearch": query,
              "srlimit": WIKI_PAGES, "format": "json"}
    with WIKI_LOCK:  # one Wikipedia request at a time across all sources
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return tuple(x["title"] for x in r.json()["query"]["search"])


@lru_cache(maxsize=256)
def _wiki_text(title: str) -> str:
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "titles": title, "redirects": 1, "format": "json"}
    with WIKI_LOCK:
        r = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "")


def split_sentences(text: str) -> list[str]:
    """Plain-text article -> clean sentences (headings and reference sections removed)."""
    stop = _STOP_SECTIONS.search(text)
    if stop:
        text = text[: stop.start()]
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if not para or para.startswith("="):
            continue
        for s in _SENTENCE_SPLIT.split(para):
            s = s.strip()
            if 40 <= len(s) <= 500:
                sentences.append(s)
    return sentences[:MAX_SENTENCES_PER_PAGE]


def search_wikipedia(claim: str) -> tuple[list[dict], str | None]:
    """Return (sentences, error). One dict per sentence: {sentence, title, url}."""
    try:
        titles = _wiki_search(claim)
        out = []
        for title in titles:
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            for s in split_sentences(_wiki_text(title)):
                out.append({"sentence": s, "title": title, "url": url})
        return out, None
    except (requests.RequestException, KeyError, ValueError) as e:
        return [], f"Wikipedia request failed ({type(e).__name__})."
