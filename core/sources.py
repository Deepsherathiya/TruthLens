"""
Online sources used by evidence mode (V2). Only free APIs:

  Wikipedia articles (MediaWiki API)        no key needed
  News archive: Wikipedia "Current events"  no key needed; daily news summaries
      portal, each item citing the outlet   back to the early 2000s
      that reported it (Reuters, BBC, ...)
  Recent news: GDELT DOC 2.0                no key needed; worldwide news from
                                            the last ~3 months; often busy
  Google Fact Check Tools                   free API key needed; skipped without one

Every fetcher takes the claim's search keywords and returns a list of
"raw evidence" dicts:

    {"source": WIKIPEDIA | NEWS_ARCHIVE | NEWS | FACT_CHECK,
     "publisher": str, "title": str, "url": str, "date": str | None,
     "rating": str | None,        # fact-checker's verdict, e.g. "False"
     "passages": [str]}           # text to compare with the claim

On failure a fetcher raises SourceError with a short reason a user can read.
core/evidence.py catches it, so one source failing never breaks the others.
"""

import html
import os
import re
import threading
import time
from datetime import datetime
from urllib.parse import quote, urlparse

import requests

WIKIPEDIA = "Wikipedia"
NEWS_ARCHIVE = "News archive"
NEWS = "Recent news"
FACT_CHECK = "Fact checks"

USER_AGENT = os.environ.get(
    "TRUTHLENS_USER_AGENT", "TruthLens/2.0 (educational NLP project; python-requests)"
)
TIMEOUT = (5, 20)  # seconds: (connect, read)

WIKI_API = "https://en.wikipedia.org/w/api.php"
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
FACTCHECK_API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# GDELT asks for at most one request every 5 seconds. When busy it is slow
# to answer at all, so it gets a shorter timeout than the other sources.
GDELT_MIN_INTERVAL = 5.5
GDELT_TIMEOUT = (5, 10)
_gdelt_lock = threading.Lock()
_gdelt_last_call = 0.0
_wiki_lock = threading.Lock()

_TAGS = re.compile(r"<[^>]+>")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

# Wikitext markup, for reading Current events pages.
_WIKILINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_EXTLINK = re.compile(r"\[(https?://[^\s\]]+)[^\]]*\]")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_SPACES = re.compile(r"\s+")


class SourceError(Exception):
    pass


def _get_json(url: str, params: dict, timeout=TIMEOUT, retry_wait: float = 0) -> dict:
    """GET a JSON API. With retry_wait > 0, a "too many requests" reply is retried once."""
    try:
        r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    except requests.Timeout:
        raise SourceError("timed out") from None
    except requests.RequestException:
        raise SourceError("could not connect (are you online?)") from None
    if r.status_code == 429 and retry_wait:
        time.sleep(retry_wait)
        return _get_json(url, params, timeout)
    if r.status_code == 429:
        raise SourceError("busy (rate limit); try again in a few seconds")
    if r.status_code in (400, 403):
        raise SourceError("request refused; check the API key")
    if r.status_code != 200:
        raise SourceError(f"returned HTTP {r.status_code}")
    try:
        return r.json()
    except ValueError:
        # GDELT reports some problems as plain text with status 200.
        first_line = r.text.strip().split("\n")[0][:120]
        raise SourceError(first_line or "returned an unreadable response") from None


def _plain(text: str) -> str:
    """Strip HTML tags/entities from API text."""
    return html.unescape(_TAGS.sub("", text or "")).strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text or "") if s.strip()]


def _wiki_get(params: dict) -> dict:
    # One Wikipedia call at a time, as Wikimedia's API etiquette asks. Its
    # rate limit is short-lived, so a refused call is retried once.
    with _wiki_lock:
        return _get_json(WIKI_API, {"format": "json", "formatversion": 2, **params}, retry_wait=2)


def search_wikipedia(keywords: list[str], limit: int = 3) -> list[dict]:
    """Search English Wikipedia and return the intro of the best-matching articles."""
    pages = []
    # Try up to six key terms first; if nothing matches, retry with the top three.
    # generator=search returns the search hits with their content in one call.
    for n in dict.fromkeys([min(len(keywords), 6), 3]):
        data = _wiki_get({
            "action": "query", "generator": "search", "gsrsearch": " ".join(keywords[:n]),
            "gsrlimit": limit, "prop": "extracts|info", "exintro": 1, "explaintext": 1,
            "exlimit": limit, "inprop": "url",
        })
        pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
        if pages:
            break
    items = []
    for page in pages:
        title = page.get("title", "")
        items.append({
            "source": WIKIPEDIA,
            "publisher": "wikipedia.org",
            "title": title,
            "url": page.get("fullurl")
                   or "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
            "date": None,
            "rating": None,
            # The intro states the article's main facts, one sentence per passage.
            "passages": split_sentences(page.get("extract", ""))[:15],
        })
    return items


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _day_page_date(title: str) -> datetime | None:
    # "Portal:Current events/2017 December 2" -> datetime(2017, 12, 2)
    try:
        return datetime.strptime(title.rsplit("/", 1)[-1], "%Y %B %d")
    except ValueError:
        return None


def wikitext_news_items(wikitext: str) -> list[tuple[str, list[str]]]:
    """
    (sentence, cited URLs) for each news item on a Current events page.
    Items look like:
        **The [[United States Senate]] passes the tax bill 51-49. ([https://... Reuters])
    """
    text = _COMMENT.sub("", _REF.sub("", wikitext or ""))
    items = []
    for line in text.splitlines():
        if not line.startswith("*"):
            continue
        urls = _EXTLINK.findall(line)
        plain = _EXTLINK.sub("", line)          # drop the "(Reuters)" citation links
        plain = _WIKILINK.sub(r"\1", plain)     # [[target|label]] -> label
        for _ in range(3):                      # nested {{templates}}
            plain = _TEMPLATE.sub("", plain)
        plain = plain.lstrip("*:# ").replace("'''", "").replace("''", "")
        plain = _SPACES.sub(" ", re.sub(r"\(\s*[,;]?\s*\)", "", plain)).strip()
        if len(plain.split()) >= 6:             # skip topic headings like "* [[Brexit]]"
            items.append((plain, urls))
    return items


def search_news_archive(keywords: list[str], limit: int = 3) -> list[dict]:
    """
    Search Wikipedia's daily "Current events" pages. Each news item there
    cites the outlet that reported it, so the outlet is used as publisher.
    """
    # Monthly overview pages ("…/December 2017") also match, but they only
    # embed the daily pages, so search titles first and keep daily pages.
    days = []
    for n in dict.fromkeys([min(len(keywords), 6), 3]):
        data = _wiki_get({
            "action": "query", "list": "search", "srnamespace": 100,  # 100 = Portal
            "srsearch": " ".join(keywords[:n]) + " prefix:Portal:Current events/",
            "srlimit": 20, "srprop": "",
        })
        days = [(d, h["title"]) for h in data.get("query", {}).get("search", [])
                if (d := _day_page_date(h["title"]))][:limit]
        if days:
            break
    if not days:
        return []

    data = _wiki_get({
        "action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "titles": "|".join(title for _, title in days),
    })
    pages = {p.get("title"): p for p in data.get("query", {}).get("pages", [])}

    items = []
    for day, title in days:
        page = pages.get(title, {})
        page_url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
        revisions = page.get("revisions") or [{}]
        wikitext = revisions[0].get("slots", {}).get("main", {}).get("content", "")
        for sentence, urls in wikitext_news_items(wikitext):
            items.append({
                "source": NEWS_ARCHIVE,
                "publisher": _domain(urls[0]) if urls else "wikipedia.org",
                "title": f"Wikipedia Current events, {day.day} {day:%B %Y}",
                "url": urls[0] if urls else page_url,
                "date": day.strftime("%Y-%m-%d"),
                "rating": None,
                "passages": [sentence],
            })
    return items


def _gdelt_date(seendate: str | None) -> str | None:
    # "20260812T211500Z" -> "2026-08-12"
    if not seendate or len(seendate) < 8:
        return None
    return f"{seendate[:4]}-{seendate[4:6]}-{seendate[6:8]}"


def search_news(keywords: list[str], limit: int = 30) -> list[dict]:
    """Search worldwide English-language news from roughly the last 3 months (GDELT)."""
    global _gdelt_last_call
    terms = [k for k in keywords if len(k) >= 3][:4]  # GDELT rejects very short words
    if len(terms) < 2:
        raise SourceError("needs at least two distinctive words to search news")
    query = " ".join(terms) + " sourcelang:english"

    with _gdelt_lock:
        wait = GDELT_MIN_INTERVAL - (time.monotonic() - _gdelt_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            data = _get_json(GDELT_API, {
                "query": query, "mode": "ArtList", "maxrecords": limit,
                "format": "json", "sort": "HybridRel",
            }, timeout=GDELT_TIMEOUT)
        finally:
            _gdelt_last_call = time.monotonic()

    items = []
    for a in data.get("articles", []):
        title = _plain(a.get("title", ""))
        if not title:
            continue
        items.append({
            "source": NEWS,
            "publisher": a.get("domain", ""),
            "title": title,
            "url": a.get("url"),
            "date": _gdelt_date(a.get("seendate")),
            "rating": None,
            "passages": [title],
        })
    return items


def search_fact_checks(keywords: list[str], api_key: str, limit: int = 10) -> list[dict]:
    """Search published fact-checks (ClaimReview) via Google Fact Check Tools."""
    data = _get_json(FACTCHECK_API, {
        "query": " ".join(keywords[:6]), "key": api_key,
        "languageCode": "en", "pageSize": limit,
    })
    items = []
    for claim in data.get("claims", []):
        reviews = claim.get("claimReview") or []
        if not reviews:
            continue
        review = reviews[0]
        publisher = review.get("publisher", {})
        items.append({
            "source": FACT_CHECK,
            "publisher": publisher.get("site") or publisher.get("name") or "fact-checker",
            "title": _plain(review.get("title") or claim.get("text", "")),
            "url": review.get("url"),
            "date": (review.get("reviewDate") or claim.get("claimDate") or "")[:10] or None,
            "rating": review.get("textualRating"),
            "passages": [_plain(claim.get("text", ""))],
        })
    return items
