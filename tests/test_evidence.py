"""
Tests for the evidence check. No network and no model downloads: sources are
replaced by small fake fetchers, and the NLI models by fake functions.
"""

from core import evidence, sources
from core.analyze import analyze
from core.schema import NEUTRAL, NOT_ENOUGH, REFUTED, REFUTES, SUPPORTED, SUPPORTS


def item(text, publisher="example.com", source=sources.NEWS, rating=None, url=None):
    return {"source": source, "publisher": publisher, "title": text[:40], "url": url or f"https://{publisher}/{hash(text)}",
            "date": "2017-12-02", "rating": rating, "passages": [text]}


def fetcher(*items):
    return lambda keywords: list(items)


def failing(message):
    def fetch(keywords):
        raise sources.SourceError(message)
    return fetch


# ---------------------------------------------------------------- claim parsing

def test_parse_claim_weights_names_and_numbers():
    c = evidence.parse_claim("BREAKING: Senate passes $1.5 trillion tax bill, Trump says")
    assert c["terms"]["trump"] == evidence.WEIGHT_NAME
    assert c["terms"]["1"] == evidence.WEIGHT_NUMBER
    assert "say" not in c["terms"] and "breaking" not in c["terms"]  # reporting words dropped
    assert c["keywords"][0] in ("1", "5", "Trump")  # names/numbers searched first


def test_title_case_headline_has_no_guessed_names():
    c = evidence.parse_claim("Hillary Just Destroyed This GOP Moron On Live TV")
    assert c["entities"] == {"gop", "tv"}  # only acronyms; capitals mean nothing in Title Case


def test_negated_claim():
    assert evidence.parse_claim("Senate did not pass the tax bill")["negated"]
    assert not evidence.parse_claim("Senate passes the tax bill")["negated"]


def test_stem():
    assert evidence.stem("passes") == evidence.stem("passed") == evidence.stem("pass") == "pass"
    assert evidence.stem("Paris") == "paris" and evidence.stem("US") == "us"


# ---------------------------------------------------------------- matching & stance

def test_negation_only_counts_before_a_matched_word():
    claim = evidence.parse_claim("UK votes to leave the European Union")
    assert not evidence.match(claim, "MPs vote to leave the European Union with no deal")["negated"]
    assert evidence.match(claim, "The UK did not vote to leave the European Union")["negated"]


def test_short_claim_needs_every_key_word():
    # Regression: passages about Nazi-occupied Paris used to "corroborate" this.
    claim = evidence.parse_claim("Paris is the capital of Germany")
    m = evidence.match(claim, "Paris had been occupied by Nazi Germany since 1940.")
    assert evidence.stance(claim, m) == NEUTRAL


def test_stances():
    claim = evidence.parse_claim("Senate passes tax bill")
    assert evidence.stance(claim, evidence.match(claim, "The Senate passed the tax bill 51-49.")) == SUPPORTS
    assert evidence.stance(claim, evidence.match(claim, "The Senate did not pass the tax bill.")) == REFUTES
    assert evidence.stance(claim, evidence.match(claim, "Weather is sunny in Paris.")) is None




# ---------------------------------------------------------------- verdicts

def _ev(stance, publisher, source=sources.NEWS, score=0.9):
    return {"stance": stance, "publisher": publisher, "source": source, "score": score}


def test_two_independent_publishers_corroborate():
    assert evidence.decide([_ev(SUPPORTS, "bbc.com"), _ev(SUPPORTS, "reuters.com")])[0] == SUPPORTED


def test_one_publisher_is_not_enough():
    verdict, reason = evidence.decide([_ev(SUPPORTS, "bbc.com"), _ev(SUPPORTS, "bbc.com")])
    assert verdict == NOT_ENOUGH and "bbc.com" in reason


def test_one_publisher_cannot_count_both_ways():
    ev = [_ev(SUPPORTS, "bbc.com", score=0.9), _ev(REFUTES, "bbc.com", score=0.8), _ev(SUPPORTS, "cnn.com")]
    assert evidence.decide(ev)[0] == SUPPORTED


def test_close_wikipedia_passage_can_corroborate_alone():
    assert evidence.decide([_ev(SUPPORTS, "wikipedia.org", sources.WIKIPEDIA, 0.85)])[0] == SUPPORTED
    assert evidence.decide([_ev(SUPPORTS, "wikipedia.org", sources.WIKIPEDIA, 0.76)])[0] == NOT_ENOUGH


def test_fact_checks_decide_first():
    ev = [_ev(SUPPORTS, "bbc.com"), _ev(SUPPORTS, "cnn.com"), _ev(REFUTES, "snopes.com", sources.FACT_CHECK)]
    assert evidence.decide(ev)[0] == REFUTED


# ---------------------------------------------------------------- full check (fake sources)

def test_check_sources_end_to_end():
    fetchers = {
        sources.WIKIPEDIA: fetcher(item("The Senate passed the tax bill on 2 December 2017.", "wikipedia.org",
                                        sources.WIKIPEDIA)),
        sources.NEWS_ARCHIVE: fetcher(item("The United States Senate passes the tax reform bill 51-49.", "reuters.com")),
        sources.NEWS: failing("busy (rate limit); try again in a few seconds"),
    }
    r = analyze("Senate passes tax bill", mode="evidence", fetchers=fetchers)
    assert r["error"] is None and r["verdict"] == SUPPORTED
    assert {e["publisher"] for e in r["evidence"]} == {"wikipedia.org", "reuters.com"}
    assert any("busy" in w for w in r["warnings"])            # failed source is reported...
    status = {c["name"]: c for c in r["sources_checked"]}
    assert status[sources.NEWS]["ok"] is False                 # ...not hidden
    assert status[sources.FACT_CHECK]["skipped"] is True       # no API key given
    assert r["evidence"][0]["matched"]                         # words to highlight


def test_fact_check_rating_disputes():
    fetchers = {sources.FACT_CHECK: fetcher(item("Pope Francis endorses Donald Trump for president", "politifact.com",
                                                 sources.FACT_CHECK, rating="Pants on Fire"))}
    r = analyze("Pope Francis endorses Donald Trump for president", mode="evidence", fetchers=fetchers)
    assert r["verdict"] == REFUTED and r["evidence"][0]["stance"] == REFUTES


def test_all_sources_failing_is_an_error_not_a_verdict():
    fetchers = {sources.WIKIPEDIA: failing("timed out"), sources.NEWS: failing("timed out")}
    r = analyze("Senate passes tax bill", mode="evidence", fetchers=fetchers)
    assert r["verdict"] is None and "Could not reach" in r["error"]


def test_too_vague_to_search():
    r = analyze("This is it", mode="evidence", fetchers={})
    assert r["error"] and r["verdict"] is None


def test_wikitext_news_items():
    text = ("*[[Brexit]]\n"
            "**The [[United Kingdom|UK]] votes to leave the [[European Union]]. "
            "([https://www.bbc.com/news/x BBC]) ([http://reuters.com/y Reuters])\n"
            "<!-- comment -->")
    [(sentence, urls)] = sources.wikitext_news_items(text)
    assert sentence == "The UK votes to leave the European Union."
    assert urls == ["https://www.bbc.com/news/x", "http://reuters.com/y"]


# ---------------------------------------------------------------- engines

def test_questions_are_rewritten_before_searching():
    fetchers = {sources.WIKIPEDIA: fetcher(item("The Senate passed the tax bill.", "wikipedia.org", sources.WIKIPEDIA))}
    r = analyze("Did the Senate pass the tax bill?", mode="evidence", fetchers=fetchers)
    assert any("Question rewritten" in w for w in r["warnings"])
    assert r["engine"] == "words"  # fake fetchers always use word matching


def test_nli_engine_combines_fact_checks_wikipedia_and_news(monkeypatch):
    from core import nli_evidence, retrieval
    monkeypatch.setattr(retrieval, "search_factchecks",
                        lambda claim, api_key=None: ([], "No Fact Check API key set in .env, so ..."))
    monkeypatch.setattr(retrieval, "search_wikipedia", lambda claim: ([
        {"sentence": "The tax bill was debated in the Senate for weeks.", "title": "Tax Cuts and Jobs Act",
         "url": "https://en.wikipedia.org/wiki/Tax_Cuts_and_Jobs_Act"}], None))
    monkeypatch.setattr(evidence, "NEWS_FETCHERS", {
        sources.NEWS_ARCHIVE: fetcher(item("The United States Senate passes the tax bill 51-49.", "reuters.com",
                                           sources.NEWS_ARCHIVE)),
        sources.NEWS: failing("busy (rate limit); try again in a few seconds"),
    })
    # Fake models: the news sentence is the most similar and is entailed; the other is neutral.
    monkeypatch.setattr(nli_evidence, "rank_by_similarity",
                        lambda q, texts, k: sorted([(i, 0.9 if "passes" in t else 0.7) for i, t in enumerate(texts)],
                                                   key=lambda x: -x[1]))
    monkeypatch.setattr(nli_evidence, "nli", lambda premises, hyp: [
        {"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04} if "passes" in p else
        {"entailment": 0.10, "contradiction": 0.05, "neutral": 0.85} for p in premises])

    r = analyze("Senate passes tax bill", mode="evidence", engine="nli")
    assert r["error"] is None and r["engine"] == "nli" and r["verdict"] == SUPPORTED
    top = r["evidence"][0]
    assert top["deciding"] and top["publisher"] == "reuters.com" and top["matched"]
    assert [c["name"] for c in r["sources_checked"]] == evidence.SOURCE_ORDER
    status = {c["name"]: c for c in r["sources_checked"]}
    assert status[sources.FACT_CHECK]["skipped"] and not status[sources.NEWS]["ok"]
    assert any("busy" in w for w in r["warnings"]) and "reuters.com" in r["summary"]


def test_nli_failure_falls_back_to_word_matching(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("could not download the model")
    monkeypatch.setattr(evidence, "_check_nli", broken)
    monkeypatch.setattr(evidence, "default_fetchers", lambda api_key=None: {
        sources.WIKIPEDIA: fetcher(item("The Senate passed the tax bill on 2 December 2017.", "wikipedia.org",
                                        sources.WIKIPEDIA))})
    r = analyze("Senate passes tax bill", mode="evidence", engine="nli")
    assert r["engine"] == "words" and r["verdict"] == SUPPORTED
    assert any("used word matching instead" in w for w in r["warnings"])


def test_busy_news_source_is_skipped_for_a_while(monkeypatch):
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise sources.SourceError("timed out")
    monkeypatch.setattr(sources, "_get_json", unavailable)
    monkeypatch.setattr(sources, "_gdelt_last_call", 0.0)
    monkeypatch.setattr(sources, "_gdelt_down_until", 0.0)
    for _ in range(2):
        try:
            sources.search_news(["Senate", "tax", "bill"])
        except sources.SourceError as e:
            message = str(e)
    assert len(calls) == 1 and message.startswith("skipped")  # second call didn't wait on GDELT
