"""
Offline tests for the NLI engine's decision logic (no internet, no model downloads).
The live pipeline is checked with:  python check_demo_claims.py
"""

import pytest

from core import nli_evidence, retrieval
from core.nli_evidence import decide, keyword_queries, rating_to_stance
from core.nlp_models import normalize
from core.preprocessing import question_to_statement
from core.schema import NOT_ENOUGH, REFUTED, SUPPORTED


@pytest.mark.parametrize("rating, stance", [
    ("False", "refutes"), ("FALSE", "refutes"), ("Mostly False", "refutes"),
    ("This is not true. 5G does not spread viruses", "refutes"), ("Pants on Fire!", "refutes"),
    ("Misleading", "refutes"), ("True", "supports"), ("Mostly true", "supports"),
    ("Half True", "neutral"), ("Missing context", "neutral"), ("", "neutral"),
])
def test_rating_to_stance(rating, stance):
    assert rating_to_stance(rating) == stance


@pytest.mark.parametrize("question, statement", [
    ("Is the Earth flat?", "the Earth is flat"),
    ("Is the Great Wall of China visible from the Moon?", "the Great Wall of China is visible from the Moon"),
    ("Was the Taj Mahal built by Shah Jahan?", "the Taj Mahal was built by Shah Jahan"),
    ("Is coffee bad for health?", "coffee is bad for health"),
    ("Did Einstein win the Nobel Prize?", "Einstein win the Nobel Prize"),
])
def test_question_to_statement(question, statement):
    assert question_to_statement(question) == (statement, True)


def test_statements_are_left_alone():
    assert question_to_statement("The Earth is flat") == ("The Earth is flat", False)


def test_keyword_queries_drop_stopwords_and_leave_one_out():
    qs = keyword_queries("The new 2000 rupee notes contain a GPS nano chip")
    assert qs[0] == "2000 rupee notes contain GPS nano chip"
    assert "2000 rupee notes contain nano chip" in qs


def test_normalize_numbers_and_quotes():
    assert normalize("India’s 2,000 notes") == "India's 2000 notes"


def wiki(stance, score, sim, on_topic=True):
    return {"stance": stance, "score": score, "similarity": sim, "on_topic": on_topic}


def fc(stance, score=0.9):
    return {"stance": stance, "score": score}


def test_factchecks_decide_first():
    verdict, conf, _ = decide([fc("refutes")], [wiki("supports", 0.99, 0.9)])
    assert verdict == REFUTED and conf == 0.9


def test_most_relevant_confident_sentence_decides():
    # The off-topic contradiction is less similar, so the supporting sentence wins.
    items = [wiki("refutes", 0.99, 0.70), wiki("supports", 0.95, 0.80), wiki("neutral", 0.9, 0.85)]
    verdict, conf, note = decide([], items)
    assert verdict == SUPPORTED and conf == 0.95 and note is None
    assert [bool(e.get("deciding")) for e in items] == [False, True, False]


def test_close_conflict_adds_warning():
    verdict, _, note = decide([], [wiki("supports", 0.99, 0.79), wiki("refutes", 0.99, 0.76)])
    assert verdict == SUPPORTED and note


def test_irrelevant_evidence_is_not_enough():
    verdict, _, _ = decide([], [wiki("refutes", 0.99, 0.40), wiki("neutral", 0.9, 0.7)])
    assert verdict == NOT_ENOUGH


def test_conflicting_factchecks_fall_back_to_wikipedia():
    verdict, _, _ = decide([fc("refutes"), fc("supports")], [wiki("supports", 0.9, 0.8)])
    assert verdict == SUPPORTED


def test_off_topic_sentence_cannot_decide():
    items = [wiki("supports", 0.92, 0.80, on_topic=False), wiki("refutes", 0.95, 0.70)]
    verdict, _, _ = decide([], items)
    assert verdict == REFUTED and items[1]["deciding"]


@pytest.mark.parametrize("claim", ["narendra modi resigned", "Narendra Modi resigned"])
def test_sentence_about_someone_else_is_off_topic(monkeypatch, claim):
    # Real case: the NLI model gave this sentence 92% entailment for the claim.
    sentence = ("12 November 2019: Heavy Industries and Public Enterprises Minister Arvind Sawant "
                "of Shiv Sena resigned from the cabinet after Shiv Sena's withdrawal from NDA.")
    monkeypatch.setattr(nli_evidence, "rank_by_similarity", lambda claim, texts, k: [(0, 0.56)])
    monkeypatch.setattr(nli_evidence, "nli", lambda premises, hypothesis: [
        {"entailment": 0.92, "contradiction": 0.01, "neutral": 0.07}])
    items = nli_evidence.sentence_evidence(claim, [{"sentence": sentence}])
    assert items[0]["stance"] == "supports" and not items[0]["on_topic"]
    assert decide([], items)[0] == NOT_ENOUGH


def test_sentence_about_the_claim_is_on_topic(monkeypatch):
    sentence = "The Great Wall of China cannot be seen by the naked human eye from the Moon."
    monkeypatch.setattr(nli_evidence, "rank_by_similarity", lambda claim, texts, k: [(0, 0.84)])
    monkeypatch.setattr(nli_evidence, "nli", lambda premises, hypothesis: [
        {"entailment": 0.0, "contradiction": 0.99, "neutral": 0.01}])
    items = nli_evidence.sentence_evidence(
        "The Great Wall of China is visible from the Moon with the naked eye", [{"sentence": sentence}])
    assert items[0]["on_topic"] and decide([], items)[0] == REFUTED


@pytest.mark.parametrize("title, categories, fact", [
    ("Narendra Modi", ("Category:Living people", "Category:Prime ministers of India"),
     ("Narendra Modi is alive.", "Living people")),
    ("Ratan Tata", ("Category:1937 births", "Category:2024 deaths"), ("Ratan Tata died in 2024.", "2024 deaths")),
    ("Michael Jordan (footballer)", ("Category:Living people",),
     ("Michael Jordan is alive.", "Living people")),
    ("Great Wall of China", ("Category:Walls in China",), None),
])
def test_life_fact_from_categories(title, categories, fact):
    assert retrieval.life_fact(title, categories) == fact


def test_wikipedia_adds_life_fact_for_people(monkeypatch):
    monkeypatch.setattr(retrieval, "_wiki_search", lambda q: ("Narendra Modi",))
    monkeypatch.setattr(retrieval, "_wiki_categories",
                        lambda titles: (("Narendra Modi", ("Category:Living people",)),))
    monkeypatch.setattr(retrieval, "_wiki_text", lambda t: "Narendra Modi is an Indian politician and statesman.")
    sentences, error = retrieval.search_wikipedia("narendra modi suicides")
    assert error is None
    assert sentences[0]["sentence"] == "Narendra Modi is alive."
    assert "Living people" in sentences[0]["title"]


def test_best_known_person_checks_the_title(monkeypatch):
    # Sorted by incoming links; "Manmohan Singh" matched "intitle:kohli" through a redirect.
    monkeypatch.setattr(retrieval, "_wiki_best_known",
                        lambda q: ("Manmohan Singh", "Virat Kohli", "Rahul Dravid", "Kohli family"))
    monkeypatch.setattr(retrieval, "_wiki_categories", lambda titles: tuple(
        (t, ("Category:Living people",) if t != "Kohli family" else ()) for t in titles))
    assert retrieval.best_known_person(("kohli",)) == "Virat Kohli"
    assert retrieval.best_known_person(("rahul",)) is None  # one word must be the surname
    assert retrieval.best_known_person(("virat", "kohli")) == "Virat Kohli"


def test_death_claim_about_a_surname_uses_the_full_name(monkeypatch):
    # Regression: "modi died" was Supported by "Krishan Kumar Modi died in 2019."
    from core import evidence
    monkeypatch.setattr(retrieval, "best_known_person",
                        lambda name: "Narendra Modi" if name in {("modi",), ("narendra", "modi")} else None)
    assert evidence.name_in_full("modi died") == ("Narendra Modi died", "Narendra Modi")
    assert evidence.name_in_full("Narendra Modi died") == ("Narendra Modi died", None)
    assert evidence.name_in_full("Modi visits Japan") == ("Modi visits Japan", None)  # not a death claim
