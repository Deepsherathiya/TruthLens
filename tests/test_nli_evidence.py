"""
Offline tests for the NLI engine's decision logic (no internet, no model downloads).
The live pipeline is checked with:  python check_demo_claims.py
"""

import pytest

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


def wiki(stance, score, sim):
    return {"stance": stance, "score": score, "similarity": sim}


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
