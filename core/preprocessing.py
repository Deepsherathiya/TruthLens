"""
Text cleaning shared by training and prediction.

The same clean_title() runs on the dataset and on user input, so the model
never sees text processed differently at inference time.

Besides normal tidying, it removes "collection artifacts" of the ISOT dataset:
patterns that tell the model which *website* a headline came from rather than
whether it is true (e.g. "[VIDEO]" tags appear almost only in fake titles,
"Factbox:" prefixes only in Reuters titles). Leaving them in would let the
model score very high accuracy by spotting the source instead of the content.
"""

import re
import unicodedata

MAX_INPUT_CHARS = 300

# Anything in brackets: [VIDEO], (TWEETS), (IMAGES), (DEC 28) ...
_BRACKETS = re.compile(r"[\(\[][^\)\]]*[\)\]]")

# Leading "TAG:" labels used by one source or the other.
_PREFIX_TAGS = [
    "breaking news", "breaking", "watch", "just in", "report", "flashback",
    "busted", "bombshell", "sunday screening", "update", "revealed",
    "not kidding", "video", "wow", "factbox", "exclusive", "highlights",
    "timeline", "instant view", "explainer", "breakingviews",
]
# The tag must be followed by a colon ("WATCH: ...") or a spaced dash
# ("WATCH - ..."), so ordinary words like "Watch-list" are left alone.
_PREFIX = re.compile(
    r"^\s*(" + "|".join(re.escape(t) for t in _PREFIX_TAGS) + r")(?:\s*:|\s+-)\s*",
    re.IGNORECASE,
)

# Curly quotes/dashes -> plain ASCII, so encoding differences between the
# two source files can't become a feature.
_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "\xa0": " ", "​": "",
})

_WHITESPACE = re.compile(r"\s+")

# "u.s." -> "us", "u.n." -> "un": Reuters writes "U.S.", fake sites write "US".
_ACRONYM = re.compile(r"\b((?:[a-zA-Z]\.){2,})")

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def normalise(text: str) -> str:
    """
    Remove source artifacts and tidy whitespace, but keep the original case.
    Evidence mode needs the case to spot names ("Paris", "NATO").
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_PUNCT_MAP)
    text = _BRACKETS.sub(" ", text)
    # Strip stacked prefixes such as "BREAKING: WATCH: ..."
    while True:
        stripped = _PREFIX.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    text = _WHITESPACE.sub(" ", text).strip()
    return _ACRONYM.sub(lambda m: m.group(1).replace(".", ""), text)


def clean_title(text: str) -> str:
    """Normalise a headline for the model. Returns "" for empty/None input."""
    # Lowercase removes the ALL-CAPS signal (common in fake titles only).
    # Negations like "not"/"never" are kept: no stop-word removal.
    return normalise(text).lower()


_BE_VERBS = {"is", "are", "was", "were"}
_DO_VERBS = {"does", "do", "did"}
_DETERMINERS = {"the", "a", "an"}


def question_to_statement(text: str) -> tuple[str, bool]:
    """
    Turn simple yes/no questions into statements, because the NLI model
    compares statements:  "Is the Earth flat?" -> "the Earth is flat".
    Returns (text, changed). Anything it can't handle is returned unchanged.
    """
    t = text.strip()
    words = t.rstrip("?").split()
    if not t.endswith("?") or len(words) < 3:
        return t, False
    aux, rest = words[0].lower(), words[1:]
    if aux in _DO_VERBS:
        # "Did Einstein win the Nobel Prize?" -> "Einstein win the Nobel Prize"
        # (grammar isn't perfect, but the NLI model reads it fine)
        return " ".join(rest), True
    if aux in _BE_VERBS:
        # Subject = optional determiner + run of Capitalised words (allowing "of"),
        # or just the first word: "Is the Great Wall of China visible ..." ->
        # "the Great Wall of China is visible ..."
        i = 1 if rest[0].lower() in _DETERMINERS else 0
        j = i
        while j < len(rest) and (rest[j][:1].isupper() or (rest[j] == "of" and j + 1 < len(rest) and rest[j + 1][:1].isupper())):
            j += 1
        end = max(j, i + 1)
        if end >= len(rest):
            return t, False
        return " ".join(rest[:end] + [aux] + rest[end:]), True
    return t, False


def validate_input(text) -> str | None:
    """Return an error message for bad user input, or None if it's fine."""
    if not isinstance(text, str) or not text.strip():
        return "Please enter a headline."
    if len(text) > MAX_INPUT_CHARS:
        return f"Headline is too long ({len(text)} characters). Max is {MAX_INPUT_CHARS}."
    if not clean_title(text):
        return "Nothing left to analyse after cleaning. Try a real headline."
    return None


def text_stats(text: str) -> dict:
    """
    Simple descriptive statistics of the raw headline.
    These describe writing style only; they are not evidence of truth.
    """
    text = text.strip()
    words = _WORD.findall(text)
    return {
        "words": len(words),
        "characters": len(text),
        "avg_word_length": round(sum(map(len, words)) / len(words), 1) if words else 0.0,
        # Shouting in capitals is a style cue the model can no longer see
        # (clean_title lowercases), but it is useful to point out. Two-letter
        # words are skipped so "UK"/"EU" don't count as shouting.
        "all_caps_words": sum(1 for w in words if len(w) > 2 and w.isupper() and w.isalpha()),
        "exclamations": text.count("!"),
        "questions": text.count("?"),
    }
