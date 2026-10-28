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
_PREFIX = re.compile(
    r"^\s*(" + "|".join(re.escape(t) for t in _PREFIX_TAGS) + r")\s*[:\-–—]\s*",
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
_ACRONYM = re.compile(r"\b((?:[a-z]\.){2,})")


def clean_title(text: str) -> str:
    """Normalise a headline. Returns "" for empty/None input."""
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
    # Lowercase removes the ALL-CAPS signal (common in fake titles only).
    # Negations like "not"/"never" are kept: no stop-word removal.
    text = _WHITESPACE.sub(" ", text).strip().lower()
    text = _ACRONYM.sub(lambda m: m.group(1).replace(".", ""), text)
    return text


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


def validate_input(text: str) -> str | None:
    """Return an error message for bad user input, or None if it's fine."""
    if text is None or not str(text).strip():
        return "Please enter a headline."
    if len(text) > MAX_INPUT_CHARS:
        return f"Headline is too long ({len(text)} characters). Max is {MAX_INPUT_CHARS}."
    if not clean_title(text):
        return "Nothing left to analyse after cleaning. Try a real headline."
    return None
