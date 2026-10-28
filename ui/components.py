"""
Small HTML building blocks rendered with st.html().

Everything that comes from the user or an online source is escaped with
esc(), and links are only rendered for http(s) URLs, so a headline or a news
title can never inject markup into the page.
"""

import html
import re
from datetime import date, datetime

import streamlit as st

from core.schema import (
    FAKE, NEUTRAL, NOT_ENOUGH, REAL, REFUTED, REFUTES, SUPPORTED, SUPPORTS, UNCERTAIN,
)
from core.sources import FACT_CHECK
from core.style_model import UNCERTAIN_BAND
from ui.theme import ICON_PATHS


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def safe_url(url) -> str | None:
    return url if isinstance(url, str) and re.match(r"^https?://", url) else None


def show(markup: str) -> None:
    st.html(markup)


# Line icons, drawn by CSS (see ui/theme.py). Decorative: the text next to
# each icon always says the same thing, so colour/shape is never the only cue.
ICON = {name: f'<i class="tl-ic tl-ic-{name}" aria-hidden="true"></i>' for name in ICON_PATHS}

VERDICT_STYLE = {
    REAL: ("tone-real", "check"),
    FAKE: ("tone-fake", "cross"),
    UNCERTAIN: ("tone-unc", "question"),
    SUPPORTED: ("tone-real", "check"),
    REFUTED: ("tone-fake", "cross"),
    NOT_ENOUGH: ("tone-neutral", "question"),
}

STANCE_STYLE = {SUPPORTS: ("tone-real", "check"), REFUTES: ("tone-fake", "cross"), NEUTRAL: ("tone-neutral", "tilde")}
# What each stance means depends on which engine judged it.
STANCE_LABEL = {
    "nli": {SUPPORTS: "Supports", REFUTES: "Contradicts", NEUTRAL: "Neutral"},
    "words": {SUPPORTS: "Close match", REFUTES: "Opposite wording", NEUTRAL: "Related"},
}
ENGINE_LABEL = {
    "nli": "NLI model (DeBERTa-v3) reading each sentence",
    "words": "word matching (install PyTorch for the NLI model)",
}


def pct(p: float | None, digits: int = 1) -> str:
    return "–" if p is None else f"{p * 100:.{digits}f}%"


def nice_date(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        d = datetime.strptime(iso[:10], "%Y-%m-%d")
    except ValueError:
        return esc(iso)
    return f"{d.day} {d:%b %Y}"


# ---------------------------------------------------------------- headers

def masthead() -> None:
    today = date.today()
    show(f"""
<header class="tl-mast">
  <div class="tl-mast-top">
    <span>NLP-powered news analysis</span>
    <span>{today:%A}, {today.day} {today:%B %Y}</span>
    <span>Edition 2 &middot; Style model + evidence check</span>
  </div>
  <h1 class="tl-wordmark">Truth<em>Lens</em></h1>
  <div class="tl-double-rule"></div>
  <p class="tl-mast-dek">Read a headline two ways: how its <em>wording</em> compares with
  thousands of real and fake headlines, and whether <em>independent sources</em>
  report the same thing.</p>
</header>""")


def page_header(kicker: str, title: str, dek: str = "") -> None:
    show(f"""
<header class="tl-page-head">
  <p class="tl-kicker">{esc(kicker)}</p>
  <h1>{esc(title)}</h1>
  {f'<p class="tl-page-dek">{dek}</p>' if dek else ''}
</header>""")


def section(title: str, note: str = "") -> None:
    show(f'<div class="tl-section"><h2>{esc(title)}</h2>{f"<p>{note}</p>" if note else ""}</div>')


def note(text: str, tone: str = "", icon: str = "info") -> None:
    """A short callout. `text` may contain trusted inline HTML; escape user text first."""
    show(f'<div class="tl-note {tone}">{ICON[icon]}<div>{text}</div></div>')


def empty_state(title: str, body: str) -> None:
    show(f'<div class="tl-empty"><h3>{esc(title)}</h3><p>{body}</p></div>')


def footer(extra: str = "") -> None:
    show(f"""
<footer class="tl-foot">
  <span>TruthLens &middot; B.Tech NLP lab project &middot; the evidence check calls free public APIs</span>
  <span>{extra}</span>
</footer>""")


# ---------------------------------------------------------------- verdict cards

def _verdict_head(kicker: str, verdict: str) -> tuple[str, str]:
    """(header HTML, tone class) for a verdict card."""
    tone, icon = VERDICT_STYLE.get(verdict, ("tone-neutral", "question"))
    return (f'<p class="tl-kicker">{esc(kicker)}</p>'
            f'<div class="tl-verdict"><span class="tl-verdict-icon">{ICON[icon]}</span>'
            f'<h2 class="tl-verdict-word">{esc(verdict)}</h2></div>'), tone


def meter(p_fake: float) -> str:
    lo, hi = UNCERTAIN_BAND
    label = f"Model-estimated probability that the wording is fake-like: {pct(p_fake)}"
    return f"""
<div class="tl-meter" role="img" aria-label="{esc(label)}">
  <div class="tl-meter-track">
    <span class="tl-zone z-real" style="width:{lo * 100:.0f}%"></span>
    <span class="tl-zone z-unc" style="width:{(hi - lo) * 100:.0f}%"></span>
    <span class="tl-zone z-fake" style="width:{(1 - hi) * 100:.0f}%"></span>
    <span class="tl-needle" style="left:{p_fake * 100:.1f}%"></span>
  </div>
  <div class="tl-meter-scale"><span>Real-like</span><span>Uncertain</span><span>Fake-like</span></div>
</div>"""


STYLE_DEK = {
    REAL: "The wording resembles the <strong>real</strong> news headlines the model learned from.",
    FAKE: "The wording resembles the <strong>fake</strong> news headlines the model learned from.",
    UNCERTAIN: "The model can't confidently place this wording on either side.",
}


def style_card(r: dict | None, model_ready: bool) -> str:
    if not model_ready:
        return """
<section class="tl-card tone-off">
  <p class="tl-kicker">Writing-style model</p>
  <div class="tl-verdict"><span class="tl-verdict-icon">""" + ICON["dash"] + """</span>
  <h2 class="tl-verdict-word">Not trained yet</h2></div>
  <p class="tl-dek">Put <code>Fake.csv</code> and <code>True.csv</code> in <code>data/</code>, then run
  <code>python train_model.py</code> and reload this page.</p>
</section>"""
    head, tone = _verdict_head("Writing-style model", r["verdict"])
    probs = r["probabilities"]
    known = r["stats"].get("known_word_ratio")
    return f"""
<section class="tl-card {tone}">
  {head}
  <p class="tl-dek">{STYLE_DEK[r['verdict']]}</p>
  {meter(probs[FAKE])}
  <dl class="tl-figs">
    <div class="tl-fig"><dt>P(fake)</dt><dd>{pct(probs[FAKE])}</dd></div>
    <div class="tl-fig"><dt>P(real)</dt><dd>{pct(probs[REAL])}</dd></div>
    <div class="tl-fig"><dt>Words known</dt><dd>{pct(known, 0)}</dd></div>
  </dl>
</section>"""


def _source_status(checked: list[dict]) -> str:
    rows = []
    for c in checked:
        if c["ok"]:
            cls, icon, msg = "ok", "check", f"{c['found']} result{'s' if c['found'] != 1 else ''}"
        elif c["skipped"]:
            cls, icon, msg = "skip", "dash", "not configured"
        else:
            cls, icon, msg = "fail", "alert", c["message"]
        rows.append(f'<li class="{cls}">{ICON[icon]}<span>{esc(c["name"])}</span><span>{esc(msg)}</span></li>')
    return f'<ul class="tl-sources">{"".join(rows)}</ul>'


def sources_card(r: dict | None, online: bool) -> str:
    if not online or r is None:
        return """
<section class="tl-card tone-off">
  <p class="tl-kicker">Evidence check</p>
  <div class="tl-verdict"><span class="tl-verdict-icon">""" + ICON["dash"] + """</span>
  <h2 class="tl-verdict-word">Switched off</h2></div>
  <p class="tl-dek">Turn on <strong>Check evidence online</strong> to search fact-checks,
  Wikipedia and news for the same claim.</p>
</section>"""
    if r["error"]:
        return f"""
<section class="tl-card tone-off">
  <p class="tl-kicker">Evidence check</p>
  <div class="tl-verdict"><span class="tl-verdict-icon">{ICON['alert']}</span>
  <h2 class="tl-verdict-word">Unavailable</h2></div>
  <p class="tl-dek">{esc(r['error'])}</p>
  {_source_status(r['sources_checked'])}
</section>"""
    head, tone = _verdict_head("Evidence check", r["verdict"])
    keys = "".join(f"<code>{esc(k)}</code>" for k in r["keywords"])
    confidence = (f" Model-estimated confidence of the deciding evidence: {pct(r['confidence'], 0)}."
                  if r.get("engine") == "nli" and r["verdict"] != NOT_ENOUGH and r.get("confidence") else "")
    return f"""
<section class="tl-card {tone}">
  {head}
  <p class="tl-dek">{esc(r['summary'])}{esc(confidence)}</p>
  {_source_status(r['sources_checked'])}
  <div class="tl-keys"><span class="tl-kicker">Judged by</span><span class="tl-fine">{esc(ENGINE_LABEL.get(r.get('engine'), ''))}</span></div>
  <div class="tl-keys"><span class="tl-kicker">Searched for</span>{keys}</div>
</section>"""


def report_header(text: str, when: str) -> None:
    show(f"""
<div class="tl-section"><h2>The report</h2><p>Analysed at {esc(when)}</p></div>
<p class="tl-quote">{esc(text.strip())}</p>""")


# ---------------------------------------------------------------- evidence

def highlight(text: str, words: list[str]) -> str:
    """Escape text and wrap the given words in <mark>."""
    if not words:
        return esc(text)
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b")
    parts = pattern.split(text)
    # re.split with one group alternates: plain, match, plain, match, ...
    return "".join(f"<mark>{esc(p)}</mark>" if i % 2 else esc(p) for i, p in enumerate(parts))


def _measure(e: dict, engine: str) -> str:
    """How strongly this item was judged, in words that fit the engine."""
    if engine != "nli":
        return f"Contains {pct(e['score'], 0)} of the claim's key terms"
    if e["source"] == FACT_CHECK:
        return f"Same claim (NLI entailment {pct(e['score'], 0)})"
    return f"NLI {pct(e['score'], 0)} · relevance {e.get('similarity', 0):.2f}"


def clippings(evidence: list[dict], engine: str = "words") -> str:
    items = []
    for e in evidence:
        tone, icon = STANCE_STYLE[e["stance"]]
        label = STANCE_LABEL.get(engine, STANCE_LABEL["words"])[e["stance"]]
        if e["source"] == FACT_CHECK and e["stance"] != NEUTRAL:
            label = "Rated accurate" if e["stance"] == SUPPORTS else "Rated false"
        deciding = '<span class="tl-deciding">Decided the verdict</span>' if e.get("deciding") else ""
        rating = (f'<span class="tl-rating">Rating: &ldquo;{esc(e["rating"])}&rdquo;</span>'
                  if e.get("rating") else "")
        url = safe_url(e.get("url"))
        link = (f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(e["title"])} {ICON["arrow"]}</a>'
                if url else esc(e["title"]))
        date_html = f"<time>{nice_date(e.get('date'))}</time>" if e.get("date") else ""
        items.append(f"""
<li class="tl-clip {tone}{' is-deciding' if e.get('deciding') else ''}">
  <div class="tl-clip-meta"><span class="tl-src">{esc(e['source'])}</span>
    <span class="tl-pub">{esc(e['publisher'])}</span>{date_html}</div>
  <div>
    <span class="tl-stance">{ICON[icon]}{label}</span>{deciding}
    <p class="tl-clip-text">{highlight(e['sentence'], e.get('matched', []))}</p>
    <div class="tl-clip-foot">{link}<span>{esc(_measure(e, engine))}</span>{rating}</div>
  </div>
</li>""")
    return f'<ol class="tl-clips">{"".join(items)}</ol>'


# ---------------------------------------------------------------- stats & tables

def stat_row(stats: list[tuple[str, str, str]]) -> None:
    """stats: (label, value, note) triples."""
    cells = "".join(
        f'<div class="tl-stat"><p class="tl-kicker">{esc(label)}</p>'
        f'<div class="tl-stat-value">{esc(value)}</div><div class="tl-stat-note">{esc(sub)}</div></div>'
        for label, value, sub in stats
    )
    show(f'<div class="tl-stats">{cells}</div>')


def term_table(rows: list[tuple[str, float]], tone: str, caption: str = "",
               value_format: str = "{:+.2f}", header: tuple[str, str] = ("Term", "Weight")) -> str:
    """Terms with an inline bar scaled to the largest absolute value."""
    if not rows:
        return '<p class="tl-fine">Nothing to show.</p>'
    biggest = max(abs(v) for _, v in rows) or 1
    body = "".join(
        f'<tr><td class="term">{esc(term)}</td><td class="weight"><div class="tl-bar">'
        f'<i style="--w:{abs(v) / biggest * 100:.0f}%"></i><span>{value_format.format(v)}</span></div></td></tr>'
        for term, v in rows
    )
    cap = f'<caption><span class="tl-kicker">{caption}</span></caption>' if caption else ""
    return (f'<table class="tl-terms {tone}">{cap}<thead><tr><th>{esc(header[0])}</th>'
            f'<th>{esc(header[1])}</th></tr></thead><tbody>{body}</tbody></table>')


def distribution(counts: dict[str, int]) -> None:
    """Stacked bar + legend for batch verdict counts."""
    parts = [(REAL, "real", "real"), (FAKE, "fake", "fake"), (UNCERTAIN, "unc", "unc"), ("Skipped", "err", "rule")]
    total = sum(counts.get(k, 0) for k, _, _ in parts) or 1
    segs = "".join(f'<span class="{cls}" style="flex:{counts.get(k, 0)}"></span>'
                   for k, cls, _ in parts if counts.get(k, 0))
    legend = "".join(
        f'<li><i style="background:var(--tl-{token})"></i>'
        f'{esc(k)} <b>{counts.get(k, 0):,}</b> &middot; {counts.get(k, 0) / total:.0%}</li>'
        for k, _, token in parts if counts.get(k, 0)
    )
    show(f'<div class="tl-dist"><div class="tl-dist-bar" role="img" aria-label="Verdict distribution">'
         f'{segs}</div><ul class="tl-legend">{legend}</ul></div>')
