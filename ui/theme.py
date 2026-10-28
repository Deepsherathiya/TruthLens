"""
Design tokens and global CSS.

The look is editorial (a newspaper rather than a dashboard): serif headlines
(Newsreader), plain sans for interface text (IBM Plex Sans), monospaced
figures (IBM Plex Mono), hairline rules instead of shadows, and one muted
colour per verdict. Streamlit's own widgets are themed in
.streamlit/config.toml; the values there must match LIGHT/DARK below.

Chart colours were checked for colour-blind separation and contrast
against each background (real/fake, the two model colours).
"""

from urllib.parse import quote

import streamlit as st

# Line icons (24x24 SVG paths). st.html() strips inline <svg>, so icons are
# drawn as CSS masks: <i class="tl-ic tl-ic-check"> takes the text colour.
ICON_PATHS = {
    "check": '<path d="M5 12.5l4.2 4.2L19 7"/>',
    "cross": '<path d="M6 6l12 12M18 6L6 18"/>',
    "question": '<path d="M9.1 9a3 3 0 1 1 4.2 2.7c-.8.4-1.3 1.1-1.3 2v.3"/><path d="M12 17.5h.01"/>',
    "dash": '<path d="M6 12h12"/>',
    "tilde": '<path d="M4 13c2-3 4-3 6 0s4 3 6 0 3-2 4-1"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.6h.01"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17h.01"/>',
    "arrow": '<path d="M7 17L17 7M9 7h8v8"/>',
}


def _icon_css() -> str:
    rules = [".tl-ic{display:inline-block;flex:none;width:16px;height:16px;background-color:currentColor;"
             "vertical-align:-3px;-webkit-mask:var(--ic) center/contain no-repeat;mask:var(--ic) center/contain no-repeat;}"
             ".tl-ic-arrow{width:12px;height:12px;vertical-align:-1px;}"]
    for name, paths in ICON_PATHS.items():
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
               f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')
        rules.append(f'.tl-ic-{name}{{--ic:url("data:image/svg+xml,{quote(svg)}");}}')
    return "".join(rules)

LIGHT = {
    "paper": "#F7F5F0", "surface": "#FDFCF9", "surface-2": "#EFEBE2",
    "ink": "#16181D", "ink-2": "#454A53", "ink-3": "#666A72",
    "rule": "#DAD4C6", "rule-strong": "#16181D", "accent": "#1D3A5F",
    "real": "#14805A", "real-bg": "#E3F0EA", "real-ink": "#0F6447",
    "fake": "#C2412D", "fake-bg": "#F8E6E1", "fake-ink": "#A3301E",
    "unc": "#A8741A", "unc-bg": "#F6ECD8", "unc-ink": "#7E560F",
    "neutral": "#5B6472", "neutral-bg": "#E8EAED", "neutral-ink": "#434A56",
    "mark": "#F2E2A0",
    "lr": "#2563A8", "nb": "#8F8A7E",
    "seq-lo": "#EDF1F6", "seq-hi": "#1D3A5F", "on-seq-hi": "#FFFFFF",
}

DARK = {
    "paper": "#15171B", "surface": "#1B1E23", "surface-2": "#22262C",
    "ink": "#ECE8DF", "ink-2": "#C2BEB5", "ink-3": "#9A968E",
    "rule": "#30343B", "rule-strong": "#ECE8DF", "accent": "#9DB8DA",
    "real": "#2F9D74", "real-bg": "#16302A", "real-ink": "#74D1AA",
    "fake": "#E2674F", "fake-bg": "#3A201B", "fake-ink": "#F2957F",
    "unc": "#C9962F", "unc-bg": "#352A15", "unc-ink": "#E4BC66",
    "neutral": "#8A93A3", "neutral-bg": "#252A31", "neutral-ink": "#BAC2CE",
    "mark": "#544A1F",
    "lr": "#4A86D4", "nb": "#7C786F",
    "seq-lo": "#1F2630", "seq-hi": "#6F9BD1", "on-seq-hi": "#0E1116",
}

SERIF = "'Newsreader', Georgia, 'Times New Roman', serif"
SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace"

# Italics and optical sizes for the serif (theme config can't request them).
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@"
    "0,6..72,400..700;1,6..72,400..600&display=swap');"
)


def tokens() -> dict:
    """Colour tokens for the theme the viewer is currently using."""
    try:
        is_dark = st.context.theme.type == "dark"
    except Exception:  # older Streamlit or no browser context (tests)
        is_dark = False
    return DARK if is_dark else LIGHT


_CSS = """
/* ---------- page frame ---------- */
[data-testid="stMainBlockContainer"] { max-width: 1140px; padding-top: 5.5rem; padding-bottom: 3rem; }
[data-testid="stHeader"] { background: var(--tl-paper); border-bottom: 1px solid var(--tl-rule); }
.stTextArea textarea {
  font-family: var(--tl-serif); font-size: 1.3rem; line-height: 1.4;
  font-variation-settings: "opsz" 24;
}
.stTextArea textarea::placeholder { font-style: italic; }
h1, h2, h3 { letter-spacing: -0.012em; }
mark { background: var(--tl-mark); color: inherit; padding: 0 .12em; border-radius: 2px; }
a.tl-link, .tl-clip a { color: var(--tl-accent); text-underline-offset: 2px; }
@media (prefers-reduced-motion: no-preference) {
  .tl-needle { transition: left .35s ease-out; }
}

/* ---------- type ---------- */
.tl-kicker {
  font: 600 .7rem/1.2 var(--tl-sans); letter-spacing: .14em; text-transform: uppercase;
  color: var(--tl-ink-3); margin: 0;
}
.tl-serif { font-family: var(--tl-serif); }
.tl-num { font-family: var(--tl-mono); font-variant-numeric: tabular-nums; }

/* ---------- masthead & page headers ---------- */
.tl-mast { margin: 0 0 1.6rem; }
.tl-mast-top {
  display: flex; justify-content: space-between; gap: .5rem 1.5rem; flex-wrap: wrap;
  font: 500 .7rem/1 var(--tl-sans); letter-spacing: .14em; text-transform: uppercase;
  color: var(--tl-ink-3); padding-bottom: .65rem; border-bottom: 1px solid var(--tl-rule);
}
.tl-wordmark {
  font: 600 clamp(2.9rem, 7vw, 4.75rem)/0.95 var(--tl-serif); letter-spacing: -.025em;
  color: var(--tl-ink); margin: 1.1rem 0 .7rem; padding: 0;
  font-variation-settings: "opsz" 72;
}
.tl-wordmark em { font-style: italic; font-weight: 400; }
.tl-double-rule { border-top: 1px solid var(--tl-rule-strong); border-bottom: 1px solid var(--tl-rule-strong); height: 4px; }
.tl-mast-dek {
  font: 400 1.12rem/1.5 var(--tl-serif); color: var(--tl-ink-2); max-width: 44rem; margin: .9rem 0 0;
}
.tl-page-head { margin: 0 0 1.75rem; padding-bottom: 1.1rem; border-bottom: 1px solid var(--tl-rule-strong); }
.tl-page-head h1 {
  font: 600 clamp(2rem, 4.2vw, 2.85rem)/1.05 var(--tl-serif); letter-spacing: -.02em;
  color: var(--tl-ink); margin: .5rem 0 .55rem; padding: 0; font-variation-settings: "opsz" 60;
}
.tl-page-dek { font: 400 1.05rem/1.55 var(--tl-serif); color: var(--tl-ink-2); max-width: 46rem; margin: 0; }

/* ---------- section headings ---------- */
.tl-section {
  display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  border-top: 1px solid var(--tl-rule-strong); padding-top: .6rem; margin: 1.6rem 0 .2rem;
}
.tl-section h2 { font: 600 1.3rem/1.25 var(--tl-serif); color: var(--tl-ink); margin: 0; padding: 0; }
.tl-section p { font: .85rem/1.4 var(--tl-sans); color: var(--tl-ink-3); margin: 0; }

/* ---------- aside: how to read ---------- */
.tl-aside { border-left: 1px solid var(--tl-rule); padding-left: 1.4rem; }
.tl-aside ol { list-style: none; padding: 0 !important; margin: .7rem 0 0; counter-reset: step; }
.tl-aside li {
  counter-increment: step; display: grid; grid-template-columns: 2.1rem 1fr;
  padding: .75rem 0; border-top: 1px solid var(--tl-rule);
}
.tl-aside li::before {
  content: counter(step, decimal-leading-zero); font: 500 .78rem/1.6 var(--tl-mono); color: var(--tl-ink-3);
}
.tl-aside h3 { font: 600 1.05rem/1.3 var(--tl-serif); color: var(--tl-ink); margin: 0 0 .2rem; padding: 0; }
.tl-aside p, .tl-fine { font: .86rem/1.5 var(--tl-sans); color: var(--tl-ink-2); margin: 0; }
.tl-fine { color: var(--tl-ink-3); font-size: .8rem; }
@media (max-width: 640px) { .tl-aside { border-left: 0; padding-left: 0; } }

/* ---------- verdict cards ---------- */
.tone-real    { --tone: var(--tl-real);    --tone-bg: var(--tl-real-bg);    --tone-ink: var(--tl-real-ink); }
.tone-fake    { --tone: var(--tl-fake);    --tone-bg: var(--tl-fake-bg);    --tone-ink: var(--tl-fake-ink); }
.tone-unc     { --tone: var(--tl-unc);     --tone-bg: var(--tl-unc-bg);     --tone-ink: var(--tl-unc-ink); }
.tone-neutral { --tone: var(--tl-neutral); --tone-bg: var(--tl-neutral-bg); --tone-ink: var(--tl-neutral-ink); }
.tone-off     { --tone: var(--tl-rule);    --tone-bg: var(--tl-surface-2);  --tone-ink: var(--tl-ink-3); }

.tl-card {
  background: var(--tl-surface); border: 1px solid var(--tl-rule); border-top: 3px solid var(--tone);
  border-radius: 4px; padding: 1.15rem 1.35rem 1.25rem; height: 100%; box-sizing: border-box;
}
.tl-verdict { display: flex; align-items: center; gap: .65rem; margin: .7rem 0 .45rem; }
.tl-verdict-icon {
  width: 2.1rem; height: 2.1rem; border-radius: 50%; flex: none; display: grid; place-items: center;
  background: var(--tone-bg); color: var(--tone-ink);
}
.tl-verdict-word {
  font: 600 2rem/1.05 var(--tl-serif); letter-spacing: -.015em; color: var(--tone-ink); margin: 0; padding: 0;
}
.tl-dek { font: .95rem/1.5 var(--tl-sans); color: var(--tl-ink-2); margin: 0 0 1rem; }
.tl-figs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid var(--tl-rule); margin: 1.05rem 0 0; }
.tl-fig { padding: .7rem .6rem 0 0; min-width: 0; }
.tl-fig + .tl-fig { padding-left: .85rem; border-left: 1px solid var(--tl-rule); }
.tl-fig dt { font: 600 .66rem/1.2 var(--tl-sans); letter-spacing: .12em; text-transform: uppercase; color: var(--tl-ink-3); }
.tl-fig dd { font: 500 1.3rem/1.25 var(--tl-mono); font-variant-numeric: tabular-nums; color: var(--tl-ink); margin: .3rem 0 0; }

/* probability meter: real zone | uncertain zone | fake zone, with a needle */
.tl-meter { margin: .3rem 0 0; }
.tl-meter-track { position: relative; display: flex; gap: 2px; height: 9px; }
.tl-zone { height: 100%; }
.tl-zone.z-real { background: color-mix(in srgb, var(--tl-real) 32%, transparent); border-radius: 5px 0 0 5px; }
.tl-zone.z-unc  { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--tl-unc) 34%, transparent) 0 3px, transparent 3px 6px); }
.tl-zone.z-fake { background: color-mix(in srgb, var(--tl-fake) 32%, transparent); border-radius: 0 5px 5px 0; }
.tl-needle {
  position: absolute; top: -6px; width: 3px; height: 21px; border-radius: 2px; transform: translateX(-50%);
  background: var(--tl-ink); box-shadow: 0 0 0 2px var(--tl-surface);
}
.tl-meter-scale {
  display: flex; justify-content: space-between; margin-top: .55rem;
  font: 600 .64rem/1 var(--tl-sans); letter-spacing: .12em; text-transform: uppercase; color: var(--tl-ink-3);
}

/* source status + keywords */
.tl-sources { list-style: none; padding: 0 !important; margin: .9rem 0 0 !important; border-top: 1px solid var(--tl-rule); }
.tl-sources li {
  display: grid; grid-template-columns: 1.1rem 7.6rem 1fr; gap: .5rem; align-items: baseline;
  padding: .42rem 0; border-bottom: 1px dotted var(--tl-rule); font: .84rem/1.4 var(--tl-sans); color: var(--tl-ink);
}
.tl-sources li span:last-child { color: var(--tl-ink-3); }
.tl-sources .ok .tl-ic { color: var(--tl-real); }
.tl-sources .fail .tl-ic { color: var(--tl-unc); }
.tl-sources .skip .tl-ic { color: var(--tl-ink-3); }
.tl-keys { display: flex; flex-wrap: wrap; align-items: center; gap: .35rem; margin-top: .85rem; }
.tl-keys code {
  font: 500 .76rem/1 var(--tl-mono); color: var(--tl-ink-2); background: transparent;
  border: 1px solid var(--tl-rule); border-radius: 3px; padding: .28rem .4rem;
}

/* ---------- headline being reported on ---------- */
.tl-quote {
  font: 500 clamp(1.35rem, 2.6vw, 1.75rem)/1.3 var(--tl-serif); color: var(--tl-ink);
  margin: .45rem 0 .35rem; font-variation-settings: "opsz" 36;
}
.tl-quote::before { content: "\\201C"; } .tl-quote::after { content: "\\201D"; }
.tl-cleaned { font: .82rem/1.5 var(--tl-mono); color: var(--tl-ink-3); margin: 0; overflow-wrap: anywhere; }

/* ---------- clippings (evidence list) ---------- */
.tl-clips { list-style: none; padding: 0 !important; margin: .6rem 0 0 !important; }
.tl-clip {
  display: grid; grid-template-columns: 9.5rem minmax(0, 1fr); gap: .25rem 1.4rem;
  padding: 1rem 0 1.05rem; border-bottom: 1px solid var(--tl-rule);
}
.tl-clip:first-child { border-top: 1px solid var(--tl-rule); }
.tl-clip-meta { display: flex; flex-direction: column; gap: .3rem; font: .8rem/1.3 var(--tl-sans); color: var(--tl-ink-3); min-width: 0; }
.tl-clip-meta .tl-src { font: 600 .66rem/1.2 var(--tl-sans); letter-spacing: .12em; text-transform: uppercase; color: var(--tl-ink-2); }
.tl-clip-meta .tl-pub { color: var(--tl-ink); font-weight: 500; overflow-wrap: anywhere; }
.tl-clip-text { font: 400 1.06rem/1.5 var(--tl-serif); color: var(--tl-ink); margin: .35rem 0 .45rem; }
.tl-clip-foot { display: flex; flex-wrap: wrap; gap: .3rem 1rem; font: .8rem/1.4 var(--tl-sans); color: var(--tl-ink-3); }
.tl-stance {
  display: inline-flex; align-items: center; gap: .3rem; font: 600 .7rem/1 var(--tl-sans);
  letter-spacing: .08em; text-transform: uppercase; padding: .3rem .5rem .3rem .4rem; border-radius: 3px;
  background: var(--tone-bg); color: var(--tone-ink);
}
.tl-rating { font: 600 .78rem/1 var(--tl-sans); color: var(--tone-ink); }
@media (max-width: 640px) { .tl-clip { grid-template-columns: 1fr; } .tl-clip-meta { flex-direction: row; flex-wrap: wrap; gap: .5rem; } }

/* ---------- stat row ---------- */
.tl-stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr));
  border-top: 1px solid var(--tl-rule-strong); border-bottom: 1px solid var(--tl-rule); margin: .4rem 0 .6rem;
}
.tl-stat { padding: .85rem 1rem .9rem 0; min-width: 0; }
.tl-stat + .tl-stat { padding-left: 1rem; border-left: 1px solid var(--tl-rule); }
.tl-stat-value {
  font: 500 2rem/1.1 var(--tl-serif); font-variant-numeric: lining-nums tabular-nums;
  color: var(--tl-ink); margin: .35rem 0 .2rem; font-variation-settings: "opsz" 48;
}
.tl-stat-note { font: .78rem/1.35 var(--tl-sans); color: var(--tl-ink-3); }

/* ---------- notes, empty states, footers ---------- */
.tl-note {
  display: flex; gap: .6rem; align-items: flex-start; padding: .7rem .9rem; margin: .5rem 0;
  border-left: 3px solid var(--tone, var(--tl-ink-3)); background: var(--tl-surface-2);
  border-radius: 0 4px 4px 0; font: .88rem/1.5 var(--tl-sans); color: var(--tl-ink-2);
}
.tl-note .tl-ic { margin-top: .15rem; color: var(--tone-ink, var(--tl-ink-3)); }
.tl-empty {
  border: 1px dashed color-mix(in srgb, var(--tl-ink) 28%, transparent); border-radius: 4px;
  padding: 1.5rem 1.6rem; margin: .5rem 0;
}
.tl-empty h3 { font: 600 1.2rem/1.3 var(--tl-serif); color: var(--tl-ink); margin: 0 0 .35rem; padding: 0; }
.tl-empty p { font: .9rem/1.55 var(--tl-sans); color: var(--tl-ink-2); margin: 0; max-width: 40rem; }
.tl-empty code, .tl-prose code {
  font: .82rem var(--tl-mono); background: var(--tl-surface-2); border: 1px solid var(--tl-rule);
  border-radius: 3px; padding: .05rem .3rem; color: var(--tl-ink);
}
.tl-foot {
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: .4rem 1.5rem; margin-top: 2.5rem;
  padding-top: .75rem; border-top: 1px solid var(--tl-rule); font: .76rem/1.5 var(--tl-sans); color: var(--tl-ink-3);
}

/* ---------- term tables with inline bars ---------- */
.tl-terms { width: 100%; border-collapse: collapse; font: .9rem/1.3 var(--tl-sans); }
.tl-terms caption { text-align: left; padding: 0 0 .5rem; }
.tl-terms th {
  text-align: left; font: 600 .66rem/1.2 var(--tl-sans); letter-spacing: .12em; text-transform: uppercase;
  color: var(--tl-ink-3); padding: .45rem 0; border-bottom: 1px solid var(--tl-rule-strong);
}
.tl-terms td { padding: .38rem 0; border-bottom: 1px solid var(--tl-rule); color: var(--tl-ink); vertical-align: middle; }
.tl-terms td.term { font-family: var(--tl-mono); font-size: .84rem; padding-right: .75rem; overflow-wrap: anywhere; }
.tl-terms td.weight { width: 52%; }
.tl-bar { display: flex; align-items: center; gap: .55rem; }
.tl-bar i { display: block; height: 7px; width: var(--w); min-width: 2px; background: var(--tone); border-radius: 0 3px 3px 0; }
.tl-bar span { font: .78rem var(--tl-mono); color: var(--tl-ink-2); font-variant-numeric: tabular-nums; }

/* ---------- batch distribution ---------- */
.tl-dist-bar { display: flex; gap: 2px; height: 12px; margin: .8rem 0 .7rem; }
.tl-dist-bar span:first-child { border-radius: 4px 0 0 4px; } .tl-dist-bar span:last-child { border-radius: 0 4px 4px 0; }
.tl-dist-bar .real { background: var(--tl-real); } .tl-dist-bar .fake { background: var(--tl-fake); }
.tl-dist-bar .unc { background: var(--tl-unc); } .tl-dist-bar .err { background: var(--tl-rule); }
.tl-legend { display: flex; flex-wrap: wrap; gap: .4rem 1.4rem; list-style: none; padding: 0 !important; margin: 0; font: .84rem/1.3 var(--tl-sans); color: var(--tl-ink-2); }
.tl-legend i { display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; margin-right: .4rem; vertical-align: -1px; }
.tl-legend b { font-family: var(--tl-mono); font-weight: 500; color: var(--tl-ink); }

/* ---------- long-form prose (Method page) ---------- */
.tl-prose { max-width: 44rem; }
.tl-prose h2 { font: 600 1.45rem/1.25 var(--tl-serif); color: var(--tl-ink); margin: 2rem 0 .5rem; padding: 0; }
.tl-prose h3 { font: 600 1.1rem/1.3 var(--tl-serif); color: var(--tl-ink); margin: 1.3rem 0 .3rem; padding: 0; }
.tl-prose p, .tl-prose li { font: 1.02rem/1.65 var(--tl-serif); color: var(--tl-ink-2); }
.tl-prose ol, .tl-prose ul { padding-left: 1.2rem; }
.tl-prose li { margin: .25rem 0; }
.tl-prose strong { color: var(--tl-ink); font-weight: 600; }
"""


def apply() -> dict:
    """Inject fonts, tokens and CSS for the current theme. Returns the tokens."""
    t = tokens()
    variables = "".join(f"--tl-{k}:{v};" for k, v in t.items())
    variables += f"--tl-serif:{SERIF};--tl-sans:{SANS};--tl-mono:{MONO};"
    st.html(f"<style>{_FONT_IMPORT}:root{{{variables}}}{_CSS}{_icon_css()}</style>")
    return t
