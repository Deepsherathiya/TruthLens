"""Home page: enter a headline or claim, see both signals, keep a session history."""

import streamlit as st

from check_demo_claims import DEMO_CLAIMS
from core import nlp_models
from core.preprocessing import MAX_INPUT_CHARS, validate_input
from core.schema import DISCLAIMER, EVIDENCE_DISCLAIMER, NEUTRAL, REFUTED, REFUTES, SUPPORTED, SUPPORTS
from ui import components as ui
from ui import state

NO_PICK = "—"


def _use_example():
    pick = st.session_state.get("example")
    if pick:
        st.session_state.headline = state.EXAMPLES[pick]
        st.session_state.example = None
        st.session_state.input_error = None


def _use_demo_claim():
    pick = st.session_state.get("demo_claim")
    if pick and pick != NO_PICK:
        st.session_state.headline = pick
        st.session_state.demo_claim = NO_PICK
        st.session_state.input_error = None


def _clear():
    st.session_state.headline = ""
    st.session_state.last = None
    st.session_state.input_error = None


def _clear_history():
    st.session_state.history = []


def _input_panel(pipeline) -> None:
    st.text_area(
        "Headline or claim", key="headline", height=112, max_chars=MAX_INPUT_CHARS,
        placeholder="Paste a news headline or a claim, e.g. EU leaders agree to start next phase of Brexit talks",
    )
    if st.session_state.get("input_error"):
        ui.note(ui.esc(st.session_state.input_error), "tone-unc", "alert")

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        run = st.button("Analyze", type="primary", icon=":material/arrow_forward:", icon_position="right")
        st.button("Clear", type="tertiary", on_click=_clear)
        st.space("stretch")
        st.toggle("Check evidence online", key="online",
                  help="Searches published fact-checks, Wikipedia and news for the same claim. "
                       "Turn off to keep everything on this computer.")

    st.pills("Try a headline", list(state.EXAMPLES), key="example", on_change=_use_example, wrap=True)
    st.selectbox("…or a demo claim", [NO_PICK] + [c for c, _ in DEMO_CLAIMS], key="demo_claim",
                 on_change=_use_demo_claim)

    if run:
        text = st.session_state.headline
        error = validate_input(text)
        st.session_state.input_error = error
        if error:
            st.rerun()
        style = state.run_style(text, pipeline)
        sources = None
        if st.session_state.online:
            wait = ("Searching fact-checks, Wikipedia and news… (the first check loads the NLI models, "
                    "about 30 s)" if nlp_models.available() else "Searching fact-checks, Wikipedia and news…")
            with st.spinner(wait):
                sources = state.run_sources(text)
        state.remember(text, style, sources)


def _aside() -> None:
    ui.show("""
<aside class="tl-aside">
  <p class="tl-kicker">How to read the result</p>
  <ol>
    <li><div><h3>Writing style</h3>
      <p>A TF-IDF + logistic regression model compares the wording with ~38,000 real and
      fake headlines from 2016–17 and estimates which it resembles.</p></div></li>
    <li><div><h3>Evidence check</h3>
      <p>Published fact-checks, Wikipedia and news are searched for the claim, and an NLI
      model reads the most relevant sentences to see whether they support or contradict it.</p></div></li>
  </ol>
  <p class="tl-fine">Neither signal is a guarantee. Style is a pattern, not a fact; the evidence
  check can retrieve the wrong article. Always open the sources.</p>
</aside>""")


def _report(last: dict, model_ready: bool) -> None:
    style, sources = last["style"], last["sources"]
    ui.report_header(last["text"], last["time"])

    left, right = st.columns(2, gap="medium")
    with left:
        ui.show(ui.style_card(style, model_ready))
    with right:
        ui.show(ui.sources_card(sources, sources is not None))

    for w in (style or {}).get("warnings", []):
        ui.note(ui.esc(w), "tone-unc", "alert")

    for w in (sources or {}).get("warnings", []):
        ui.note(ui.esc(w), "", "info")

    if sources and not sources["error"]:
        _evidence(sources)

    if style:
        st.page_link(state.PAGES["insights"], label="See which words drove the model",
                     icon=":material/arrow_forward:")
    ui.show(f'<p class="tl-fine">{ui.esc(DISCLAIMER)} {ui.esc(EVIDENCE_DISCLAIMER)}</p>')


def _evidence(sources: dict) -> None:
    evidence, engine = sources["evidence"], sources.get("engine") or "words"
    ui.section("What the evidence says",
               f"{len(evidence)} passage{'s' if len(evidence) != 1 else ''} · most relevant first · "
               "claim words highlighted")
    if not evidence:
        ui.empty_state("No relevant evidence found",
                       "None of the sources returned a sentence close enough to the claim. That is "
                       "common for very recent, very local or invented stories.")
        return
    # Sentences that point against the verdict but were less relevant than the
    # deciding one go in a collapsed section, so they don't look like the main evidence.
    winning = {SUPPORTED: SUPPORTS, REFUTED: REFUTES}.get(sources["verdict"])
    opposing = [e for e in evidence
                if winning and e["stance"] not in (winning, NEUTRAL) and not e.get("deciding")]
    main = [e for e in evidence if e not in opposing]
    ui.show(ui.clippings(main, engine))
    if opposing:
        with st.expander(f"Other related sentences pointing the other way ({len(opposing)}, less relevant)"):
            ui.show('<p class="tl-fine">The NLI model often misreads sentences about a different time or '
                    'topic as contradictions (e.g. a past job vs a current one). These were less relevant '
                    'to the claim than the deciding evidence, so they did not decide the verdict.</p>')
            ui.show(ui.clippings(opposing, engine))


def _history() -> None:
    history = state.history_frame()
    ui.section("This session", "History is kept only while this browser tab is open")
    if history.empty:
        ui.show('<p class="tl-fine">Headlines you analyse will be listed here.</p>')
        return
    st.dataframe(
        history, hide_index=True, width="stretch",
        column_config={
            "Headline": st.column_config.TextColumn(width="large"),
            "P(fake)": st.column_config.ProgressColumn("P(fake)", format="percent", min_value=0.0, max_value=1.0),
        },
    )
    with st.container(horizontal=True, gap="small"):
        st.download_button("Download CSV", history.to_csv(index=False), "truthlens_history.csv", "text/csv",
                           icon=":material/download:", on_click="ignore")
        st.button("Clear history", type="tertiary", on_click=_clear_history, icon=":material/delete:")


def render() -> None:
    pipeline = state.get_pipeline()
    ui.masthead()

    main, aside = st.columns([1.7, 1], gap="large")
    with main:
        _input_panel(pipeline)
    with aside:
        _aside()

    last = st.session_state.last
    if last:
        _report(last, pipeline is not None)
    else:
        ui.empty_state("Nothing analysed yet",
                       "Paste a headline or claim above, or pick an example, then press "
                       "<strong>Analyze</strong>.")
    _history()
    ui.footer()
