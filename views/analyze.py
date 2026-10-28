"""Home page: enter a headline, see both signals, keep a session history."""

import streamlit as st

from core.preprocessing import MAX_INPUT_CHARS, validate_input
from core.schema import DISCLAIMER, EVIDENCE_DISCLAIMER
from ui import components as ui
from ui import state


def _use_example():
    pick = st.session_state.get("example")
    if pick:
        st.session_state.headline = state.EXAMPLES[pick]
        st.session_state.example = None
        st.session_state.input_error = None


def _clear():
    st.session_state.headline = ""
    st.session_state.last = None
    st.session_state.input_error = None


def _clear_history():
    st.session_state.history = []


def _input_panel(pipeline) -> None:
    st.text_area(
        "Headline", key="headline", height=112, max_chars=MAX_INPUT_CHARS,
        placeholder="Paste a news headline, e.g. EU leaders agree to start next phase of Brexit talks",
    )
    if st.session_state.get("input_error"):
        ui.note(ui.esc(st.session_state.input_error), "tone-unc", "alert")

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        run = st.button("Analyze headline", type="primary", icon=":material/arrow_forward:",
                        icon_position="right")
        st.button("Clear", type="tertiary", on_click=_clear)
        st.space("stretch")
        st.toggle("Cross-check online sources", key="online",
                  help="Sends the headline's key words to Wikipedia, GDELT and (if you add a key) "
                       "Google Fact Check. Turn off to keep everything on this computer.")

    st.pills("Try an example", list(state.EXAMPLES), key="example", on_change=_use_example,
             wrap=True)

    if run:
        text = st.session_state.headline
        error = validate_input(text)
        st.session_state.input_error = error
        if error:
            st.rerun()
        style = state.run_style(text, pipeline)
        sources = None
        if st.session_state.online:
            with st.spinner("Searching Wikipedia, its news archive and recent news…"):
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
    <li><div><h3>Source check</h3>
      <p>The headline's key words are searched in Wikipedia, its daily news archive and
      recent world news. Close matches from independent publishers corroborate the claim.</p></div></li>
  </ol>
  <p class="tl-fine">Neither signal proves a story true or false. Style is a pattern, not a fact;
  missing coverage is not a debunk.</p>
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

    if sources and not sources["error"]:
        evidence = sources["evidence"]
        ui.section("What the sources say",
                   f"{len(evidence)} passage{'s' if len(evidence) != 1 else ''} · matched words highlighted")
        if evidence:
            ui.show(ui.clippings(evidence))
        else:
            ui.empty_state("No matching coverage",
                           "None of the sources returned a passage that shares enough key words with "
                           "the headline. That is common for very recent, very local or invented stories.")

    if style:
        st.page_link(state.PAGES["insights"], label="See which words drove the model",
                     icon=":material/arrow_forward:")
    ui.show(f'<p class="tl-fine">{ui.esc(DISCLAIMER)} {ui.esc(EVIDENCE_DISCLAIMER)}</p>')


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
        ui.empty_state("No headline analysed yet",
                       "Paste a headline above, or pick one of the examples, then press "
                       "<strong>Analyze headline</strong>.")
    _history()
    ui.footer()
