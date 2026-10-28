"""Text insights: statistics and the model's word-level explanation for the last headline."""

import streamlit as st

from core.preprocessing import MAX_INPUT_CHARS
from ui import charts, components as ui, state, theme


def render() -> None:
    t = theme.tokens()
    ui.page_header(
        "Text insights", "Inside the last headline",
        "Plain statistics about the wording, and which words moved the writing-style model. "
        "These describe the model's learned associations, not evidence about the story.",
    )
    last = st.session_state.last
    style = (last or {}).get("style")
    if not style:
        ui.empty_state(
            "Nothing to inspect yet",
            "Analyse a headline on the Analyze page first"
            + ("" if state.get_pipeline() else ", after training the model with "
               "<code>python train_model.py</code>") + ".",
        )
        st.page_link(state.PAGES["analyze"], label="Go to Analyze", icon=":material/arrow_back:")
        ui.footer()
        return

    stats = style["stats"]
    ui.show(f"""
<p class="tl-kicker">Headline</p>
<p class="tl-quote">{ui.esc(last['text'].strip())}</p>
<p class="tl-cleaned">What the model read after cleaning: {ui.esc(style['cleaned'])}</p>""")

    ui.stat_row([
        ("Words", f"{stats['words']}", "tokens of letters/digits"),
        ("Characters", f"{stats['characters']}", f"of {MAX_INPUT_CHARS} allowed"),
        ("Avg. word length", f"{stats['avg_word_length']:.1f}", "letters per word"),
        ("All-caps words", f"{stats['all_caps_words']}", "3+ letters, e.g. DESTROYED; the model sees lowercase"),
        ("Known to model", ui.pct(stats.get("known_word_ratio"), 0), "of single words seen in training"),
    ])

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        ui.section("What pushed the prediction", "tf-idf value × model coefficient")
        contribs = style["word_contributions"]
        if contribs:
            st.altair_chart(charts.contributions(contribs, t), width="stretch", theme=None)
            ui.show('<p class="tl-fine">Each bar is one word or two-word phrase that appears in the headline '
                    'and in the model\'s vocabulary. Words the model never saw have no effect and are '
                    'not shown. Positive values push toward <strong>Fake</strong>, negative toward '
                    '<strong>Real</strong>.</p>')
        else:
            ui.empty_state("No known words", "None of the headline's words were in the training vocabulary.")
    with right:
        ui.section("Most distinctive terms", "highest tf-idf weight")
        rows = [(r["term"], r["weight"]) for r in style["tfidf_terms"]]
        ui.show(ui.term_table(rows, "tone-neutral", value_format="{:.3f}", header=("Term", "TF-IDF")))
        ui.show('<p class="tl-fine" style="margin-top:.6rem">TF-IDF is high for words that are in this '
                'headline but rare across the training headlines. It says what is distinctive, not which '
                'way it points.</p>')

    sources = last.get("sources")
    if sources and sources.get("keywords"):
        ui.section("Key terms used for the evidence search", "names and numbers weigh more")
        ui.show('<div class="tl-keys">' + "".join(f"<code>{ui.esc(k)}</code>" for k in sources["keywords"])
                + "</div>")
    ui.footer()
