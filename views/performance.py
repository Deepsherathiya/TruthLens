"""Model performance: held-out test metrics written by train_model.py (never hard-coded)."""

import pandas as pd
import streamlit as st

from ui import charts, components as ui, state, theme

METRIC_NAMES = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}


def render() -> None:
    t = theme.tokens()
    m = state.get_metrics()
    if m is None:
        ui.page_header("Model performance", "No results yet",
                       "Metrics appear here after the model has been trained and tested.")
        ui.empty_state("Train the model first",
                       "Run <code>python train_model.py</code>. It splits the data, trains on 80%, "
                       "tests on the held-out 20% and writes <code>models/metrics.json</code>, "
                       "which this page reads.")
        ui.footer()
        return

    d, s = m["dataset"], m["split"]
    sel = m["selected_model"]
    pos = m["positive_class"]
    ui.page_header(
        "Model performance", "How the style model does on unseen headlines",
        f"Scores from the held-out test set: {s['test_rows']:,} headlines the model never saw in training "
        f"({int(s['test_size'] * 100)}% stratified split, seed {s['random_state']}). "
        f"Dataset: {ui.esc(d['name'])}.",
    )

    r = m["models"][sel]
    ui.stat_row([
        ("Accuracy", ui.pct(r["accuracy"]), "all test headlines"),
        ("Precision", ui.pct(r["precision"]), f"of predicted {pos}, share truly {pos}"),
        ("Recall", ui.pct(r["recall"]), f"of truly {pos}, share caught"),
        ("F1 score", ui.pct(r["f1"]), "balance of precision and recall"),
    ])
    ui.show(f'<p class="tl-fine">Model: {ui.esc(sel)}. Precision, recall and F1 treat '
            f'<strong>{ui.esc(pos)}</strong> as the positive class.</p>')

    left, right = st.columns(2, gap="large")
    labels = [label.capitalize() for label in m["label_order"]]
    with left:
        ui.section("Confusion matrix", f"{ui.esc(sel)} · shading = share of the actual class")
        st.altair_chart(charts.confusion_matrix(r["confusion_matrix"], labels, t), width="stretch", theme=None)
    with right:
        ui.section("Model comparison", "same split, same features")
        rows = [{"model": name, "metric": METRIC_NAMES[k], "value": v}
                for name, res in m["models"].items() for k, v in res.items() if k in METRIC_NAMES]
        comp = pd.DataFrame(rows)
        st.altair_chart(charts.model_comparison(comp, t, sel), width="stretch", theme=None)
        with st.expander("Show as a table"):
            table = comp.pivot(index="model", columns="metric", values="value")[list(METRIC_NAMES.values())]
            st.dataframe(table.style.format("{:.2%}"), width="stretch")
        ui.show('<p class="tl-fine">Judge models on precision, recall and F1 together, not on accuracy alone.</p>')

    ui.section("Where the data came from")
    removed = d.get("dropped_empty", 0) + d.get("dropped_conflicting", 0) + d.get("dropped_duplicates", 0)
    classes = d.get("class_counts", {})
    ui.stat_row([
        ("Raw headlines", f"{d['raw_rows']:,}", "before cleaning"),
        ("Removed", f"{removed:,}", f"{d.get('dropped_duplicates', 0):,} duplicates · "
                                    f"{d.get('dropped_conflicting', 0):,} in both classes · "
                                    f"{d.get('dropped_empty', 0):,} empty"),
        ("Used", f"{d['final_rows']:,}", " · ".join(f"{k}: {v:,}" for k, v in sorted(classes.items()))),
        ("Train / test", f"{s['train_rows']:,} / {s['test_rows']:,}", "split before fitting TF-IDF"),
    ])

    ui.section("Words the model leans on", "largest logistic-regression weights")
    # The raw coefficient sign depends on the model's class order, so show the
    # size of each push; the table it sits in says which way it pushes.
    terms = m["top_terms"]
    a, b = st.columns(2, gap="large")
    with a:
        ui.show(ui.term_table([(w, abs(v)) for w, v in terms.get("fake", [])[:15]], "tone-fake",
                              caption="Pushes toward fake", value_format="{:.2f}", header=("Term", "Strength")))
    with b:
        ui.show(ui.term_table([(w, abs(v)) for w, v in terms.get("real", [])[:15]], "tone-real",
                              caption="Pushes toward real", value_format="{:.2f}", header=("Term", "Strength")))
    ui.show('<p class="tl-fine" style="margin-top:.6rem">These are statistical associations in this dataset '
            '(mostly writing style: Reuters headlines use "says" and drop small words), not proof that a '
            'word makes a story true or false.</p>')

    ui.section("Headlines the model got wrong", "from the test set")
    wrong = pd.DataFrame(m.get("misclassified_examples", []))
    if wrong.empty:
        ui.show('<p class="tl-fine">No misclassified test headlines were recorded.</p>')
    else:
        st.dataframe(
            wrong, hide_index=True, width="stretch",
            column_config={
                "title": st.column_config.TextColumn("Headline", width="large"),
                "true": "Actual", "predicted": "Predicted",
                "fake_probability": st.column_config.ProgressColumn(
                    "P(fake)", format="percent", min_value=0.0, max_value=1.0),
            },
        )
    ui.footer(f"Trained {ui.esc(m['trained_at'])} · scikit-learn {ui.esc(m.get('sklearn_version', '?'))}")
