"""
TruthLens - simple demo dashboard (V1). To be polished later.

Run:  streamlit run app.py
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from core import style_model
from core.analyze import analyze
from core.schema import DISCLAIMER, FAKE, REAL

MAX_BATCH_ROWS = 2000
EXAMPLES = [
    "Senate passes tax bill, Trump says he will sign it",
    "WATCH: Hillary Just DESTROYED This GOP Moron On Live TV",
    "EU leaders agree to start next phase of Brexit talks",
    "BREAKING: Obama Caught Lying AGAIN, And It's Worse Than You Think",
]

st.set_page_config(page_title="TruthLens", page_icon="🔎", layout="wide")


@st.cache_resource
def get_pipeline():
    try:
        return style_model.load_pipeline()
    except style_model.ModelNotTrainedError:
        return None


pipeline = get_pipeline()
st.session_state.setdefault("history", [])
st.session_state.setdefault("headline", "")

st.title("🔎 TruthLens")
st.caption("NLP-Powered News Analysis · V1 demo")

if pipeline is None:
    st.error("No trained model found. Put Fake.csv and True.csv in data/ and run `python train_model.py`.")
    st.stop()

tab_analyze, tab_batch, tab_perf = st.tabs(["Analyze", "Batch CSV", "Model Performance"])


def badge(verdict: str) -> str:
    icon = {REAL: "✅", FAKE: "⚠️"}.get(verdict, "❔")
    return f"{icon} {verdict}"


# ---------------- Analyze ----------------
with tab_analyze:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex[:40] + "…", use_container_width=True):
            st.session_state.headline = ex

    text = st.text_area("Headline", key="headline", height=80, max_chars=300)
    if st.button("Analyze Headline", type="primary"):
        r = analyze(text, pipeline=pipeline)
        if r["error"]:
            st.warning(r["error"])
        else:
            st.subheader(badge(r["verdict"]))
            c1, c2 = st.columns(2)
            c1.metric("Model-estimated P(Fake)", f"{r['probabilities'][FAKE]:.1%}")
            c2.metric("Model-estimated P(Real)", f"{r['probabilities'][REAL]:.1%}")
            for w in r["warnings"]:
                st.info(w)

            if r["word_contributions"]:
                st.write("**Words that influenced the prediction** (positive → Fake, negative → Real)")
                wc = pd.DataFrame(r["word_contributions"]).set_index("term")
                st.bar_chart(wc["score"], horizontal=True)
            st.caption(f"Cleaned text the model saw: `{r['cleaned']}`")

            st.session_state.history.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "headline": text.strip(),
                "verdict": r["verdict"],
                "P(Fake)": r["probabilities"][FAKE],
            })
    st.caption(DISCLAIMER)

    if st.session_state.history:
        st.divider()
        st.write("**History (this session only)**")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ---------------- Batch ----------------
with tab_batch:
    up = st.file_uploader("Upload a CSV with a headline column", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None
        if df is not None and df.empty:
            st.warning("The CSV is empty.")
        elif df is not None:
            if len(df) > MAX_BATCH_ROWS:
                st.warning(f"Only the first {MAX_BATCH_ROWS} rows will be analysed.")
                df = df.head(MAX_BATCH_ROWS)
            text_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
            default = next((c for c in text_cols if c.lower() in ("title", "headline")), text_cols[0])
            col = st.selectbox("Headline column", text_cols, index=text_cols.index(default))
            if st.button("Run batch analysis", type="primary"):
                results = style_model.predict_batch(pipeline, df[col].astype("string").fillna("").tolist())
                out = df.copy()
                out["verdict"] = [r["verdict"] or "" for r in results]
                out["p_fake"] = [r["probabilities"].get(FAKE) for r in results]
                out["p_real"] = [r["probabilities"].get(REAL) for r in results]
                out["error"] = [r["error"] or "" for r in results]
                skipped = (out["error"] != "").sum()
                if skipped:
                    st.info(f"{skipped} row(s) could not be analysed; see the 'error' column.")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download results CSV", out.to_csv(index=False), "truthlens_results.csv", "text/csv")

# ---------------- Performance ----------------
with tab_perf:
    m = style_model.load_metrics()
    if m is None:
        st.info("No metrics yet. Run `python train_model.py`.")
    else:
        d, s = m["dataset"], m["split"]
        st.write(
            f"**Dataset:** {d['name']} · {d['final_rows']:,} headlines after cleaning "
            f"(removed {d['dropped_duplicates']:,} duplicates) · "
            f"**Test set:** {s['test_rows']:,} rows, stratified {int(s['test_size']*100)}% split, seed {s['random_state']}"
        )
        st.caption(f"Precision/recall are for the '{m['positive_class']}' class. Trained {m['trained_at']}.")

        comp = pd.DataFrame({name: {k: v for k, v in r.items() if k != "confusion_matrix"}
                             for name, r in m["models"].items()}).T
        st.write("**Model comparison (same split)**")
        st.dataframe(comp.style.format("{:.2%}"), use_container_width=True)

        sel = m["selected_model"]
        labels = [l.capitalize() for l in m["label_order"]]
        cm = pd.DataFrame(m["models"][sel]["confusion_matrix"],
                          index=[f"Actual {l}" for l in labels],
                          columns=[f"Predicted {l}" for l in labels])
        st.write(f"**Confusion matrix — {sel}**")
        st.dataframe(cm, use_container_width=False)

        st.write("**Top words the model associates with each class**")
        st.caption("These are statistical associations in this dataset, not proof of anything.")
        c1, c2 = st.columns(2)
        c1.write("→ Fake")
        c1.dataframe(pd.DataFrame(m["top_terms"]["fake"], columns=["term", "weight"]), hide_index=True)
        c2.write("→ Real")
        c2.dataframe(pd.DataFrame(m["top_terms"]["real"], columns=["term", "weight"]), hide_index=True)

        st.write("**Examples the model got wrong**")
        st.dataframe(pd.DataFrame(m["misclassified_examples"]), use_container_width=True, hide_index=True)
