"""Batch analysis: classify a CSV of headlines and download the results."""

import pandas as pd
import streamlit as st

from core import style_model
from core.schema import FAKE, REAL, UNCERTAIN
from ui import components as ui, state

MAX_BATCH_ROWS = 2000
MAX_UPLOAD_MB = 5  # keep in sync with server.maxUploadSize in .streamlit/config.toml
HEADLINE_NAMES = ("title", "headline", "headlines", "text")


def _read_csv(upload) -> pd.DataFrame | None:
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        ui.note(f"This file is {upload.size / 1e6:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.", "tone-fake", "alert")
        return None
    try:
        return pd.read_csv(upload)
    except UnicodeDecodeError:
        upload.seek(0)
        return pd.read_csv(upload, encoding="latin-1")  # common for files saved from Excel
    except pd.errors.EmptyDataError:
        ui.note("The file is empty.", "tone-unc", "alert")
    except pd.errors.ParserError as e:
        ui.note(f"This doesn't look like a valid CSV: {ui.esc(e)}", "tone-fake", "alert")
    return None


def _text_columns(df: pd.DataFrame) -> list[str]:
    """Columns that hold text (object dtype in pandas 2, "str" dtype in pandas 3)."""
    cols = [c for c in df.columns
            if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c])]
    return cols or list(df.columns)


def _sample_csv() -> str:
    return pd.DataFrame({"title": list(state.EXAMPLES.values())}).to_csv(index=False)


def _run(df: pd.DataFrame, column: str, pipeline) -> pd.DataFrame:
    texts = df[column].astype("string").fillna("").tolist()
    bar = st.progress(0.0, text="Analysing…")
    results = style_model.predict_batch(
        pipeline, texts,
        on_progress=lambda done, total: bar.progress(done / total, text=f"Analysing… {done:,} of {total:,}"),
    )
    bar.empty()
    out = df.copy()
    # One output row per input row, in the same order; failed rows are kept
    # with an explanation instead of being dropped.
    out["verdict"] = [r["verdict"] or "" for r in results]
    out["p_fake"] = [r["probabilities"].get(FAKE) for r in results]
    out["p_real"] = [r["probabilities"].get(REAL) for r in results]
    out["error"] = [r["error"] or "" for r in results]
    return out


def _results(batch: dict) -> None:
    out = batch["frame"]
    counts = out["verdict"].value_counts().to_dict()
    counts["Skipped"] = int((out["error"] != "").sum())
    ui.section("Results", f"{ui.esc(batch['name'])} · column “{ui.esc(batch['column'])}”")
    ui.stat_row([
        ("Rows", f"{len(out):,}", "analysed in file order"),
        ("Likely real", f"{counts.get(REAL, 0):,}", "wording resembles real news"),
        ("Likely fake", f"{counts.get(FAKE, 0):,}", "wording resembles fake news"),
        ("Uncertain", f"{counts.get(UNCERTAIN, 0):,}", "close to 50/50 or unknown words"),
        ("Skipped", f"{counts['Skipped']:,}", "empty or too long; see error column"),
    ])
    ui.distribution(counts)
    st.dataframe(
        out, hide_index=True, width="stretch",
        column_config={
            "p_fake": st.column_config.ProgressColumn("P(fake)", format="percent", min_value=0.0, max_value=1.0),
            "p_real": st.column_config.NumberColumn("P(real)", format="percent"),
        },
    )
    st.download_button("Download results CSV", out.to_csv(index=False), "truthlens_results.csv", "text/csv",
                       type="primary", icon=":material/download:", on_click="ignore")


def render() -> None:
    ui.page_header(
        "Batch analysis", "Score a whole file of headlines",
        f"Upload a CSV with one headline per row. Each row gets the writing-style prediction; the "
        f"online evidence check is not run here because it is slow and the free APIs limit how "
        f"often they can be called. Up to {MAX_UPLOAD_MB} MB and the first {MAX_BATCH_ROWS:,} rows.",
    )
    pipeline = state.get_pipeline()
    if pipeline is None:
        ui.empty_state("The model isn't trained yet",
                       "Put <code>Fake.csv</code> and <code>True.csv</code> in <code>data/</code>, run "
                       "<code>python train_model.py</code>, then reload this page.")
        ui.footer()
        return

    upload = st.file_uploader("CSV file", type=["csv"],
                              help=f"Up to {MAX_UPLOAD_MB} MB. Needs a column of headline text.")
    if upload is None:
        st.session_state.batch = None
        ui.empty_state("No file yet",
                       "Your CSV needs a column of headlines, for example <code>title</code> or "
                       "<code>headline</code>. Other columns are kept and returned unchanged.")
        st.download_button("Download a sample CSV", _sample_csv(), "truthlens_sample.csv", "text/csv",
                           icon=":material/download:", on_click="ignore")
        ui.footer()
        return

    df = _read_csv(upload)
    if df is None:
        ui.footer()
        return
    if df.empty:
        ui.note("The CSV has a header but no rows.", "tone-unc", "alert")
        ui.footer()
        return
    if len(df) > MAX_BATCH_ROWS:
        ui.note(f"The file has {len(df):,} rows; only the first {MAX_BATCH_ROWS:,} will be analysed.",
                "tone-unc", "alert")
        df = df.head(MAX_BATCH_ROWS)

    columns = _text_columns(df)
    default = next((c for c in columns if str(c).strip().lower() in HEADLINE_NAMES), columns[0])
    left, right = st.columns([1, 2], gap="large", vertical_alignment="bottom")
    with left:
        column = st.selectbox("Headline column", columns, index=columns.index(default))
    with right:
        run = st.button(f"Analyze {len(df):,} rows", type="primary", icon=":material/play_arrow:")
    with st.expander("Preview the first rows"):
        st.dataframe(df.head(8), hide_index=True, width="stretch")

    file_id = (upload.name, upload.size, column)
    if run:
        st.session_state.batch = {"id": file_id, "name": upload.name, "column": column,
                                  "frame": _run(df, column, pipeline)}
    batch = st.session_state.batch
    if batch and batch["id"] == file_id:
        _results(batch)
    ui.footer()
