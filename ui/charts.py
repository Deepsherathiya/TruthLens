"""
Altair charts, drawn with the theme tokens from ui/theme.py.

Rules followed throughout: thin bars with rounded ends, light gridlines,
a legend whenever colour encodes a category, a tooltip on every mark, and
labels/signs that repeat what the colour says (never colour alone).
"""

import altair as alt
import pandas as pd

SANS_FONT = "IBM Plex Sans"   # loaded by the theme (see .streamlit/config.toml)
MONO_FONT = "IBM Plex Mono"
TOWARD_REAL, TOWARD_FAKE = "Pushes toward Real", "Pushes toward Fake"


def _finish(chart, t: dict, height: int):
    return (
        chart.properties(height=height)
        .configure(background="transparent", font=SANS_FONT)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=t["ink-2"], titleColor=t["ink-3"], labelFontSize=12, titleFontSize=11,
            titleFontWeight=600, gridColor=t["rule"], domainColor=t["rule"], tickColor=t["rule"],
        )
        .configure_legend(
            labelColor=t["ink-2"], titleColor=t["ink-3"], labelFontSize=12, orient="top",
            title=None, symbolStrokeWidth=0, padding=0, offset=12,
        )
    )


def contributions(rows: list[dict], t: dict):
    """Diverging bars: each word's push toward Fake (right) or Real (left)."""
    df = pd.DataFrame(rows)
    df["direction"] = df["score"].map(lambda s: TOWARD_FAKE if s > 0 else TOWARD_REAL)
    order = df.sort_values("score", ascending=False)["term"].tolist()
    reach = max(df["score"].abs().max(), 0.05) * 1.25
    y = alt.Y("term:N", sort=order, title=None,
              axis=alt.Axis(ticks=False, domain=False, labelPadding=10, labelFont=MONO_FONT,
                            labelColor=t["ink"], labelLimit=200))
    base = alt.Chart(df).encode(y=y)
    bars = base.mark_bar(height=14, cornerRadiusEnd=3).encode(
        x=alt.X("score:Q", title="← toward Real · contribution · toward Fake →",
                scale=alt.Scale(domain=[-reach, reach]), axis=alt.Axis(tickCount=5, format="+.2f")),
        color=alt.Color("direction:N", scale=alt.Scale(domain=[TOWARD_REAL, TOWARD_FAKE],
                                                       range=[t["real"], t["fake"]])),
        tooltip=[alt.Tooltip("term:N", title="Term"), alt.Tooltip("score:Q", title="Contribution", format="+.3f"),
                 alt.Tooltip("direction:N", title="Effect")],
    )
    labels_right = base.transform_filter("datum.score > 0").mark_text(
        align="left", dx=5, fontSize=11, color=t["ink-2"], font=MONO_FONT,
    ).encode(x="score:Q", text=alt.Text("score:Q", format="+.2f"))
    labels_left = base.transform_filter("datum.score <= 0").mark_text(
        align="right", dx=-5, fontSize=11, color=t["ink-2"], font=MONO_FONT,
    ).encode(x="score:Q", text=alt.Text("score:Q", format="+.2f"))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=t["ink-3"], strokeWidth=1).encode(x="x:Q")
    return _finish(bars + labels_right + labels_left + zero, t, height=max(150, 32 * len(df) + 40))


def confusion_matrix(matrix: list[list[int]], labels: list[str], t: dict):
    """Heatmap of actual vs predicted counts; shading = share of the actual class."""
    rows = []
    for i, actual in enumerate(labels):
        row_total = sum(matrix[i]) or 1
        for j, predicted in enumerate(labels):
            n = matrix[i][j]
            rows.append({"Actual": actual, "Predicted": predicted, "count": n, "share": n / row_total,
                         "correct": "correct" if i == j else "wrong"})
    df = pd.DataFrame(rows)
    axis = dict(ticks=False, domain=False, labelFontSize=13, labelPadding=8, labelAngle=0)
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=labels, title="PREDICTED", axis=alt.Axis(orient="top", **axis)),
        y=alt.Y("Actual:N", sort=labels, title="ACTUAL", axis=alt.Axis(**axis)),
    )
    cells = base.mark_rect(cornerRadius=4, stroke=t["paper"], strokeWidth=4).encode(
        color=alt.Color("share:Q", scale=alt.Scale(domain=[0, 1], range=[t["seq-lo"], t["seq-hi"]]), legend=None),
        tooltip=[alt.Tooltip("Actual:N"), alt.Tooltip("Predicted:N"), alt.Tooltip("count:Q", title="Headlines", format=","),
                 alt.Tooltip("share:Q", title="Share of actual class", format=".1%")],
    )
    text_color = alt.condition("datum.share > 0.5", alt.value(t["on-seq-hi"]), alt.value(t["ink"]))
    counts = base.mark_text(dy=-8, fontSize=20, font=MONO_FONT, fontWeight=500).encode(
        text=alt.Text("count:Q", format=","), color=text_color)
    shares = base.mark_text(dy=14, fontSize=12).encode(text=alt.Text("share:Q", format=".1%"), color=text_color)
    return _finish(cells + counts + shares, t, height=260)


def model_comparison(df: pd.DataFrame, t: dict, selected: str):
    """
    Dot plot of each metric for each model. Dots (not bars) because the
    values sit close together near the top of the scale, and bars would
    have to start at zero.
    """
    models = list(dict.fromkeys(df["model"]))
    colors = [t["lr"] if m == selected else t["nb"] for m in models]
    metric_order = ["Accuracy", "Precision", "Recall", "F1"]
    lo = max(0.0, (int(df["value"].min() * 20) / 20) - 0.05)
    scale = alt.Scale(domain=[lo, 1], zero=False)  # shared by both layers
    x = alt.X("value:Q", title=None, scale=scale, axis=alt.Axis(format=".0%", tickCount=5))
    y = alt.Y("metric:N", sort=metric_order, title=None,
              axis=alt.Axis(ticks=False, domain=False, labelPadding=10, labelColor=t["ink"]))
    span = alt.Chart(df).mark_rule(color=t["rule"], strokeWidth=3).encode(
        x=alt.X("min(value):Q", scale=scale, title=None), x2="max(value):Q", y=y)
    dots = alt.Chart(df).mark_point(filled=True, size=150, opacity=1).encode(
        x=x, y=y,
        color=alt.Color("model:N", scale=alt.Scale(domain=models, range=colors)),
        shape=alt.Shape("model:N", scale=alt.Scale(domain=models, range=["circle", "diamond"])),
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("metric:N", title="Metric"),
                 alt.Tooltip("value:Q", title="Score", format=".2%")],
    )
    return _finish(span + dots, t, height=230)
