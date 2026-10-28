"""
Single entry point for the UI: analyze(text, mode) -> result dict.

mode="style"     V1: wording-based model prediction (core/style_model.py)
mode="evidence"  V2: search independent online sources (core/evidence.py)

Both return the same result format (see core/schema.py).
"""

from core import evidence, style_model
from core.preprocessing import text_stats, validate_input
from core.schema import empty_result


def analyze(text: str, mode: str = "style", pipeline=None,
            api_key: str | None = None, fetchers: dict | None = None) -> dict:
    result = empty_result(mode, text)

    error = validate_input(text)
    if error:
        result["error"] = error
        return result
    result["stats"] = text_stats(text)

    try:
        if mode == "style":
            if pipeline is None:
                pipeline = style_model.load_pipeline()
            return style_model.predict_style(pipeline, text, result)
        if mode == "evidence":
            return evidence.check_sources(text, result, api_key=api_key, fetchers=fetchers)
        result["error"] = f"Unknown mode: {mode}"
    except style_model.ModelNotTrainedError as e:
        result["error"] = str(e)
    except Exception as e:  # never crash the UI on a bad prediction
        result["error"] = f"Analysis failed: {e}"
    return result
