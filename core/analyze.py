"""
Single entry point for the UI: analyze(text, mode) -> result dict.

V1 implements mode="style". V2 will add mode="evidence" here without
changing the result format (see core/schema.py).
"""

from core.preprocessing import validate_input
from core.schema import empty_result
from core import style_model


def analyze(text: str, mode: str = "style", pipeline=None) -> dict:
    result = empty_result(mode, text)

    error = validate_input(text)
    if error:
        result["error"] = error
        return result

    try:
        if mode == "style":
            if pipeline is None:
                pipeline = style_model.load_pipeline()
            return style_model.predict_style(pipeline, text, result)
        if mode == "evidence":
            result["error"] = "Evidence mode is not built yet (planned for V2)."
            return result
        result["error"] = f"Unknown mode: {mode}"
    except style_model.ModelNotTrainedError as e:
        result["error"] = str(e)
    except Exception as e:  # never crash the UI on a bad prediction
        result["error"] = f"Prediction failed: {e}"
    return result
