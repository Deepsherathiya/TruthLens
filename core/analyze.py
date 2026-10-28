"""
Single entry point for the UI: analyze(text, mode) -> result dict.

mode="style"    -> V1 TF-IDF classifier (writing style)
mode="evidence" -> V2 Fact Check API + Wikipedia + NLI
Both return the same result format (see core/schema.py).
"""

from core.preprocessing import question_to_statement, validate_input
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
            from core.evidence import check_claim  # heavy imports only when needed
            claim, changed = question_to_statement(text)
            result["cleaned"] = claim
            if changed:
                result["warnings"].append(f'Question rewritten as a statement: "{claim}"')
            return check_claim(claim, result)
        result["error"] = f"Unknown mode: {mode}"
    except style_model.ModelNotTrainedError as e:
        result["error"] = str(e)
    except Exception as e:  # never crash the UI on a bad prediction
        result["error"] = f"Prediction failed: {e}"
    return result
