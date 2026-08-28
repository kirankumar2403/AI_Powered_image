"""Human-readable explanations from features, issues, and model output."""

from __future__ import annotations

from typing import Any


def build_explanation(result: dict[str, Any]) -> dict[str, Any]:
    stats = result["statistics"]
    issues = result["issues"]
    label = result["quality_label"]
    factors: list[str] = []

    if stats["sharpness"] < 80:
        factors.append(f"Low sharpness (Laplacian variance {stats['sharpness']:.1f}) indicates blur")
    else:
        factors.append(f"Sharpness is in a usable range ({stats['sharpness']:.1f})")

    if stats["brightness"] < 70:
        factors.append(f"Brightness is low ({stats['brightness']:.1f}/255), suggesting underexposure")
    elif stats["brightness"] > 190:
        factors.append(f"Brightness is high ({stats['brightness']:.1f}/255), suggesting overexposure")
    else:
        factors.append(f"Brightness is balanced ({stats['brightness']:.1f}/255)")

    if stats["noise"] > 8:
        factors.append(f"Elevated noise residual ({stats['noise']:.2f})")
    else:
        factors.append(f"Noise residual is modest ({stats['noise']:.2f})")

    if stats["contrast"] < 22:
        factors.append(f"Low contrast ({stats['contrast']:.1f}) can indicate wash-out or severe degradation")
    else:
        factors.append(f"Contrast is adequate ({stats['contrast']:.1f})")

    if stats["local_anomaly"] > 1.4:
        factors.append(
            f"Localized brightness anomaly ({stats['local_anomaly']:.2f}) may indicate a visual defect"
        )

    issue_types = {i["type"] for i in issues}
    if "visual_defect" in issue_types:
        factors.append(
            "Potential visual defect means a localized scratch, blob, or stain-like pattern "
            "similar to the synthetic defect formulation used in training — not a generic product-defect detector"
        )
    if "severe_degradation" in issue_types:
        factors.append("Combined quality collapse was classified as severe visual degradation")

    if not issues:
        summary = "Image quality is acceptable; no issue heads exceeded the reporting threshold."
    elif label == "POTENTIALLY_DEFECTIVE":
        summary = "The model flagged a potential localized visual defect in addition to global quality cues."
    elif label == "DEGRADED":
        names = ", ".join(sorted(issue_types))
        summary = f"Image quality is degraded. Detected issue types: {names}."
    else:
        summary = "Overall quality is acceptable with only low-severity findings."

    return {
        "summary": summary,
        "contributing_factors": factors[:8],
        "feature_importances": result.get("feature_importances", []),
    }
