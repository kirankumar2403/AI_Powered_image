"""Load the trained pipeline and run quality / issue inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.feature_extraction.features import (
    FEATURE_NAMES,
    extract_feature_vector,
    extract_features,
    features_to_public_statistics,
)

QUALITY_LABELS = ("ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE")
ISSUE_NAMES = (
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "severe_degradation",
    "visual_defect",
)


class QualityPredictor:
    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Trained model not found: {path}")
        payload = joblib.load(path)
        self.quality_model = payload["quality_model"]
        self.issue_model = payload["issue_model"]
        self.anomaly_model = payload["anomaly_model"]
        self.feature_names: list[str] = payload["feature_names"]
        self.quality_classes: list[str] = list(payload["quality_classes"])
        self.issue_names: list[str] = list(payload["issue_names"])
        self.model_version: str = payload.get("model_version", "unknown")
        self.anomaly_threshold: float = float(payload.get("anomaly_threshold", 0.0))
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("Model feature names do not match current extractor")

    def predict(self, image_bgr: np.ndarray) -> dict[str, Any]:
        feats = extract_features(image_bgr)
        x = np.array([[feats[n] for n in self.feature_names]], dtype=np.float64)

        quality_idx = int(self.quality_model.predict(x)[0])
        quality_proba = self.quality_model.predict_proba(x)[0]
        quality_label = self.quality_classes[quality_idx]
        quality_conf = float(quality_proba[quality_idx])

        issue_proba = self.issue_model.predict_proba(x)
        # MultiOutputClassifier returns a list of (n, 2) arrays
        issue_scores: dict[str, float] = {}
        for i, name in enumerate(self.issue_names):
            proba = issue_proba[i][0]
            # class 1 probability if both classes exist
            if proba.shape[0] == 2:
                issue_scores[name] = float(proba[1])
            else:
                issue_scores[name] = float(self.issue_model.predict(x)[0][i])

        anomaly_raw = float(self.anomaly_model.decision_function(x)[0])
        # IsolationForest: lower (more negative) = more anomalous
        is_anomaly = anomaly_raw < self.anomaly_threshold

        issues = self._build_issues(feats, issue_scores, is_anomaly, anomaly_raw)
        quality_label, quality_conf = self._reconcile_label(
            quality_label, quality_conf, quality_proba, issues, is_anomaly, feats
        )
        quality_score = self._quality_score(quality_proba, issues, feats)

        importances = self._top_importances()
        return {
            "quality_label": quality_label,
            "quality_score": quality_score,
            "quality_confidence": round(quality_conf, 4),
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(self.quality_classes, quality_proba)
            },
            "issues": issues,
            "statistics": features_to_public_statistics(feats),
            "features": {k: round(float(v), 6) for k, v in feats.items()},
            "feature_importances": importances,
            "anomaly_score": round(anomaly_raw, 4),
            "anomaly_flag": bool(is_anomaly),
            "model_version": self.model_version,
        }

    def _build_issues(
        self,
        feats: dict[str, float],
        scores: dict[str, float],
        is_anomaly: bool,
        anomaly_raw: float,
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for name, prob in scores.items():
            # Anomaly model can support visual_defect when local cues are strong
            if name == "visual_defect" and is_anomaly and feats["local_mean_deviation"] > 1.35:
                prob = max(prob, 0.55)
            if prob < 0.40:
                continue
            severity = _severity(name, prob, feats)
            issues.append(
                {
                    "type": name,
                    "severity": severity,
                    "confidence": round(float(prob), 4),
                }
            )
        issues.sort(key=lambda i: i["confidence"], reverse=True)
        return issues

    def _reconcile_label(
        self,
        quality_label: str,
        quality_conf: float,
        quality_proba: np.ndarray,
        issues: list[dict[str, Any]],
        is_anomaly: bool,
        feats: dict[str, float],
    ) -> tuple[str, float]:
        """Keep the RF class unless a high-confidence defect issue is present."""
        defect = next((i for i in issues if i["type"] == "visual_defect" and i["confidence"] >= 0.6), None)
        if defect and quality_label != "POTENTIALLY_DEFECTIVE":
            return "POTENTIALLY_DEFECTIVE", min(0.99, (quality_conf + defect["confidence"]) / 2)
        if (
            is_anomaly
            and feats["local_mean_deviation"] > 1.6
            and feats["extreme_blob_ratio"] > 0.004
            and quality_label == "ACCEPTABLE"
        ):
            return "POTENTIALLY_DEFECTIVE", 0.55
        return quality_label, quality_conf

    def _quality_score(
        self,
        quality_proba: np.ndarray,
        issues: list[dict[str, Any]],
        feats: dict[str, float],
    ) -> int:
        """Transparent score: class probabilities plus bounded issue penalties.

        score = 100*P(ACCEPTABLE) + 55*P(DEGRADED) + 25*P(POTENTIALLY_DEFECTIVE)
                - severity penalties, clipped to [0, 100]
        """
        proba_map = {cls: float(p) for cls, p in zip(self.quality_classes, quality_proba)}
        base = (
            100.0 * proba_map.get("ACCEPTABLE", 0.0)
            + 55.0 * proba_map.get("DEGRADED", 0.0)
            + 25.0 * proba_map.get("POTENTIALLY_DEFECTIVE", 0.0)
        )
        penalty = 0.0
        for issue in issues:
            penalty += {"low": 4.0, "medium": 10.0, "high": 18.0}[issue["severity"]]
        score = int(round(np.clip(base - penalty, 0, 100)))
        return score

    def _top_importances(self, k: int = 8) -> list[dict[str, float | str]]:
        imp = getattr(self.quality_model, "feature_importances_", None)
        if imp is None:
            return []
        order = np.argsort(imp)[::-1][:k]
        return [
            {"feature": self.feature_names[i], "importance": round(float(imp[i]), 4)}
            for i in order
        ]


def _severity(issue_type: str, prob: float, feats: dict[str, float]) -> str:
    extreme = False
    if issue_type == "blur" and feats["sharpness_laplacian_var"] < 40:
        extreme = True
    if issue_type == "underexposure" and feats["brightness_mean"] < 45:
        extreme = True
    if issue_type == "overexposure" and feats["brightness_mean"] > 210:
        extreme = True
    if issue_type == "noise" and feats["noise_median_residual"] > 12:
        extreme = True
    if issue_type == "severe_degradation" and feats["contrast_std"] < 18:
        extreme = True
    if issue_type == "visual_defect" and feats["local_mean_deviation"] > 2.0:
        extreme = True
    if extreme or prob >= 0.80:
        return "high"
    if prob >= 0.58:
        return "medium"
    return "low"
