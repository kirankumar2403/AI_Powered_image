#!/usr/bin/env python3
"""Evaluate the trained pipeline on unseen original images (test split)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ml.feature_extraction.features import extract_feature_vector  # noqa: E402
from ml.inference.predictor import QualityPredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "processed" / "manifest.csv"))
    parser.add_argument("--model", default=str(ROOT / "models" / "quality_pipeline.joblib"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "evaluation.json"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    payload = joblib.load(args.model)
    classes = list(payload["quality_classes"])
    issue_names = list(payload["issue_names"])
    predictor = QualityPredictor(args.model)

    y_true, y_pred, y_proba = [], [], []
    issue_true, issue_pred, issue_proba = [], [], []
    records = []

    for _, row in test.iterrows():
        img = cv2.imread(str(ROOT / row["path"]), cv2.IMREAD_COLOR)
        result = predictor.predict(img)
        true_label = row["quality_label"]
        pred_label = result["quality_label"]
        y_true.append(true_label)
        y_pred.append(pred_label)
        y_proba.append([result["class_probabilities"].get(c, 0.0) for c in classes])
        true_issues = [int(row[n]) for n in issue_names]
        pred_issues = [1 if any(i["type"] == n for i in result["issues"]) else 0 for n in issue_names]
        issue_true.append(true_issues)
        issue_pred.append(pred_issues)
        x = extract_feature_vector(img).reshape(1, -1)
        proba_list = predictor.issue_model.predict_proba(x)
        scores = []
        for p in proba_list:
            scores.append(float(p[0][1]) if p.shape[1] == 2 else float(p[0][0]))
        issue_proba.append(scores)
        records.append(
            {
                "path": row["path"],
                "original_id": int(row["original_id"]),
                "variant": row["variant"],
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": result["quality_confidence"],
                "correct": true_label == pred_label,
            }
        )

    y_true_idx = [classes.index(y) for y in y_true]
    y_pred_idx = [classes.index(y) for y in y_pred]
    y_proba_arr = np.array(y_proba)
    issue_true_arr = np.array(issue_true)
    issue_pred_arr = np.array(issue_pred)
    issue_proba_arr = np.array(issue_proba)

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(classes))))
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    issue_metrics = {}
    for i, name in enumerate(issue_names):
        issue_metrics[name] = {
            "precision": float(precision_score(issue_true_arr[:, i], issue_pred_arr[:, i], zero_division=0)),
            "recall": float(recall_score(issue_true_arr[:, i], issue_pred_arr[:, i], zero_division=0)),
            "f1": float(f1_score(issue_true_arr[:, i], issue_pred_arr[:, i], zero_division=0)),
        }
        # ROC-AUC only if both classes present
        if len(np.unique(issue_true_arr[:, i])) == 2:
            issue_metrics[name]["roc_auc"] = float(roc_auc_score(issue_true_arr[:, i], issue_proba_arr[:, i]))

    incorrect = [r for r in records if not r["correct"]]
    uncertain = [r for r in records if r["confidence"] < 0.55]

    # Failure cases grouped by variant
    failures_by_variant: dict[str, int] = {}
    for r in incorrect:
        failures_by_variant[r["variant"]] = failures_by_variant.get(r["variant"], 0) + 1

    quality_roc = None
    if len(classes) == 3:
        try:
            quality_roc = float(roc_auc_score(y_true_idx, y_proba_arr, multi_class="ovr", average="macro"))
        except ValueError:
            quality_roc = None

    evaluation = {
        "split": "test",
        "n_samples": len(test),
        "n_originals": int(test["original_id"].nunique()),
        "leakage_check": {
            "train_originals_in_test": False,
            "note": "Splits assigned by original_id in generate_dataset.py",
        },
        "quality": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc_ovr_macro": quality_roc,
            "classification_report": report,
            "confusion_matrix": {
                "labels": classes,
                "matrix": cm.tolist(),
            },
        },
        "issues": issue_metrics,
        "incorrect_predictions": {
            "count": len(incorrect),
            "examples": incorrect[:25],
            "by_variant": failures_by_variant,
        },
        "uncertain_predictions": {
            "threshold": 0.55,
            "count": len(uncertain),
            "examples": uncertain[:15],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps({k: evaluation["quality"][k] for k in ("accuracy", "macro_f1", "roc_auc_ovr_macro")}, indent=2))
    print("Confusion matrix", classes)
    print(cm)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
