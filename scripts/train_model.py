#!/usr/bin/env python3
"""Train Random Forest quality + issue models. Does not run at inference time."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.multioutput import MultiOutputClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from ml.feature_extraction.features import FEATURE_NAMES, extract_feature_vector  # noqa: E402

ISSUE_NAMES = [
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "severe_degradation",
    "visual_defect",
]
QUALITY_CLASSES = ["ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE"]


def load_split(manifest: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    subset = manifest[manifest["split"] == split].reset_index(drop=True)
    xs, ys, issues, paths = [], [], [], []
    for _, row in subset.iterrows():
        path = ROOT / row["path"]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        xs.append(extract_feature_vector(img))
        ys.append(QUALITY_CLASSES.index(row["quality_label"]))
        issues.append([int(row[name]) for name in ISSUE_NAMES])
        paths.append(row["path"])
    return np.vstack(xs), np.array(ys), np.array(issues), paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "processed" / "manifest.csv"))
    parser.add_argument("--out", default=str(ROOT / "models" / "quality_pipeline.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-version", default="1.0.0")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    x_train, y_train, i_train, _ = load_split(manifest, "train")
    x_val, y_val, i_val, _ = load_split(manifest, "val")

    quality_model = RandomForestClassifier(
        n_estimators=220,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    quality_model.fit(x_train, y_train)

    issue_model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    )
    issue_model.fit(x_train, i_train)

    # Anomaly model: fit on non-defect training samples only
    non_defect = i_train[:, ISSUE_NAMES.index("visual_defect")] == 0
    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.06,
        random_state=args.seed,
        n_jobs=-1,
    )
    anomaly_model.fit(x_train[non_defect])
    # Threshold: 5th percentile of non-defect val scores (lower = more anomalous)
    val_non_defect = i_val[:, ISSUE_NAMES.index("visual_defect")] == 0
    val_scores = anomaly_model.decision_function(x_val[val_non_defect])
    anomaly_threshold = float(np.percentile(val_scores, 5)) if len(val_scores) else 0.0

    y_pred = quality_model.predict(x_val)
    print("=== Validation quality classification ===")
    print(classification_report(y_val, y_pred, target_names=QUALITY_CLASSES, digits=4))
    i_pred = issue_model.predict(x_val)
    print("=== Validation issue F1 (macro per label) ===")
    for idx, name in enumerate(ISSUE_NAMES):
        print(f"  {name}: {f1_score(i_val[:, idx], i_pred[:, idx], zero_division=0):.4f}")

    payload = {
        "quality_model": quality_model,
        "issue_model": issue_model,
        "anomaly_model": anomaly_model,
        "anomaly_threshold": anomaly_threshold,
        "feature_names": FEATURE_NAMES,
        "quality_classes": QUALITY_CLASSES,
        "issue_names": ISSUE_NAMES,
        "model_version": args.model_version,
        "train_originals": int(manifest[manifest["split"] == "train"]["original_id"].nunique()),
        "val_originals": int(manifest[manifest["split"] == "val"]["original_id"].nunique()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)

    importances = dict(zip(FEATURE_NAMES, quality_model.feature_importances_.tolist()))
    meta = {
        "model_path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "model_version": args.model_version,
        "quality_val_macro_f1": float(f1_score(y_val, y_pred, average="macro")),
        "feature_importances": importances,
        "anomaly_threshold": anomaly_threshold,
        "n_estimators_quality": 220,
    }
    (ROOT / "models" / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
