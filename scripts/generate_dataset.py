#!/usr/bin/env python3
"""Generate synthetic clean images and controlled degradations.

Source: procedurally generated 256x256 scenes (not scraped photos).
Split is performed on ORIGINAL image IDs so no degraded twin leaks across splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

QUALITY = {
    "clean": "ACCEPTABLE",
    "blur": "DEGRADED",
    "underexposure": "DEGRADED",
    "overexposure": "DEGRADED",
    "noise": "DEGRADED",
    "severe": "DEGRADED",
    "defect": "POTENTIALLY_DEFECTIVE",
}

ISSUE_COLUMNS = [
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "severe_degradation",
    "visual_defect",
]


def make_clean_image(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Structured synthetic 'scene': gradients, shapes, and sinusoidal texture."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)

    # Base gradient / lighting
    gx = rng.uniform(-0.8, 0.8)
    gy = rng.uniform(-0.8, 0.8)
    base = 90 + 80 * (gx * x / size + gy * y / size + 1) / 2
    for c in range(3):
        img[:, :, c] = base + rng.uniform(-25, 25)

    # Soft sinusoidal texture (non-photographic but edge-rich)
    for _ in range(rng.integers(3, 7)):
        fx, fy = rng.uniform(2, 18), rng.uniform(2, 18)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(8, 28)
        wave = amp * np.sin(2 * np.pi * (fx * x / size + fy * y / size) + phase)
        img += wave[:, :, None] * rng.uniform(0.4, 1.0, size=(1, 1, 3))

    # Geometric objects for edges / saturation
    overlay = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(4, 10))):
        color = tuple(int(v) for v in rng.integers(20, 240, size=3))
        if rng.random() < 0.5:
            p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
            p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
            cv2.rectangle(overlay, p1, p2, color, thickness=int(rng.integers(-1, 4)))
        else:
            center = tuple(int(v) for v in rng.integers(0, size, size=2))
            axes = tuple(int(v) for v in rng.integers(8, 70, size=2))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(overlay, center, axes, angle, 0, 360, color, thickness=int(rng.integers(-1, 3)))
    # Fine lines for sharpness signal
    for _ in range(int(rng.integers(6, 14))):
        p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.line(overlay, p1, p2, color, thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA)
    return overlay


def apply_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.choice([11, 15, 21, 25]))
    return cv2.GaussianBlur(img, (k, k), sigmaX=rng.uniform(3.0, 8.0))


def apply_underexposure(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = rng.uniform(0.18, 0.42)
    out = img.astype(np.float32) * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_overexposure(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = rng.uniform(1.7, 2.4)
    bias = rng.uniform(30, 70)
    out = img.astype(np.float32) * scale + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(18, 42)
    noise = rng.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    # occasional salt-pepper
    if rng.random() < 0.5:
        amount = rng.uniform(0.01, 0.04)
        mask = rng.random(img.shape[:2]) < amount
        out[mask] = rng.choice([0, 255])
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_severe(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = apply_blur(img, rng)
    out = apply_noise(out, rng)
    # contrast crush
    lo, hi = rng.uniform(40, 80), rng.uniform(160, 200)
    out = np.clip((out.astype(np.float32) - lo) * (80 / (hi - lo)) + 80, 0, 255)
    if rng.random() < 0.5:
        out = out * rng.uniform(0.35, 0.55)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_defect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Localized defects: scratches, blobs, or color stains — not global quality loss."""
    out = img.copy()
    h, w = out.shape[:2]
    kind = rng.choice(["scratch", "blob", "stain"])
    if kind == "scratch":
        for _ in range(int(rng.integers(1, 4))):
            pts = np.array(
                [[int(rng.integers(0, w)), int(rng.integers(0, h))] for _ in range(int(rng.integers(3, 6)))],
                dtype=np.int32,
            )
            color = (0, 0, 0) if rng.random() < 0.5 else (255, 255, 255)
            cv2.polylines(out, [pts], False, color, thickness=int(rng.integers(2, 5)), lineType=cv2.LINE_AA)
    elif kind == "blob":
        for _ in range(int(rng.integers(1, 3))):
            center = (int(rng.integers(w * 0.15, w * 0.85)), int(rng.integers(h * 0.15, h * 0.85)))
            radius = int(rng.integers(10, 28))
            color = tuple(int(v) for v in (rng.choice([0, 255]), rng.choice([0, 255]), rng.choice([0, 40])))
            cv2.circle(out, center, radius, color, thickness=-1)
    else:
        overlay = out.copy()
        center = (int(rng.integers(w * 0.2, w * 0.8)), int(rng.integers(h * 0.2, h * 0.8)))
        axes = (int(rng.integers(18, 40)), int(rng.integers(12, 30)))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3))
        cv2.ellipse(overlay, center, axes, float(rng.uniform(0, 180)), 0, 360, color, -1)
        out = cv2.addWeighted(out, 0.65, overlay, 0.35, 0)
    return out


DEGRADERS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "severe": apply_severe,
    "defect": apply_defect,
}


def issue_flags(variant: str) -> dict[str, int]:
    flags = {k: 0 for k in ISSUE_COLUMNS}
    if variant == "blur":
        flags["blur"] = 1
    elif variant == "underexposure":
        flags["underexposure"] = 1
    elif variant == "overexposure":
        flags["overexposure"] = 1
    elif variant == "noise":
        flags["noise"] = 1
    elif variant == "severe":
        flags["severe_degradation"] = 1
        flags["blur"] = 1
        flags["noise"] = 1
    elif variant == "defect":
        flags["visual_defect"] = 1
    return flags


def assign_splits(n: int, rng: np.random.Generator, train: float, val: float) -> list[str]:
    ids = np.arange(n)
    rng.shuffle(ids)
    n_train = int(n * train)
    n_val = int(n * val)
    split = {}
    for i, orig in enumerate(ids):
        if i < n_train:
            split[int(orig)] = "train"
        elif i < n_train + n_val:
            split[int(orig)] = "val"
        else:
            split[int(orig)] = "test"
    return [split[i] for i in range(n)]


def copy_samples(processed: Path, samples: Path) -> None:
    """Copy one example of each condition into samples/ for the README."""
    samples.mkdir(parents=True, exist_ok=True)
    mapping = {
        "clean": "01_acceptable_clean.png",
        "blur": "02_blurry.png",
        "underexposure": "03_underexposed.png",
        "overexposure": "04_overexposed.png",
        "noise": "05_noisy.png",
        "severe": "06_severely_degraded.png",
        "defect": "07_potential_visual_defect.png",
    }
    for variant, dest_name in mapping.items():
        matches = sorted((processed / "images").glob(f"*_{variant}.png"))
        if matches:
            data = Path.read_bytes(matches[0])
            (samples / dest_name).write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-originals", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    raw_dir = ROOT / "data" / "raw" / "originals"
    img_dir = ROOT / "data" / "processed" / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    splits = assign_splits(args.n_originals, rng, args.train, args.val)
    rows = []
    variants = ["clean"] + list(DEGRADERS.keys())

    for orig_id in range(args.n_originals):
        clean = make_clean_image(rng, args.size)
        orig_path = raw_dir / f"{orig_id:04d}.png"
        cv2.imwrite(str(orig_path), clean)
        split = splits[orig_id]
        for variant in variants:
            img = clean if variant == "clean" else DEGRADERS[variant](clean, rng)
            rel = f"{orig_id:04d}_{variant}.png"
            dest = img_dir / rel
            cv2.imwrite(str(dest), img)
            flags = issue_flags(variant)
            row = {
                "original_id": orig_id,
                "split": split,
                "variant": variant,
                "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
                "quality_label": QUALITY[variant],
                **flags,
            }
            rows.append(row)

    manifest_path = ROOT / "data" / "processed" / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "source": "Procedurally generated synthetic scenes (OpenCV/NumPy), seed=42",
        "n_originals": args.n_originals,
        "image_size": args.size,
        "variants": variants,
        "split_rule": "Split by original_id only; all variants of one original stay in one split",
        "train_originals": sum(1 for s in splits if s == "train"),
        "val_originals": sum(1 for s in splits if s == "val"),
        "test_originals": sum(1 for s in splits if s == "test"),
        "degradation": {
            "blur": "GaussianBlur kernel 11-25, sigma 3-8",
            "underexposure": "multiplicative scale 0.18-0.42",
            "overexposure": "scale 1.7-2.4 plus bias 30-70, clipped",
            "noise": "Gaussian sigma 18-42, optional 1-4% salt-pepper",
            "severe": "blur + noise + contrast crush + optional darkening",
            "defect": "localized scratch polyline, compact blob, or color stain",
        },
        "labels": QUALITY,
        "seed": args.seed,
    }
    (ROOT / "data" / "processed" / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    copy_samples(ROOT / "data" / "processed", ROOT / "samples")
    print(f"Wrote {len(rows)} samples, manifest={manifest_path}")
    print(json.dumps({k: meta[k] for k in ("train_originals", "val_originals", "test_originals")}, indent=2))


if __name__ == "__main__":
    main()
