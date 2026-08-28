"""Image-quality feature extraction with OpenCV and NumPy.

Each feature is documented in FEATURE_SPECS (what / why / how / ML role).
"""

from __future__ import annotations

import cv2
import numpy as np

FEATURE_NAMES: list[str] = [
    "sharpness_laplacian_var",
    "edge_density",
    "gradient_mean",
    "brightness_mean",
    "brightness_p5",
    "brightness_p95",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "contrast_std",
    "histogram_entropy",
    "histogram_skew",
    "noise_median_residual",
    "high_freq_energy",
    "saturation_mean",
    "saturation_std",
    "color_channel_imbalance",
    "local_mean_deviation",
    "tophat_residue",
    "blackhat_residue",
    "extreme_blob_ratio",
]

FEATURE_SPECS: dict[str, dict[str, str]] = {
    "sharpness_laplacian_var": {
        "measures": "Focus / high-frequency edge energy",
        "why": "Blurred images have a much smaller Laplacian response",
        "how": "Variance of Laplacian (second derivative) on grayscale",
        "ml_role": "Primary cue for blur / insufficient sharpness",
    },
    "edge_density": {
        "measures": "Fraction of pixels that are Canny edges",
        "why": "Blur and severe degradation suppress edges",
        "how": "Canny edge map mean",
        "ml_role": "Supports sharpness and severe-degradation classes",
    },
    "gradient_mean": {
        "measures": "Average Sobel gradient magnitude (texture)",
        "why": "Texture collapses under blur and some defects add local gradients",
        "how": "Mean of hypot(Sobel-x, Sobel-y)",
        "ml_role": "Texture statistic for quality and defect models",
    },
    "brightness_mean": {
        "measures": "Overall luminance",
        "why": "Underexposure is dark; overexposure is bright",
        "how": "Mean of 8-bit grayscale",
        "ml_role": "Exposure-related class separation",
    },
    "brightness_p5": {
        "measures": "Dark-end luminance (5th percentile)",
        "why": "Captures crushed shadows better than the mean alone",
        "how": "np.percentile(gray, 5)",
        "ml_role": "Underexposure cue",
    },
    "brightness_p95": {
        "measures": "Highlight luminance (95th percentile)",
        "why": "Captures clipped highlights",
        "how": "np.percentile(gray, 95)",
        "ml_role": "Overexposure cue",
    },
    "dark_pixel_ratio": {
        "measures": "Share of near-black pixels",
        "why": "Underexposed images have large dark regions",
        "how": "Mean of gray < 20",
        "ml_role": "Underexposure issue head",
    },
    "bright_pixel_ratio": {
        "measures": "Share of near-white pixels",
        "why": "Overexposed images have large clipped highlights",
        "how": "Mean of gray > 235",
        "ml_role": "Overexposure issue head",
    },
    "contrast_std": {
        "measures": "Global contrast",
        "why": "Severe degradation and wash-out reduce contrast",
        "how": "Standard deviation of grayscale",
        "ml_role": "Severe degradation and exposure extremes",
    },
    "histogram_entropy": {
        "measures": "Information content of the intensity histogram",
        "why": "Clipped or washed images have lower entropy",
        "how": "Shannon entropy of 256-bin histogram",
        "ml_role": "Global quality / corruption-like wash-out",
    },
    "histogram_skew": {
        "measures": "Asymmetry of the intensity distribution",
        "why": "Dark-skewed vs bright-skewed histograms indicate exposure issues",
        "how": "Third standardized moment of grayscale",
        "ml_role": "Underexposure vs overexposure",
    },
    "noise_median_residual": {
        "measures": "Grain / sensor-like noise",
        "why": "Impulse and Gaussian noise remain after a median filter",
        "how": "Mean absolute residual vs medianBlur(k=5)",
        "ml_role": "Primary noise cue",
    },
    "high_freq_energy": {
        "measures": "High-frequency amplitude (noise + fine detail)",
        "why": "Complements median residual; noisy images have extra HF energy without edges",
        "how": "Mean |gray - GaussianBlur(gray)|",
        "ml_role": "Noise vs sharpness disambiguation with Laplacian",
    },
    "saturation_mean": {
        "measures": "Average HSV saturation",
        "why": "Overexposure and wash-out desaturate; some stains increase saturation locally",
        "how": "Mean of HSV S channel / 255",
        "ml_role": "Color quality and stain-like defects",
    },
    "saturation_std": {
        "measures": "Saturation variation",
        "why": "Localized color stains raise saturation variance",
        "how": "Std of HSV S / 255",
        "ml_role": "Visual defect cue",
    },
    "color_channel_imbalance": {
        "measures": "Max |channel mean - gray mean|",
        "why": "Color casts and stains unbalance BGR channels",
        "how": "Max absolute difference of B/G/R means vs gray mean",
        "ml_role": "Defect / severe color degradation",
    },
    "local_mean_deviation": {
        "measures": "Strongest tile-level brightness anomaly",
        "why": "Localized defects (blobs) differ from global mean more than uniform issues",
        "how": "Max |tile_mean - global_mean| / (global_std + eps) over 8x8 tiles",
        "ml_role": "Potential visual defect / anomaly cue",
    },
    "tophat_residue": {
        "measures": "Bright thin structures (scratches, glints)",
        "why": "Morphological white top-hat highlights small bright defects",
        "how": "Mean of morphological top-hat with 9x9 ellipse",
        "ml_role": "Scratch-like visual defects",
    },
    "blackhat_residue": {
        "measures": "Dark thin structures (scratches, dirt)",
        "why": "Black-hat highlights small dark defects",
        "how": "Mean of morphological black-hat with 9x9 ellipse",
        "ml_role": "Scratch/spot visual defects",
    },
    "extreme_blob_ratio": {
        "measures": "Fraction of pixels in compact extreme-intensity blobs",
        "why": "Defect spots are compact; global over/underexposure is widespread",
        "how": "After thresholding tails, keep components with area in [20, 0.08*N]",
        "ml_role": "Potential visual defect vs global exposure",
    },
}


def _shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _skewness(values: np.ndarray) -> float:
    x = values.astype(np.float64).ravel()
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-8:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def _local_mean_deviation(gray: np.ndarray, tiles: int = 8) -> float:
    h, w = gray.shape
    th, tw = max(h // tiles, 1), max(w // tiles, 1)
    g = gray.astype(np.float64)
    global_mean = g.mean()
    global_std = g.std() + 1e-6
    max_z = 0.0
    for i in range(tiles):
        for j in range(tiles):
            tile = g[i * th : (i + 1) * th, j * tw : (j + 1) * tw]
            if tile.size == 0:
                continue
            z = abs(tile.mean() - global_mean) / global_std
            if z > max_z:
                max_z = z
    return float(max_z)


def _extreme_blob_ratio(gray: np.ndarray) -> float:
    """Compact extreme blobs, excluding huge clipped regions."""
    h, w = gray.shape
    n = h * w
    low = (gray < 18).astype(np.uint8)
    high = (gray > 237).astype(np.uint8)
    mask = cv2.bitwise_or(low, high)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0.0
    min_area, max_area = 20, int(0.08 * n)
    blob_px = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            blob_px += area
    return float(blob_px / n)


def extract_features(image_bgr: np.ndarray) -> dict[str, float]:
    """Return a named dict of quality features for a BGR uint8 image."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray_f = gray.astype(np.float64)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mean = float(np.mean(np.hypot(gx, gy)))

    brightness_mean = float(gray_f.mean())
    brightness_p5 = float(np.percentile(gray_f, 5))
    brightness_p95 = float(np.percentile(gray_f, 95))
    dark_pixel_ratio = float(np.mean(gray < 20))
    bright_pixel_ratio = float(np.mean(gray > 235))
    contrast_std = float(gray_f.std())
    histogram_entropy = _shannon_entropy(gray)
    histogram_skew = _skewness(gray_f)

    median = cv2.medianBlur(gray, 5)
    noise_median_residual = float(np.mean(np.abs(gray_f - median.astype(np.float64))))
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    high_freq_energy = float(np.mean(np.abs(gray_f - blurred.astype(np.float64))))

    sat = hsv[:, :, 1].astype(np.float64) / 255.0
    saturation_mean = float(sat.mean())
    saturation_std = float(sat.std())
    b_mean, g_mean, r_mean = [float(image_bgr[:, :, c].mean()) for c in range(3)]
    color_channel_imbalance = float(
        max(abs(b_mean - brightness_mean), abs(g_mean - brightness_mean), abs(r_mean - brightness_mean))
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    return {
        "sharpness_laplacian_var": sharpness,
        "edge_density": edge_density,
        "gradient_mean": gradient_mean,
        "brightness_mean": brightness_mean,
        "brightness_p5": brightness_p5,
        "brightness_p95": brightness_p95,
        "dark_pixel_ratio": dark_pixel_ratio,
        "bright_pixel_ratio": bright_pixel_ratio,
        "contrast_std": contrast_std,
        "histogram_entropy": histogram_entropy,
        "histogram_skew": histogram_skew,
        "noise_median_residual": noise_median_residual,
        "high_freq_energy": high_freq_energy,
        "saturation_mean": saturation_mean,
        "saturation_std": saturation_std,
        "color_channel_imbalance": color_channel_imbalance,
        "local_mean_deviation": _local_mean_deviation(gray),
        "tophat_residue": float(tophat.mean()),
        "blackhat_residue": float(blackhat.mean()),
        "extreme_blob_ratio": _extreme_blob_ratio(gray),
    }


def extract_feature_vector(image_bgr: np.ndarray) -> np.ndarray:
    feats = extract_features(image_bgr)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


def features_to_public_statistics(feats: dict[str, float]) -> dict[str, float]:
    """Compact statistics exposed in the API/UI."""
    return {
        "sharpness": round(feats["sharpness_laplacian_var"], 4),
        "brightness": round(feats["brightness_mean"], 4),
        "contrast": round(feats["contrast_std"], 4),
        "noise": round(feats["noise_median_residual"], 4),
        "saturation": round(feats["saturation_mean"], 4),
        "dark_pixel_ratio": round(feats["dark_pixel_ratio"], 4),
        "bright_pixel_ratio": round(feats["bright_pixel_ratio"], 4),
        "local_anomaly": round(feats["local_mean_deviation"], 4),
        "edge_density": round(feats["edge_density"], 4),
    }
