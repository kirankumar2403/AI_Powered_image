import numpy as np

from ml.feature_extraction.features import FEATURE_NAMES, extract_features, extract_feature_vector


def test_feature_vector_length_and_finite(sample_bgr):
    feats = extract_features(sample_bgr)
    assert list(feats.keys()) == FEATURE_NAMES
    vec = extract_feature_vector(sample_bgr)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vec).all()


def test_blur_reduces_sharpness(sample_bgr):
    import cv2

    sharp = extract_features(sample_bgr)["sharpness_laplacian_var"]
    blurred = cv2.GaussianBlur(sample_bgr, (21, 21), 6)
    blur_sharp = extract_features(blurred)["sharpness_laplacian_var"]
    assert blur_sharp < sharp


def test_dark_image_has_low_brightness(sample_bgr):
    dark = (sample_bgr.astype(np.float32) * 0.25).astype(np.uint8)
    assert extract_features(dark)["brightness_mean"] < extract_features(sample_bgr)["brightness_mean"]
