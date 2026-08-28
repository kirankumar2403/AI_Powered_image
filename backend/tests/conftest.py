from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import cv2
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def sample_bgr():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = (80, 120, 160)
    cv2.rectangle(img, (20, 20), (90, 90), (200, 40, 40), -1)
    cv2.line(img, (0, 0), (127, 127), (255, 255, 255), 2)
    return img


@pytest.fixture(scope="session")
def png_bytes(sample_bgr):
    ok, buf = cv2.imencode(".png", sample_bgr)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c
