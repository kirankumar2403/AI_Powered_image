import io

from app.db.models import Analysis
from app.db.session import SessionLocal


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["model_loaded"] is True


def test_missing_file(client):
    res = client.post("/api/analyze")
    assert res.status_code == 422


def test_invalid_file_type(client):
    res = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_file_type"


def test_corrupted_image(client):
    res = client.post(
        "/api/analyze",
        files={"file": ("bad.png", io.BytesIO(b"png-but-broken"), "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "corrupted_file"


def test_valid_image_schema_and_persistence(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("sample.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    for key in (
        "analysis_id",
        "quality_score",
        "quality_label",
        "issues",
        "statistics",
        "explanation",
        "quality_confidence",
        "image_data",
        "image_mime_type",
    ):
        assert key in body
    assert body["quality_label"] in {"ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE"}
    assert 0 <= body["quality_score"] <= 100
    assert "sharpness" in body["statistics"]
    assert "summary" in body["explanation"]
    assert body["image_data"].startswith("data:image/")
    analysis_id = int(body["analysis_id"])
    db = SessionLocal()
    try:
        row = db.get(Analysis, analysis_id)
        assert row is not None
        assert row.filename == "sample.png"
    finally:
        db.close()

    listed = client.get("/api/analyses")
    assert listed.status_code == 200
    ids = {item["analysis_id"] for item in listed.json()}
    assert body["analysis_id"] in ids

    one = client.get(f"/api/analyses/{body['analysis_id']}")
    assert one.status_code == 200
    assert one.json()["analysis_id"] == body["analysis_id"]
