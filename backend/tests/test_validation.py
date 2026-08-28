import pytest

from app.services.image_validation import ImageValidationError, decode_image, validate_upload


def test_empty_upload_rejected():
    with pytest.raises(ImageValidationError) as exc:
        validate_upload("a.png", "image/png", b"", 1000)
    assert exc.value.code == "empty_upload"


def test_invalid_extension_rejected():
    with pytest.raises(ImageValidationError) as exc:
        validate_upload("notes.txt", "text/plain", b"hello", 1000)
    assert exc.value.code == "invalid_file_type"


def test_oversize_rejected():
    with pytest.raises(ImageValidationError) as exc:
        validate_upload("a.png", "image/png", b"x" * 50, 10)
    assert exc.value.status_code == 413


def test_corrupted_bytes_not_decoded():
    with pytest.raises(ImageValidationError) as exc:
        decode_image(b"not-an-image")
    assert exc.value.code == "corrupted_file"
