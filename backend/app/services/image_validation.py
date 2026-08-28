from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_MIME_HINTS = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/webp",
    "application/octet-stream",
}


class ImageValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400, code: str = "invalid_image") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


def safe_filename(name: str) -> str:
    base = Path(name or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return cleaned[:200] or "upload"


def validate_upload(filename: str, content_type: str | None, data: bytes, max_bytes: int) -> None:
    if data is None or len(data) == 0:
        raise ImageValidationError("Empty upload: no file bytes were received.", code="empty_upload")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"File exceeds the maximum size of {max_bytes} bytes.",
            status_code=413,
            code="file_too_large",
        )
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            "Unsupported file type. Allowed: JPEG, PNG, BMP, WEBP.",
            code="invalid_file_type",
        )
    if content_type and content_type.lower() not in ALLOWED_MIME_HINTS:
        # Some browsers send empty or generic types; only reject clearly wrong types.
        if not content_type.lower().startswith("image/"):
            raise ImageValidationError(
                "Unsupported content type. Upload an image file.",
                code="invalid_file_type",
            )


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes with OpenCV. Distinguishes unreadable files from valid pixels."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError(
            "The file is corrupted or not a decodable image. "
            "This is file corruption, not a visual quality defect.",
            code="corrupted_file",
        )
    if image.size == 0 or image.shape[0] < 8 or image.shape[1] < 8:
        raise ImageValidationError("Image is too small to analyze.", code="invalid_image")
    return image
