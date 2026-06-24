"""Read the usage-rate percentage shown directly below an avatar."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR


PERCENT_PATTERN = re.compile(r"(\d{1,3}(?:[\.,]\d+)?)\s*%")


def create_ocr_engine() -> RapidOCR:
    """Create the OCR engine with models stored in a writable cache directory."""
    model_dir = Path(tempfile.gettempdir()) / "genshin_avatar_rapidocr_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return RapidOCR(params={"Global.model_root_dir": str(model_dir)})


def crop_usage_rate_region(
    image_rgb: np.ndarray,
    avatar_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop the percentage line immediately below an avatar."""
    x, y, width, height = avatar_box
    image_height, image_width = image_rgb.shape[:2]
    horizontal_pad = round(width * 0.04)
    top = min(image_height, y + height)
    bottom = min(image_height, y + height + round(height * 0.42))
    left = max(0, x - horizontal_pad)
    right = min(image_width, x + width + horizontal_pad)
    return image_rgb[top:bottom, left:right]


def _format_usage_rate(value: float) -> str:
    return f"{value:g}%"


def normalize_usage_rate_text(text: str) -> tuple[float | None, str]:
    """Extract and normalize a percentage from OCR text."""
    cleaned = str(text).replace("O", "0").replace("o", "0").replace("％", "%").replace(",", ".")
    match = PERCENT_PATTERN.search(cleaned)
    if not match:
        return None, ""

    value = float(match.group(1).replace(",", "."))
    if not 0 <= value <= 100:
        return None, ""
    return value, _format_usage_rate(value)


def recognize_usage_rate(
    image_rgb: np.ndarray,
    avatar_box: tuple[int, int, int, int],
    ocr_engine: RapidOCR,
) -> tuple[float | None, str, float]:
    """Return usage-rate number, normalized percentage text, and OCR confidence."""
    region = crop_usage_rate_region(image_rgb, avatar_box)
    if region.size == 0:
        return None, "", 0.0

    enlarged = cv2.resize(region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = ocr_engine(enlarged, use_det=False, use_cls=False, use_rec=True)
    texts = getattr(result, "txts", ()) or ()
    scores = getattr(result, "scores", ()) or ()
    if not texts:
        return None, "", 0.0

    value, normalized = normalize_usage_rate_text(str(texts[0]))
    return value, normalized, float(scores[0]) if scores else 0.0
