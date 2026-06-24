"""Load standard avatars and compare screenshot crops with several methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


MATCH_SIZE = 128


def display_name(name: str) -> str:
    """Merge the two Traveler avatar variants into one exported name."""
    return "旅行者" if name in {"旅行者男", "旅行者女"} else name


@dataclass
class AvatarReference:
    name: str
    rgb: np.ndarray
    alpha: np.ndarray
    gray: np.ndarray
    histogram: np.ndarray
    phash: imagehash.ImageHash
    orb_descriptors: np.ndarray | None


def _prepare(image_rgb: np.ndarray, trim_border: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if trim_border:
        height, width = image_rgb.shape[:2]
        inset = max(1, round(min(height, width) * 0.025))
        image_rgb = image_rgb[inset:height - inset, inset:width - inset]
    resized = cv2.resize(image_rgb, (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return resized, gray


def _badge_neutralized(image_rgb: np.ndarray) -> np.ndarray:
    """Reduce the influence of top-right NEW/trend badges before comparison."""
    image = image_rgb.copy()
    height, width = image.shape[:2]
    badge_w = max(10, round(width * 0.24))
    badge_h = max(10, round(height * 0.24))
    corner_pixels = np.concatenate(
        (image[:12, :12], image[-12:, :12], image[-12:, -12:]),
        axis=0,
    )
    fill = np.median(corner_pixels.reshape(-1, 3), axis=0).astype(np.uint8)
    mask = np.zeros((badge_h, badge_w), dtype=bool)
    for y in range(badge_h):
        limit = badge_w - round(y * badge_w / max(badge_h, 1))
        mask[y, :max(1, limit)] = True
    patch = image[:badge_h, width - badge_w:width]
    patch[mask] = fill
    image[:badge_h, width - badge_w:width] = patch
    return image


def _histogram(image_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _orb(gray: np.ndarray) -> np.ndarray | None:
    detector = cv2.ORB_create(nfeatures=350)
    _, descriptors = detector.detectAndCompute(gray, None)
    return descriptors


def load_avatar_library(folder: Path) -> list[AvatarReference]:
    references: list[AvatarReference] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        try:
            rgba = np.asarray(Image.open(path).convert("RGBA"))
            rgb, gray = _prepare(rgba[:, :, :3])
            alpha = cv2.resize(rgba[:, :, 3], (MATCH_SIZE, MATCH_SIZE), interpolation=cv2.INTER_AREA)
            references.append(
                AvatarReference(
                    name=str(path.stem),
                    rgb=rgb,
                    alpha=alpha,
                    gray=gray,
                    histogram=_histogram(rgb),
                    phash=imagehash.phash(Image.fromarray(rgb)),
                    orb_descriptors=_orb(gray),
                )
            )
        except Exception:
            # A single broken reference image should not stop the whole tool.
            continue
    return references


def _orb_score(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a, b, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    return min(1.0, len(good) / max(12, min(len(a), len(b)) * 0.18))


def match_crop(crop_rgb: np.ndarray, library: list[AvatarReference]) -> list[dict]:
    """Rank the library using hash, histogram, SSIM, and ORB feature matching."""
    rgb, gray = _prepare(_badge_neutralized(crop_rgb), trim_border=True)
    crop_hash = imagehash.phash(Image.fromarray(rgb))
    crop_hist = _histogram(rgb)
    crop_orb = _orb(gray)
    rankings: list[dict] = []

    for ref in library:
        mask = ref.alpha.astype(np.float32) / 255.0
        mask_3d = mask[:, :, None]
        corner_pixels = np.concatenate(
            (rgb[:12, :12], rgb[:12, -12:], rgb[-12:, :12], rgb[-12:, -12:]),
            axis=0,
        )
        background = np.median(corner_pixels.reshape(-1, 3), axis=0)
        composite = (ref.rgb * mask_3d + background * (1.0 - mask_3d)).astype(np.uint8)
        composite_for_compare = _badge_neutralized(composite)
        composite_gray = cv2.cvtColor(composite_for_compare, cv2.COLOR_RGB2GRAY)
        composite_hash = imagehash.phash(Image.fromarray(composite_for_compare))
        composite_hist = _histogram(composite_for_compare)

        hash_score = 1.0 - (crop_hash - composite_hash) / 64.0
        hist_score = max(0.0, cv2.compareHist(crop_hist, composite_hist, cv2.HISTCMP_CORREL))
        ssim_score = max(0.0, structural_similarity(gray, composite_gray, data_range=255))
        orb_score = _orb_score(crop_orb, ref.orb_descriptors)
        foreground = mask >= 0.35
        badge_h = round(MATCH_SIZE * 0.24)
        badge_w = round(MATCH_SIZE * 0.24)
        foreground[:badge_h, MATCH_SIZE - badge_w:MATCH_SIZE] = False
        if np.any(foreground):
            color_error = np.mean(np.abs(rgb.astype(np.float32)[foreground] - ref.rgb.astype(np.float32)[foreground]))
            foreground_score = max(0.0, 1.0 - color_error / 110.0)
        else:
            foreground_score = 0.0
        raw = (
            0.22 * hash_score
            + 0.15 * hist_score
            + 0.18 * ssim_score
            + 0.15 * orb_score
            + 0.30 * foreground_score
        )

        # Make high confidence deliberately strict; uncertain layout/crop differences
        # should be reviewed rather than presented as certain.
        confidence = float(np.clip((raw - 0.18) / 0.72, 0.0, 1.0))
        rankings.append(
            {
                "name": ref.name,
                "confidence": confidence,
                "hash": hash_score,
                "histogram": hist_score,
                "ssim": ssim_score,
                "orb": orb_score,
                "foreground": foreground_score,
            }
        )
    return sorted(rankings, key=lambda item: item["confidence"], reverse=True)
