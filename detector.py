"""Detect likely character-avatar regions in a screenshot."""

from __future__ import annotations

import cv2
import numpy as np


Box = tuple[int, int, int, int]


def _cluster(values: list[int], tolerance: int) -> list[int]:
    groups: list[list[int]] = []
    for value in sorted(values):
        if groups and value - int(np.median(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(float(np.median(group))) for group in groups]


def _cluster_with_counts(values: list[int], tolerance: int) -> list[tuple[int, int]]:
    groups: list[list[int]] = []
    for value in sorted(values):
        if groups and value - int(np.median(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [(round(float(np.median(group))), len(group)) for group in groups]


def _avatar_color_mask(image_rgb: np.ndarray, min_saturation: int = 45, min_value: int = 55) -> np.ndarray:
    """Mask the orange/purple portrait panels used behind Genshin avatars."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    return (
        (
            (hsv[:, :, 0] < 35)
            | ((hsv[:, :, 0] > 128) & (hsv[:, :, 0] < 170))
        )
        & (hsv[:, :, 1] > min_saturation)
        & (hsv[:, :, 2] > min_value)
    )


def _complete_missing_regular_grid_rows(image_rgb: np.ndarray, boxes: list[Box]) -> list[Box]:
    """Fill Teyvat-style missing rows inferred from large gaps in an otherwise regular grid."""
    if len(boxes) < 8:
        return boxes

    height, width = image_rgb.shape[:2]
    sizes = [round((box[2] + box[3]) / 2) for box in boxes]
    size = round(float(np.median(sizes)))
    if size <= 0:
        return boxes

    x_positions = _cluster([box[0] for box in boxes], tolerance=max(5, round(size * 0.10)))
    y_positions = _cluster([box[1] for box in boxes], tolerance=max(5, round(size * 0.10)))
    if len(x_positions) < 2 or len(y_positions) < 2:
        return boxes

    diffs = [b - a for a, b in zip(y_positions, y_positions[1:])]
    regular_diffs = [diff for diff in diffs if size * 1.05 <= diff <= size * 1.70]
    if not regular_diffs:
        return boxes
    row_pitch = round(float(np.median(regular_diffs)))
    if row_pitch <= size:
        return boxes

    mask = _avatar_color_mask(image_rgb)
    additions: list[Box] = []
    existing = list(boxes)

    for previous_y, next_y in zip(y_positions, y_positions[1:]):
        if next_y - previous_y <= row_pitch * 1.45:
            continue
        candidate_y = previous_y + row_pitch
        while candidate_y < next_y - row_pitch * 0.45:
            candidate_boxes: list[Box] = []
            for x in x_positions:
                if x + size > width or candidate_y + size > height:
                    continue
                region = mask[candidate_y:candidate_y + size, x:x + size]
                if region.size == 0:
                    continue
                if float(np.mean(region)) >= 0.18:
                    candidate_boxes.append((int(x), int(candidate_y), int(min(size, width - x)), int(min(size, height - candidate_y))))
            if len(candidate_boxes) >= 2:
                additions.extend(candidate_boxes)
            candidate_y += row_pitch

    if not additions:
        return boxes
    return sorted(_deduplicate(existing + additions, threshold=0.65), key=lambda box: (box[1], box[0]))


def _detect_regular_avatar_grid(image_rgb: np.ndarray) -> list[Box]:
    """Detect repeated rounded-square avatar tiles and complete their grid."""
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    anchors: list[Box] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        ratio = box_width / max(box_height, 1)
        fill = cv2.contourArea(contour) / max(box_width * box_height, 1)
        if (
            width * 0.10 <= box_width <= width * 0.20
            and 0.88 <= ratio <= 1.12
            and fill >= 0.82
        ):
            anchors.append((x, y, box_width, box_height))

    anchors = _deduplicate(anchors, threshold=0.65)
    if len(anchors) < 4:
        return []

    sizes = [round((box[2] + box[3]) / 2) for box in anchors]
    size = round(float(np.median(sizes)))
    anchors = [box for box in anchors if abs(((box[2] + box[3]) / 2) - size) <= size * 0.08]
    if len(anchors) < 4:
        return []

    tolerance = max(5, round(size * 0.10))
    x_positions = _cluster([box[0] for box in anchors], tolerance)
    y_positions = _cluster([box[1] for box in anchors], tolerance)
    if len(x_positions) < 2:
        return []

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    boxes: list[Box] = []
    inset = max(5, round(size * 0.12))
    for y in y_positions:
        for x in x_positions:
            if x + size > width + tolerance or y + size > height + tolerance:
                continue
            x2, y2 = min(width, x + size), min(height, y + size)
            inside = hsv[min(height, y + inset):max(0, y2 - inset), min(width, x + inset):max(0, x2 - inset)]
            if inside.size == 0:
                continue
            # Avatar tiles are colorful and bright; empty grid positions use the near-black page background.
            mean_saturation = float(np.mean(inside[:, :, 1]))
            mean_value = float(np.mean(inside[:, :, 2]))
            if mean_value >= 48 and (mean_saturation >= 30 or mean_value >= 90):
                boxes.append((max(0, x), max(0, y), min(size, width - x), min(size, height - y)))
    return _complete_missing_regular_grid_rows(image_rgb, sorted(boxes, key=lambda box: (box[1], box[0])))


def _detect_stygian_compact_grid(image_rgb: np.ndarray) -> list[Box]:
    """Detect compact Stygian-style character cards by their orange/purple avatar tiles."""
    height, width = image_rgb.shape[:2]
    if height < width * 1.8:
        return []

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    y_index = np.indices((height, width))[0]
    top_cutoff = max(280, round(height * 0.055))

    # Stygian compact cards use orange 5-star and purple 4-star portrait panels.
    colored_card_mask = (_avatar_color_mask(image_rgb, 60, 70) & (y_index > top_cutoff)).astype("uint8") * 255

    colored_card_mask = cv2.morphologyEx(
        colored_card_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    colored_card_mask = cv2.morphologyEx(
        colored_card_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
    )

    contours, _ = cv2.findContours(colored_card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes: list[Box] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (
            width * 0.035 <= box_width <= width * 0.16
            and width * 0.035 <= box_height <= width * 0.16
            and area >= width * width * 0.00055
        ):
            contour_boxes.append((int(x), int(y), int(box_width), int(box_height)))

    if len(contour_boxes) < 18:
        return []

    tall_boxes = [box for box in contour_boxes if box[3] >= width * 0.07]
    if not tall_boxes:
        return []

    median_height = round(float(np.median([box[3] for box in tall_boxes])))
    x_clusters = _cluster_with_counts(
        [box[0] for box in tall_boxes],
        tolerance=max(18, round(median_height * 0.26)),
    )
    if len(x_clusters) < 4:
        return []

    max_x_support = max(count for _, count in x_clusters)
    x_positions = [
        x for x, count in x_clusters
        if count >= max(3, round(max_x_support * 0.60))
    ]
    x_positions = sorted(x_positions)
    if len(x_positions) < 4:
        return []

    if len(x_positions) > 8:
        strongest = sorted(
            ((count, x) for x, count in x_clusters if x in x_positions),
            reverse=True,
        )[:8]
        x_positions = sorted(x for _, x in strongest)

    y_positions = [
        y for y, _ in _cluster_with_counts(
            [box[1] for box in tall_boxes],
            tolerance=max(20, round(median_height * 0.22)),
        )
    ]
    if len(y_positions) < 3:
        return []

    if len(x_positions) >= 2:
        pitch = float(np.median(np.diff(x_positions)))
        wide_widths = [box[2] for box in tall_boxes if box[2] >= pitch * 0.60]
        tile_width = round(float(np.median(wide_widths))) if wide_widths else round(pitch * 0.86)
    else:
        tile_width = round(float(np.median([box[2] for box in tall_boxes])))
    tile_height = median_height
    tile_width = max(36, min(tile_width, round(width * 0.18)))
    tile_height = max(36, min(tile_height, round(width * 0.18)))

    boxes: list[Box] = []
    binary_mask = colored_card_mask.astype(bool)
    for y in y_positions:
        if y <= top_cutoff or y + tile_height >= height - 20:
            continue
        for x in x_positions:
            if x + tile_width >= width + 4:
                continue
            region = binary_mask[y:min(height, y + tile_height), x:min(width, x + tile_width)]
            if region.size == 0:
                continue
            if float(np.mean(region)) >= 0.16:
                boxes.append((int(x), int(y), int(min(tile_width, width - x)), int(min(tile_height, height - y))))

    boxes = _deduplicate(boxes, threshold=0.35)
    return sorted(boxes, key=lambda box: (box[1], box[0])) if len(boxes) >= 24 else []


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _deduplicate(boxes: list[Box], threshold: float = 0.45) -> list[Box]:
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[Box] = []
    for box in boxes:
        if not any(_iou(box, old) > threshold for old in kept):
            kept.append(box)
    return sorted(kept, key=lambda b: (b[1], b[0]))


def _prefer_repeated_sizes(boxes: list[Box], limit: int = 80) -> list[Box]:
    """Rank repeated, similarly sized UI tiles ahead of incidental square edges."""
    if len(boxes) <= 20:
        return boxes
    scored = []
    for box in boxes:
        size = (box[2] + box[3]) / 2
        support = sum(
            1
            for other in boxes
            if abs(((other[2] + other[3]) / 2) - size) <= max(5, size * 0.13)
        )
        squareness = 1.0 - abs(box[2] - box[3]) / max(box[2], box[3])
        scored.append((support + squareness, box))
    chosen = [box for _, box in sorted(scored, reverse=True)[:limit]]
    return sorted(chosen, key=lambda b: (b[1], b[0]))


def detect_avatar_regions(image_rgb: np.ndarray) -> list[Box]:
    """Return square-ish candidate boxes using contours and circle detection."""
    grid_boxes = _detect_regular_avatar_grid(image_rgb)
    if len(grid_boxes) >= 4:
        return grid_boxes

    stygian_boxes = _detect_stygian_compact_grid(image_rgb)
    if len(stygian_boxes) >= 24:
        return stygian_boxes

    height, width = image_rgb.shape[:2]
    short_side = min(height, width)
    min_size = max(32, int(short_side * 0.035))
    max_size = int(short_side * 0.32)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    candidates: list[Box] = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        ratio = w / max(h, 1)
        area_ratio = cv2.contourArea(contour) / max(w * h, 1)
        if min_size <= w <= max_size and min_size <= h <= max_size and 0.72 <= ratio <= 1.38:
            if area_ratio >= 0.18:
                pad = max(1, round(min(w, h) * 0.02))
                candidates.append((max(0, x - pad), max(0, y - pad), min(width - x, w + 2 * pad), min(height - y, h + 2 * pad)))

    # Circular portraits often have no strong square border, so search for them too.
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.25,
        minDist=min_size,
        param1=100,
        param2=36,
        minRadius=min_size // 2,
        maxRadius=max_size // 2,
    )
    if circles is not None:
        for cx, cy, radius in np.round(circles[0]).astype(int):
            size = radius * 2
            x, y = max(0, cx - radius), max(0, cy - radius)
            candidates.append((x, y, min(size, width - x), min(size, height - y)))

    return _prefer_repeated_sizes(_deduplicate(candidates, threshold=0.35))
