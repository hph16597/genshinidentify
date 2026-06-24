from pathlib import Path
import sys
import unittest

import pandas as pd
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector import detect_avatar_regions
from result_export import build_excel_export_dataframe
from usage_ocr import normalize_usage_rate_text


SAMPLE_DIR = Path(r"C:\Users\hph16\Pictures")
SAMPLES = {
    "teyvat_short": SAMPLE_DIR / "微信图片_2026-05-19_092956_314.jpg",
    "stygian_long": SAMPLE_DIR / "387CCD4CD6A707D8D768F767B0DDC799.jpg",
    "teyvat_long": SAMPLE_DIR / "E1F3EB5A0348FDBE4071D7B68FA34C63.png",
}


def _load_rgb(path: Path) -> np.ndarray:
    assert path.exists(), f"样例图不存在：{path}"
    return np.asarray(Image.open(path).convert("RGB"))


class UsageRateWorkflowTests(unittest.TestCase):
    def test_normalize_usage_rate_accepts_common_percent_formats(self):
        cases = {
            "75%": (75.0, "75%"),
            "75.0%": (75.0, "75%"),
            "97.9 %": (97.9, "97.9%"),
            "0.0%": (0.0, "0%"),
            "97,9%": (97.9, "97.9%"),
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize_usage_rate_text(text), expected)

    def test_build_excel_export_dataframe_has_four_sorted_columns(self):
        source = pd.DataFrame(
            [
                {"角色名": "低", "使用率数字": 12.8, "使用率文本": "12.8%", "原图顺序": 2},
                {"角色名": "高", "使用率数字": 87.4, "使用率文本": "87.4%", "原图顺序": 1},
                {"角色名": "并列后", "使用率数字": 12.8, "使用率文本": "12.8%", "原图顺序": 3},
                {"角色名": "失败", "使用率数字": None, "使用率文本": "", "原图顺序": 4},
            ]
        )

        exported = build_excel_export_dataframe(source)

        self.assertEqual(list(exported.columns), ["排名", "角色中文名", "使用率", "使用率文本"])
        self.assertEqual(exported["角色中文名"].tolist(), ["高", "低", "并列后", "失败"])
        self.assertEqual(exported["排名"].tolist(), [1, 2, 3, 4])
        self.assertEqual(exported["使用率文本"].tolist(), ["87.4%", "12.8%", "12.8%", ""])

    def test_sample_layout_detection_finds_visible_teyvat_short_cards(self):
        boxes = detect_avatar_regions(_load_rgb(SAMPLES["teyvat_short"]))

        self.assertEqual(len(boxes), 38)
        self.assertTrue(all(y > 450 for _, y, _, _ in boxes))

    def test_sample_layout_detection_finds_full_teyvat_long_cards(self):
        boxes = detect_avatar_regions(_load_rgb(SAMPLES["teyvat_long"]))

        self.assertEqual(len(boxes), 116)
        self.assertTrue(all(y > 250 for _, y, _, _ in boxes))

    def test_sample_layout_detection_finds_stygian_cards_without_top_ui(self):
        boxes = detect_avatar_regions(_load_rgb(SAMPLES["stygian_long"]))

        self.assertGreaterEqual(len(boxes), 100)
        self.assertTrue(all(y > 300 for _, y, _, _ in boxes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
