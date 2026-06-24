"""Streamlit interface for detecting and identifying Genshin character avatars."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from detector import detect_avatar_regions
from matcher import display_name, load_avatar_library, match_crop
from result_export import build_excel_export_dataframe
from usage_ocr import create_ocr_engine, recognize_usage_rate
from avatar_updater import update_avatar_library


ROOT = Path(__file__).resolve().parent
AVATAR_DIR = ROOT / "avatars"
OUTPUT_DIR = ROOT / "output"
CROP_DIR = OUTPUT_DIR / "crops"
REVIEW_THRESHOLD = 0.85

st.set_page_config(page_title="原神角色头像识别", page_icon="🔎", layout="wide")
st.title("原神角色头像识别")
st.caption("上传截图 → 检查、修改头像框 → 开始识别 → 下载四列 Excel")


@st.cache_resource(show_spinner="正在读取标准头像库……")
def get_library():
    return load_avatar_library(AVATAR_DIR)


@st.cache_resource(show_spinner="正在准备使用率识别……")
def get_ocr_engine():
    return create_ocr_engine()


def sanitize_boxes(rows: pd.DataFrame, image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Convert editable table rows to valid boxes inside the image."""
    image_width, image_height = image_size
    boxes = []
    for _, row in rows.iterrows():
        try:
            x, y = max(0, int(row["X"])), max(0, int(row["Y"]))
            width, height = int(row["宽"]), int(row["高"])
        except (TypeError, ValueError):
            continue
        width = min(width, image_width - x)
        height = min(height, image_height - y)
        if width >= 12 and height >= 12:
            boxes.append((x, y, width, height))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def boxes_dataframe(boxes: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"X": x, "Y": y, "宽": width, "高": height} for x, y, width, height in boxes],
        columns=["X", "Y", "宽", "高"],
    )


def make_preview(image: Image.Image, boxes, highlight_box=None) -> Image.Image:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    line_width = max(2, image.width // 400)
    for index, (x, y, width, height) in enumerate(boxes, 1):
        draw.rectangle((x, y, x + width, y + height), outline="#00ff66", width=line_width)
        draw.text((x + 3, y + 3), str(index), fill="#ff3030", stroke_fill="white", stroke_width=2)
    if highlight_box:
        x, y, width, height = highlight_box
        draw.rectangle((x, y, x + width, y + height), outline="#00aaff", width=line_width * 2)
    return preview


with st.sidebar:
    st.header("头像库")
    local_avatar_count = len([p for p in AVATAR_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}])
    st.caption(f"当前本地头像：{local_avatar_count} 个")
    if st.button("联网更新新角色头像", use_container_width=True):
        try:
            with st.spinner("正在更新头像库……"):
                summary = update_avatar_library(AVATAR_DIR)
            get_library.clear()
            st.success(f"头像库更新完成：新增 {summary['downloaded']} 个，跳过 {summary['skipped']} 个。")
            if summary["failed"]:
                st.warning(f"有 {len(summary['failed'])} 个头像未能下载，可稍后再试或手动补充。")
        except Exception as exc:
            st.error(f"头像库更新失败：{exc}")
            st.info("可以手动把新角色头像放进 avatars 文件夹，文件名使用角色中文名。")


uploaded = st.file_uploader("选择一张截图", type=["png", "jpg", "jpeg", "webp", "bmp"])
if uploaded is None:
    st.info("请先上传截图。自动检测失败时，也可以在表格中手动添加头像框。")
    st.stop()

file_bytes = uploaded.getvalue()
file_key = hashlib.sha1(file_bytes).hexdigest()
image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
image_rgb = np.asarray(image)

if st.session_state.get("file_key") != file_key:
    with st.spinner("正在自动寻找疑似头像区域……"):
        detected = detect_avatar_regions(image_rgb)
    for key in ("result_df", "result_csv", "result_xlsx", "result_preview"):
        st.session_state.pop(key, None)
    st.session_state.file_key = file_key
    st.session_state.boxes = detected
    st.session_state.table_revision = 0

top_left, top_right = st.columns([3, 1])
with top_left:
    st.success(f"自动检测到 {len(st.session_state.boxes)} 个疑似头像。绿色框可以在下方表格中修改。")
with top_right:
    if st.button("重新自动检测", use_container_width=True):
        st.session_state.boxes = detect_avatar_regions(image_rgb)
        st.session_state.table_revision += 1
        st.rerun()

st.image(make_preview(image, st.session_state.boxes), caption="当前头像框预览", use_container_width=True)

st.subheader("检查与调整头像框")
st.markdown(
    "直接修改表格中的 **X、Y、宽、高**。删除一行就是删除误检框；点击表格底部的 `+` 可补充漏检框。"
    "修改完成后点击“应用表格修改并刷新预览”。"
)

edited_rows = st.data_editor(
    boxes_dataframe(st.session_state.boxes),
    num_rows="dynamic",
    hide_index=False,
    use_container_width=True,
    key=f"box_editor_{file_key}_{st.session_state.table_revision}",
    column_config={
        "X": st.column_config.NumberColumn("X（左边位置）", min_value=0, step=1, required=True),
        "Y": st.column_config.NumberColumn("Y（顶部位置）", min_value=0, step=1, required=True),
        "宽": st.column_config.NumberColumn("宽", min_value=12, step=1, required=True),
        "高": st.column_config.NumberColumn("高", min_value=12, step=1, required=True),
    },
)
confirmed_boxes = sanitize_boxes(edited_rows, image.size)

if st.button("应用表格修改并刷新预览", use_container_width=True):
    st.session_state.boxes = confirmed_boxes
    st.session_state.table_revision += 1
    st.rerun()

with st.expander("可视化补充一个头像框"):
    st.caption("用滑块移动和缩放蓝色框，确认框住头像后点击添加。")
    default_size = max(40, round(image.width * 0.14))
    x = st.slider("新框 X（左右）", 0, max(0, image.width - 12), 0)
    y = st.slider("新框 Y（上下）", 0, max(0, image.height - 12), 0)
    width = st.slider("新框宽度", 12, max(12, min(image.width - x, image.width // 2)), min(default_size, image.width - x))
    height = st.slider("新框高度", 12, max(12, min(image.height - y, image.height // 2)), min(default_size, image.height - y))
    new_box = (x, y, width, height)
    st.image(make_preview(image, confirmed_boxes, new_box), caption="蓝色框是即将添加的新框", use_container_width=True)
    if st.button("添加这个蓝色框", use_container_width=True):
        st.session_state.boxes = sorted(confirmed_boxes + [new_box], key=lambda box: (box[1], box[0]))
        st.session_state.table_revision += 1
        st.rerun()

if st.button("开始识别并导出结果", type="primary", disabled=not confirmed_boxes, use_container_width=True):
    library = get_library()
    ocr_engine = get_ocr_engine()
    if not library:
        st.error("avatars 文件夹中没有可读取的标准头像图片。")
        st.stop()

    OUTPUT_DIR.mkdir(exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    progress = st.progress(0, text="准备识别……")
    safe_stem = Path(uploaded.name).stem

    for index, (x, y, width, height) in enumerate(confirmed_boxes, 1):
        crop = image_rgb[y:y + height, x:x + width]
        crop_path = CROP_DIR / f"{safe_stem}_{index:03d}.png"
        Image.fromarray(crop).save(crop_path)
        ranking = match_crop(crop, library)
        usage_value, usage_text, usage_ocr_confidence = recognize_usage_rate(
            image_rgb, (x, y, width, height), ocr_engine
        )
        top = ranking[0]
        alternatives = ranking[1:3]
        results.append(
            {
                "序号": index,
                "原图顺序": index,
                "角色名": display_name(top["name"]),
                "置信度": round(top["confidence"], 4),
                "使用率数字": usage_value,
                "使用率文本": usage_text,
                "使用率识别置信度": round(usage_ocr_confidence, 4),
                "备选角色1": display_name(alternatives[0]["name"]) if len(alternatives) > 0 else "",
                "备选角色2": display_name(alternatives[1]["name"]) if len(alternatives) > 1 else "",
                "截图文件名": uploaded.name,
                "头像框坐标": json.dumps([x, y, width, height], ensure_ascii=False),
                "是否需要复核": (
                    "是"
                    if top["confidence"] < REVIEW_THRESHOLD
                    or usage_value is None
                    or usage_ocr_confidence < 0.80
                    else "否"
                ),
            }
        )
        progress.progress(index / len(confirmed_boxes), text=f"正在识别 {index}/{len(confirmed_boxes)}")

    result_df = pd.DataFrame(results)
    export_df = build_excel_export_dataframe(result_df)
    csv_path = OUTPUT_DIR / "result.csv"
    xlsx_path = OUTPUT_DIR / "result.xlsx"
    export_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    export_df.to_excel(xlsx_path, index=False)
    progress.empty()
    st.session_state.result_df = result_df
    st.session_state.export_df = export_df
    st.session_state.result_csv = csv_path.read_bytes()
    st.session_state.result_xlsx = xlsx_path.read_bytes()
    st.session_state.result_preview = make_preview(image, confirmed_boxes)
    st.success("识别完成。裁剪头像和结果文件已经保存到 output 文件夹。")

if "result_df" in st.session_state:
    st.subheader("识别结果")
    st.dataframe(
        st.session_state.result_df.style.format(
            {"置信度": "{:.1%}", "使用率识别置信度": "{:.1%}"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Excel 导出预览")
    st.dataframe(st.session_state.export_df, hide_index=True, use_container_width=True)
    st.image(st.session_state.result_preview, caption="最终确认的头像框", use_container_width=True)
    download_left, download_right = st.columns(2)
    download_left.download_button(
        "下载 Excel 结果",
        st.session_state.result_xlsx,
        file_name="result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    download_right.download_button(
        "下载 CSV 结果",
        st.session_state.result_csv,
        file_name="result.csv",
        mime="text/csv",
        use_container_width=True,
    )
