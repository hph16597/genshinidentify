"""Download missing Genshin character avatars into the local avatars folder."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


API_BASE = "https://genshin-db-api.vercel.app/api/v5/"
ENKA_IMAGE_BASE = "https://enka.network/ui/"
ALIAS_FILE = "character_aliases.json"
IMAGE_SUFFIXES = {
    "image/webp": ".webp",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


def _fetch_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": "genshin-avatar-recognizer/1.0"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_bytes(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": "genshin-avatar-recognizer/1.0"})
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
        suffix = IMAGE_SUFFIXES.get(content_type) or Path(urlparse(url).path).suffix or ".webp"
        return response.read(), suffix.lower()


def _characters_url(language: str) -> str:
    return (
        API_BASE
        + "characters?query=names&matchCategories=true&verboseCategories=true&resultLanguage="
        + quote(language)
    )


def _as_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _image_url(record: dict) -> str:
    images = record.get("images") if isinstance(record.get("images"), dict) else {}
    for key in ("mihoyo_icon", "hoyowiki_icon", "image", "icon"):
        value = images.get(key) or record.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    for key in ("filename_icon", "filename_sideIcon"):
        value = images.get(key) or record.get(key)
        if isinstance(value, str) and value:
            return ENKA_IMAGE_BASE + value + ".png"
    return ""


def _safe_name(name: str) -> str:
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name.strip()


def _pair_records(zh_records: list[dict], en_records: list[dict]) -> list[tuple[dict, dict]]:
    en_by_id = {str(item.get("id")): item for item in en_records if item.get("id") is not None}
    pairs = []
    for zh in zh_records:
        en = en_by_id.get(str(zh.get("id")), {})
        pairs.append((zh, en))
    return pairs


def update_avatar_library(avatar_dir: Path) -> dict:
    """Download missing avatars and return a small summary dictionary."""
    avatar_dir.mkdir(parents=True, exist_ok=True)
    zh_records = _as_records(_fetch_json(_characters_url("ChineseSimplified")))
    en_records = _as_records(_fetch_json(_characters_url("English")))
    if not zh_records:
        raise RuntimeError("没有从 genshin-db-api 读取到角色列表")

    downloaded = 0
    skipped = 0
    failed: list[dict] = []
    aliases: dict[str, dict] = {}

    for zh, en in _pair_records(zh_records, en_records):
        name = _safe_name(str(zh.get("name") or "").strip())
        if not name:
            continue
        existing = list(avatar_dir.glob(name + ".*"))
        aliases[name] = {
            "id": zh.get("id"),
            "中文名": name,
            "英文名": en.get("name") or "",
        }
        if existing:
            skipped += 1
            continue

        url = _image_url(en) or _image_url(zh)
        if not url:
            failed.append({"name": name, "reason": "没有头像 URL"})
            continue

        try:
            content, suffix = _download_bytes(url)
            target = avatar_dir / f"{name}{suffix}"
            target.write_bytes(content)
            downloaded += 1
        except Exception as exc:
            failed.append({"name": name, "reason": str(exc)})

    (avatar_dir.parent / ALIAS_FILE).write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    summary = update_avatar_library(root / "avatars")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
