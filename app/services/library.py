"""Saved post kits and their files: SQLite for records, data/media/<kit id>/ for artifacts."""
import io
import json
import re
import sqlite3
import time
import uuid
import zipfile
from pathlib import Path

from app.config import ROOT_DIR

DATA_DIR = ROOT_DIR / "data"
MEDIA_DIR = DATA_DIR / "media"
DB_FILE = DATA_DIR / "trendcraft.db"
STATUSES = ("new", "posted", "archived")
UPDATABLE = {"status", "kit_json", "cover_file", "cover_placeholder", "reel_status", "reel_stage", "reel_progress"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS kits (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    niche TEXT NOT NULL DEFAULT '',
    format TEXT NOT NULL,
    angle_title TEXT NOT NULL,
    why_now TEXT NOT NULL DEFAULT '',
    kit_json TEXT NOT NULL,
    articles_json TEXT NOT NULL,
    style_json TEXT,
    cover_file TEXT,
    cover_placeholder INTEGER NOT NULL DEFAULT 0,
    reel_status TEXT NOT NULL DEFAULT 'none',
    reel_stage TEXT NOT NULL DEFAULT '',
    reel_progress INTEGER NOT NULL DEFAULT 0
)
"""


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def init() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as db:
        db.execute(SCHEMA)
        # A render can't survive a restart, so don't leave it looking like it's still running.
        db.execute(
            "UPDATE kits SET reel_status='failed', reel_stage='Rendering was interrupted. Please try again.' "
            "WHERE reel_status='rendering'"
        )


def valid_id(kit_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", kit_id))


def kit_dir(kit_id: str) -> Path:
    path = MEDIA_DIR / kit_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_kit(kit: dict, title: str, why_now: str, niche: str, articles: list[dict], style: dict | None) -> dict:
    kit_id = uuid.uuid4().hex
    with _connect() as db:
        db.execute(
            "INSERT INTO kits (id, created_at, niche, format, angle_title, why_now, kit_json, articles_json, style_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (kit_id, time.time(), niche, kit["format"], title, why_now, json.dumps(kit),
             json.dumps(articles), json.dumps(style) if style else None),
        )
    return get_kit(kit_id)


def get_row(kit_id: str) -> sqlite3.Row | None:
    if not valid_id(kit_id):
        return None
    with _connect() as db:
        return db.execute("SELECT * FROM kits WHERE id=?", (kit_id,)).fetchone()


def get_kit(kit_id: str) -> dict | None:
    row = get_row(kit_id)
    return _to_api(row) if row else None


def list_kits() -> list[dict]:
    with _connect() as db:
        rows = db.execute("SELECT * FROM kits ORDER BY created_at DESC").fetchall()
    return [_to_api(row) for row in rows]


def update(kit_id: str, **fields) -> None:
    unknown = set(fields) - UPDATABLE
    if unknown:
        raise ValueError(f"not updatable: {unknown}")
    assignments = ", ".join(f"{name}=?" for name in fields)
    with _connect() as db:
        db.execute(f"UPDATE kits SET {assignments} WHERE id=?", (*fields.values(), kit_id))


def update_kit_fields(kit_id: str, **fields) -> None:
    row = get_row(kit_id)
    update(kit_id, kit_json=json.dumps({**json.loads(row["kit_json"]), **fields}))


def _artifact_kind(name: str) -> str:
    if name.startswith("scenes/"):
        return "scene"
    return {"cover": "cover", "reel": "reel"}.get(Path(name).stem, "file")


def artifacts(kit_id: str) -> list[dict]:
    folder = MEDIA_DIR / kit_id
    if not folder.exists():
        return []
    files = sorted(p for p in folder.rglob("*") if p.is_file() and ".tmp" not in p.name)
    names = [p.relative_to(folder).as_posix() for p in files]
    return [{"name": n, "url": f"/media/{kit_id}/{n}", "kind": _artifact_kind(n)} for n in names]


def _to_api(row: sqlite3.Row) -> dict:
    kit_id = row["id"]
    reel_ready = row["reel_status"] == "ready"
    return {
        **json.loads(row["kit_json"]),
        "id": kit_id,
        "created_at": row["created_at"],
        "status": row["status"],
        "niche": row["niche"],
        "angle_title": row["angle_title"],
        "why_now": row["why_now"],
        "styled": bool(row["style_json"]),
        "cover_url": f"/media/{kit_id}/{row['cover_file']}" if row["cover_file"] else None,
        "cover_placeholder": bool(row["cover_placeholder"]),
        "reel_status": row["reel_status"],
        "reel_stage": row["reel_stage"],
        "reel_progress": row["reel_progress"],
        "reel_url": f"/media/{kit_id}/reel.mp4" if reel_ready else None,
        "artifacts": artifacts(kit_id),
    }


def kit_markdown(kit: dict) -> str:
    captions = kit["captions"]
    lines = [f"# {kit['angle_title']}", "", f"Format: {kit['format']}", "", "## Hooks"]
    lines += [f"- {hook}" for hook in kit["hooks"]]
    lines += ["", "## Captions", "", "### Informative", captions["informative"], "", "### Witty", captions["witty"],
              "", "### Hot take", captions["hot_take"], "", "## Content"]
    for item in kit["script_or_slides"]:
        lines += ["", f"### {item['label']}"]
        if item.get("on_screen_text"):
            lines.append(f"On screen: {item['on_screen_text']}")
        lines.append(item["text"])
    lines += ["", "## Hashtags", " ".join(kit["hashtags"]), "", "## Tips"]
    lines += [f"- {tip}" for tip in kit["tips"]]
    lines += ["", "## Fact-check", f"Result: {kit['fact_check']}"]
    lines += [f"- Unsupported: {claim}" for claim in kit["unsupported_claims"]]
    lines += ["", "## Sources"]
    lines += [f"- [{s['title']}]({s['url']})" for s in kit["sources"]]
    return "\n".join(lines) + "\n"


def bundle(kit: dict) -> bytes:
    """Zip the kit text and every generated file."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("kit.md", kit_markdown(kit))
        for artifact in kit["artifacts"]:
            archive.write(MEDIA_DIR / kit["id"] / artifact["name"], artifact["name"])
    return buffer.getvalue()
