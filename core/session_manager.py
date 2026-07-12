import json
import os
import uuid
from datetime import datetime

INDEX_PATH = "sessions_index.json"


def _load_index() -> list:
    """Return the list of all saved session records, newest first."""
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(index: list) -> None:
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def create_session(title: str) -> str:
    """
    Register a new session (one per processed video/meeting) and return its
    unique session_id. Call this once, right after a transcript is produced.
    """
    session_id = uuid.uuid4().hex[:12]  # short, unique, filesystem/collection-name safe

    index = _load_index()
    index.insert(0, {
        "session_id": session_id,
        "title": title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save_index(index)

    return session_id


def list_sessions() -> list:
    """Return all saved sessions, newest first — for populating a picker UI."""
    return _load_index()


def get_session(session_id: str) -> dict | None:
    for record in _load_index():
        if record["session_id"] == session_id:
            return record
    return None


def delete_session(session_id: str) -> None:
    index = [r for r in _load_index() if r["session_id"] != session_id]
    _save_index(index)