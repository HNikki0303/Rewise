import json
import os
import uuid
from datetime import datetime

INDEX_PATH = "sessions_index.json"
DATA_DIR = "sessions_data"


# ---------------------------------------------------------------------------
# Lightweight index — just enough to render the sidebar list quickly
# ---------------------------------------------------------------------------

def _load_index() -> list:
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(index: list) -> None:
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def list_sessions() -> list:
    """Return all saved sessions, newest first — for populating a picker UI."""
    return _load_index()


def delete_session(session_id: str) -> None:
    index = [r for r in _load_index() if r["session_id"] != session_id]
    _save_index(index)

    data_path = os.path.join(DATA_DIR, f"{session_id}.json")
    if os.path.exists(data_path):
        os.remove(data_path)


# ---------------------------------------------------------------------------
# Full session data — transcript, summary, extracted items, chat history.
# One JSON file per session, so resuming restores everything, not just the
# ability to chat with an empty history.
# ---------------------------------------------------------------------------

def create_session(title: str) -> str:
    """
    Register a new session (one per processed recording) and return its
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


def save_session_data(session_id: str, data: dict) -> None:
    """
    Persist everything about a session — transcript, summary, action items,
    key decisions, open questions, chat history — to disk. Call this after
    processing a new recording, and again after every chat exchange so the
    conversation survives a resume.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    data_path = os.path.join(DATA_DIR, f"{session_id}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session_data(session_id: str) -> dict | None:
    """Load everything saved for a session, or None if nothing was ever saved."""
    data_path = os.path.join(DATA_DIR, f"{session_id}.json")
    if not os.path.exists(data_path):
        return None
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)