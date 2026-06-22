"""Shared storage paths for local runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def db_dir() -> Path:
    return Path(os.getenv("DB_DIR", "data/db"))


def default_db_path(filename: str) -> str:
    return str(db_dir() / filename)


def env_db_path(env_name: str, filename: str) -> str:
    return os.getenv(env_name, default_db_path(filename))


def ensure_parent(path: str) -> None:
    parent = Path(path).expanduser().parent
    if parent != Path("."):
        parent.mkdir(parents=True, exist_ok=True)