"""
Centralized path management for LokumAI (engine package).

This is a direct port of the repo-level `lokum_paths.py` so the standalone
`lokum-engine` library can own its persistence + cache conventions.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def lokumai_home() -> Path:
    """
    Base persistence directory for LokumAI.

    Override with:
      - LOKUMAI_HOME=/custom/path
    """
    raw = (os.environ.get("LOKUMAI_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".lokumai").expanduser().resolve()


def rag_dir() -> Path:
    """
    RAG persistent store directory.

    Override with:
      - LOKUMAI_RAG_DIR=/custom/path
    """
    raw = (os.environ.get("LOKUMAI_RAG_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumai_home() / "rag"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def lora_dir() -> Path:
    """
    LoRA artifacts root (datasets, adapters, configs).

    Override with:
      - LOKUMAI_LORA_DIR=/custom/path

    Default:
      ~/.lokumai/lora_data
    """
    raw = (os.environ.get("LOKUMAI_LORA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumai_home() / "lora_data"


def models_dir() -> Path:
    """
    HuggingFace downloads / model cache directory used by LokumAI.

    Override with:
      - LOKUMAI_MODELS_DIR=/custom/path

    Default:
      ~/.lokumai/models
    """
    raw = (os.environ.get("LOKUMAI_MODELS_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumai_home() / "models"


def chat_db_path() -> Path:
    """
    Default chat history DB path.

    Override with:
      - LOKUMAI_CHAT_DB=/custom/path/app.db
    """
    raw = (os.environ.get("LOKUMAI_CHAT_DB") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumai_home() / "app.db"


def dev_password_file() -> Path:
    """
    Dev password storage file (local-only).

    Override with:
      - LOKUMAI_DEV_PASSWORD_FILE=/custom/path/dev_password.txt
    """
    raw = (os.environ.get("LOKUMAI_DEV_PASSWORD_FILE") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return lokumai_home() / "dev_password.txt"


def get_or_create_dev_password() -> tuple[str, bool, Path]:
    """
    Returns: (password, generated_now, location)

    Priority:
    1) env LOKUMAI_DEV_PASSWORD
    2) dev_password_file() contents
    3) generate random password, write to dev_password_file()
    """
    env_pw = (os.environ.get("LOKUMAI_DEV_PASSWORD") or "").strip()
    if env_pw:
        return env_pw, False, dev_password_file()

    fp = dev_password_file()
    try:
        if fp.is_file():
            pw = fp.read_text(encoding="utf-8", errors="ignore").strip()
            if pw:
                return pw, False, fp
    except Exception:
        pass

    pw = secrets.token_urlsafe(12)
    try:
        ensure_dir(fp.parent)
        fp.write_text(pw + "\n", encoding="utf-8")
        try:
            os.chmod(str(fp), 0o600)
        except Exception:
            pass
    except Exception:
        # If we can't persist, still return a password (session-only).
        return pw, True, fp
    return pw, True, fp

