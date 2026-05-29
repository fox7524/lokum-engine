from __future__ import annotations

from pathlib import Path
from typing import Iterable

from lokum_engine.paths import ensure_dir, models_dir


def _safe_repo_dirname(repo_id: str) -> str:
    """
    Map a HuggingFace repo_id into a filesystem-safe directory name.

    Requirement (Task 5):
      - Replace "/" with "__"
    """
    return repo_id.replace("/", "__")


def download_snapshot(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
    force_download: bool = False,
) -> Path:
    """
    Download a HuggingFace repo snapshot under LokumAI's configured models_dir().

    This wraps `huggingface_hub.snapshot_download` but forces storage location to:
      models_dir() / safe_repo_dirname(repo_id)

    Returns:
      Path to the local directory containing the snapshot files.
    """
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string")
    repo_id = repo_id.strip()

    target_dir = ensure_dir(models_dir()) / _safe_repo_dirname(repo_id)
    ensure_dir(target_dir)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to download model snapshots"
        ) from e

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns is not None else None,
        force_download=force_download,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )

    return target_dir

