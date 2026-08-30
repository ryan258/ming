from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


class OutputStoreError(ValueError):
    """Raised when an output artifact would violate the local output contract."""


def resolve_workspace_dir(
    directory: Path, workspace_root: Path, *, env_name: str
) -> Path:
    workspace = workspace_root.resolve()
    candidate = directory if directory.is_absolute() else workspace / directory
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise OutputStoreError(
            f"{env_name} must remain inside the current workspace"
        ) from exc
    return resolved


def resolve_output_root(directory: Path, workspace_root: Path) -> Path:
    return resolve_workspace_dir(directory, workspace_root, env_name="MING_OUTPUT_DIR")


class IterationStore:
    """Persist human-inspectable run and preview snapshots as atomic JSON files."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve(strict=False)

    def path_for(self, iteration_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", iteration_id).strip("-")
        if not safe_id:
            raise OutputStoreError("iteration id must contain a filename-safe character")
        return self.directory / f"{safe_id}.json"

    def write(self, iteration_id: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(iteration_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(rendered)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path
