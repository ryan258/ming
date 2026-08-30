from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.output_store import (
    IterationStore,
    OutputStoreError,
    resolve_output_root,
    resolve_workspace_dir,
)


class IterationStoreTests(unittest.TestCase):
    def test_iteration_is_atomic_private_and_human_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = IterationStore(Path(tmp) / "outputs" / "iterations")
            path = store.write("run-abc123", {"status": "completed", "step": 2})
            payload = json.loads(path.read_text(encoding="utf-8"))
            mode = path.stat().st_mode & 0o777
        self.assertEqual(payload, {"status": "completed", "step": 2})
        self.assertEqual(mode, 0o600)

    def test_iteration_id_cannot_collapse_to_empty_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = IterationStore(Path(tmp))
            with self.assertRaises(OutputStoreError):
                store.write("///", {"status": "failed"})

    def test_output_root_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            with self.assertRaisesRegex(OutputStoreError, "MING_OUTPUT_DIR"):
                resolve_output_root(workspace.parent / "escape", workspace)

    def test_data_dir_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            with self.assertRaisesRegex(OutputStoreError, "MING_DATA_DIR"):
                resolve_workspace_dir(
                    workspace.parent / "escape", workspace, env_name="MING_DATA_DIR"
                )


if __name__ == "__main__":
    unittest.main()
