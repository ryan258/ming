from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from engine.config import ConfigurationError, Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_free_only_and_end_in_dynamic_router(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env(None)
        for role in ("orchestrator", "coder", "fast"):
            models = settings.models_for(role)
            self.assertEqual(models[-1], "openrouter/free")
            self.assertTrue(all(m == "openrouter/free" or m.endswith(":free") for m in models))

    def test_default_named_routes_match_verified_v1_catalog(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env(None)
        self.assertEqual(settings.orchestrator_models[0], "z-ai/glm-5.2:free")
        self.assertEqual(settings.coder_models[0], "poolside/laguna-s-2.1:free")
        self.assertEqual(settings.fast_models[0], "minimax/minimax-m3:free")

    def test_paid_model_is_rejected(self) -> None:
        with patch.dict(
            os.environ, {"MING_CODER_MODELS": "openai/gpt-5"}, clear=True
        ), self.assertRaisesRegex(ConfigurationError, "non-free"):
            Settings.from_env(None)

    def test_custom_origin_requires_explicit_opt_in(self) -> None:
        with patch.dict(
            os.environ, {"OPENROUTER_BASE_URL": "https://example.test/v1"}, clear=True
        ), self.assertRaisesRegex(ConfigurationError, "requires"):
            Settings.from_env(None)

    def test_custom_origin_can_be_explicitly_enabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_BASE_URL": "https://example.test/v1/",
                "MING_ALLOW_CUSTOM_BASE_URL": "true",
            },
            clear=True,
        ):
            settings = Settings.from_env(None)
        self.assertEqual(settings.base_url, "https://example.test/v1")

    def test_dotenv_does_not_override_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("MING_HUMAN_RATE=50\nOPENROUTER_API_KEY=file-key\n")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "shell-key"}, clear=True):
                settings = Settings.from_env(env_file)
        self.assertEqual(settings.api_key, "shell-key")
        self.assertEqual(settings.human_rate, 50)

    def test_dotenv_strips_unquoted_inline_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("OPENROUTER_API_KEY=sk-or-v1-xxx # operator key\n")
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_file)
        self.assertEqual(settings.api_key, "sk-or-v1-xxx")

    def test_dotenv_keeps_hash_inside_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text('OPENROUTER_API_KEY="sk-or-v1-xxx # keep"\n')
            with patch.dict(os.environ, {}, clear=True):
                settings = Settings.from_env(env_file)
        self.assertEqual(settings.api_key, "sk-or-v1-xxx # keep")

    def test_public_status_never_contains_key(self) -> None:
        settings = Settings(api_key="super-secret")
        rendered = repr(settings.public_status())
        self.assertNotIn("super-secret", rendered)
        self.assertIn("api_key_configured", rendered)

    def test_image_route_is_separately_labeled_as_paid_opt_in(self) -> None:
        settings = Settings()
        image = settings.public_status()["image_generation"]
        self.assertEqual(image["models"], ["krea/krea-2-medium-turbo"])
        self.assertEqual(image["estimated_cost_usd"], 0.015)
        self.assertEqual(image["model_estimates"], {"krea/krea-2-medium-turbo": 0.015})
        self.assertTrue(image["paid_opt_in_required"])
        self.assertEqual(image["output_dir"], "outputs/images")

    def test_multiple_paid_image_models_require_per_model_estimates(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "MING_IMAGE_MODEL_COSTS"):
            Settings(
                image_models=("cheap/image", "expensive/image"),
                image_estimated_cost_usd=0.015,
            )

    def test_per_model_image_costs_bind_to_each_paid_model(self) -> None:
        settings = Settings(
            image_models=("cheap/image", "expensive/image"),
            image_model_estimates={"cheap/image": 0.015, "expensive/image": 0.50},
        )
        self.assertEqual(settings.estimate_for_image_model("cheap/image"), 0.015)
        self.assertEqual(settings.estimate_for_image_model("expensive/image"), 0.50)

    def test_doctor_does_not_claim_live_readiness_from_key_presence(self) -> None:
        from ming.cli import command_doctor

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "not-a-real-key"},
            clear=True,
        ):
            settings = Settings.from_env(None)
            previous = Path.cwd()
            os.chdir(tmp)
            try:
                buffer = StringIO()
                with patch("sys.stdout", buffer):
                    command_doctor(settings, as_json=True)
            finally:
                os.chdir(previous)
        payload = json.loads(buffer.getvalue())
        self.assertTrue(payload["live_config_present"])
        self.assertFalse(payload["live_provider_verified"])
        self.assertNotIn("ready_live", payload)

    def test_all_runtime_output_categories_share_one_root(self) -> None:
        settings = Settings(output_dir=Path("custom-outputs"))
        status = settings.public_status()["outputs"]
        self.assertEqual(status["logs"], "custom-outputs/logs")
        self.assertEqual(status["iterations"], "custom-outputs/iterations")
        self.assertEqual(status["images"], "custom-outputs/images")

    def test_negative_image_estimate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "cannot be negative"):
            Settings(image_estimated_cost_usd=-0.01)

    def test_non_finite_image_estimate_is_rejected(self) -> None:
        for bad_val in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(val=bad_val), self.assertRaisesRegex(
                ConfigurationError, "must be finite"
            ):
                Settings(image_estimated_cost_usd=bad_val)

    def test_non_finite_human_rate_is_rejected(self) -> None:
        for bad_val in (float("nan"), float("inf"), float("-inf"), -5.0):
            with self.subTest(val=bad_val), self.assertRaisesRegex(
                ConfigurationError, "must be finite"
            ):
                Settings(human_rate=bad_val)

    def test_float_env_rejects_non_finite(self) -> None:
        from engine.config import _float_env

        for bad_str in ("nan", "NaN", "inf", "-inf", "Infinity", "-Infinity"):
            with self.subTest(raw=bad_str), patch.dict(
                os.environ, {"MING_TEST_FLOAT": bad_str}, clear=True
            ), self.assertRaisesRegex(ConfigurationError, "finite"):
                _float_env("MING_TEST_FLOAT", 1.0)

    def test_cli_finite_float_validation(self) -> None:
        import argparse

        from ming.cli import _finite_non_negative_float

        self.assertEqual(_finite_non_negative_float("0.05"), 0.05)
        self.assertEqual(_finite_non_negative_float("0"), 0.0)
        for bad_str in ("nan", "NaN", "inf", "-inf", "-0.01", "abc"):
            with self.subTest(val=bad_str), self.assertRaises(argparse.ArgumentTypeError):
                _finite_non_negative_float(bad_str)

    def test_build_runtime_uses_explicit_workspace_root(self) -> None:
        from ming.cli import build_image_artist, build_runtime

        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            elsewhere = Path(tmp) / "elsewhere"
            project.mkdir()
            elsewhere.mkdir()
            settings = Settings(data_dir=Path("data"), output_dir=Path("outputs"))
            previous = Path.cwd()
            os.chdir(elsewhere)
            try:
                codex, telemetry, _ = build_runtime(settings, workspace_root=project)
                _, _, artist = build_image_artist(settings, workspace_root=project)
            finally:
                os.chdir(previous)
        self.assertTrue(str(codex.db_path).startswith(str(project.resolve())))
        self.assertTrue(str(telemetry.db_path).startswith(str(project.resolve())))
        self.assertEqual(artist.workspace_root, project.resolve())
        self.assertFalse(str(codex.db_path).startswith(str(elsewhere.resolve())))

    def test_cli_data_dir_cannot_escape_workspace(self) -> None:
        import os

        from engine.output_store import OutputStoreError
        from ming.cli import _data_path

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            settings = Settings(data_dir=workspace.parent / "escape")
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                with self.assertRaisesRegex(OutputStoreError, "MING_DATA_DIR"):
                    _data_path(settings, "codex.db")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
