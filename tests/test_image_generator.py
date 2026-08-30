from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.config import Settings
from engine.image_generator import (
    PINNED_IMAGE_CAPABILITIES,
    ImageArtist,
    ImageAuthorizationError,
    ImageGenerationCancelled,
    ImageGenerationError,
    ImageRequest,
)
from engine.llm import TransportResponse
from engine.telemetry import TelemetryStore

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)
JPEG_STUB = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"


def image_response(
    *, cost: float = 0.015, media_type: str = "image/png", payload: bytes = PNG_1X1
) -> TransportResponse:
    return TransportResponse(
        200,
        {
            "model": "krea/krea-2-medium-turbo",
            "data": [
                {
                    "b64_json": base64.b64encode(payload).decode("ascii"),
                    "media_type": media_type,
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 12,
                "total_tokens": 16,
                "cost": cost,
            },
        },
    )


class ImageArtistTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, tmp: str, **overrides: object) -> Settings:
        values = {
            "api_key": "secret-key",
            "data_dir": Path(tmp) / "data",
            "output_dir": Path("outputs"),
            "max_retries": 0,
        }
        values.update(overrides)
        return Settings(**values)  # type: ignore[arg-type]

    def test_preview_builds_need_driven_prompt_without_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artist = ImageArtist(self.settings(tmp), workspace_root=Path(tmp))
            preview = artist.preview(
                ImageRequest(
                    need="A launch hero",
                    purpose="homepage",
                    audience="developers",
                    style="editorial collage",
                    aspect_ratio="16:9",
                )
            )
        self.assertFalse(preview["api_request_will_run"])
        self.assertEqual(preview["estimated_cost_usd"], 0.015)
        self.assertIn("NEED: A launch hero", preview["prompt"])
        self.assertIn("Do not include words", preview["prompt"])

    def test_exact_text_replaces_default_no_text_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompt = ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp)
            ).build_prompt(ImageRequest(need="Poster", exact_text="MING"))
        self.assertIn('Render this exact text once, spelled exactly: "MING"', prompt)
        self.assertNotIn("Do not include words", prompt)

    async def test_paid_route_requires_explicit_authorization_before_transport(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            artist = ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            )
            with self.assertRaisesRegex(ImageAuthorizationError, "--allow-paid"):
                await artist.generate(
                    ImageRequest(need="Hero"), allow_paid=False, max_cost_usd=0.02
                )
        self.assertFalse(called)

    async def test_estimate_must_fit_authorized_maximum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artist = ImageArtist(self.settings(tmp), workspace_root=Path(tmp))
            with self.assertRaisesRegex(ImageAuthorizationError, "below"):
                await artist.generate(
                    ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.01
                )

    async def test_success_saves_image_manifest_and_telemetry(self) -> None:
        captured: dict[str, object] = {}

        def transport(
            url: str, headers: object, payload: bytes, timeout: float
        ) -> TransportResponse:
            captured.update(
                url=url, headers=headers, payload=json.loads(payload), timeout=timeout
            )
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self.settings(tmp)
            artist = ImageArtist(settings, workspace_root=root, transport=transport)
            artifact = await artist.generate(
                ImageRequest(need="Dashboard hero", aspect_ratio="16:9"),
                allow_paid=True,
                max_cost_usd=0.02,
                output=Path("assets/hero.png"),
            )
            telemetry = TelemetryStore(root / "outputs" / "logs" / "telemetry.db")
            telemetry.record_image(artifact)
            summary = telemetry.summary()
            manifest = json.loads(Path(artifact.manifest_path).read_text())
            saved_image = Path(artifact.output_path).read_bytes()

        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/images")
        self.assertNotIn("n", captured["payload"])  # type: ignore[operator]
        self.assertEqual(captured["payload"]["resolution"], "1K")  # type: ignore[index]
        self.assertEqual(captured["payload"]["aspect_ratio"], "16:9")  # type: ignore[index]
        self.assertEqual(saved_image, PNG_1X1)
        self.assertTrue(
            Path(artifact.output_path).is_relative_to(
                (root / "outputs" / "images").resolve()
            )
        )
        self.assertEqual(manifest["sha256"], artifact.sha256)
        self.assertEqual(artifact.observed_cost_usd, 0.015)
        self.assertFalse(artifact.budget_exceeded)
        self.assertEqual(summary["images"], 1)
        self.assertEqual(summary["image_observed_cost_usd"], 0.015)

    async def test_output_cannot_escape_workspace_and_transport_is_not_called(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            artist = ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            )
            with self.assertRaisesRegex(ValueError, "inside"):
                await artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=Path(tmp).parent / "escape.png",
                )
        self.assertFalse(called)

    async def test_existing_output_blocks_before_billed_request(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outputs" / "images" / "hero.png"
            target.parent.mkdir(parents=True)
            target.write_bytes(PNG_1X1)
            artist = ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            )
            with self.assertRaises(FileExistsError):
                await artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=target,
                )
        self.assertFalse(called)

    async def test_extensionless_output_blocks_existing_alternate_raster(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "outputs" / "images"
            image_root.mkdir(parents=True)
            (image_root / "hero.webp").write_bytes(b"RIFF....WEBP")
            artist = ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            )
            with self.assertRaisesRegex(FileExistsError, "hero.webp"):
                await artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=Path("hero"),
                )
        self.assertFalse(called)

    async def test_existing_manifest_blocks_before_billed_request(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "outputs" / "images"
            image_root.mkdir(parents=True)
            (image_root / "hero.png.json").write_text("{}")
            artist = ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            )
            with self.assertRaisesRegex(FileExistsError, "manifest"):
                await artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=Path("hero.png"),
                )
        self.assertFalse(called)

    async def test_invalid_base64_creates_no_artifact(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return TransportResponse(200, {"data": [{"b64_json": "not!base64"}]})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artist = ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            )
            with self.assertRaisesRegex(ImageGenerationError, "base64"):
                await artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=Path("hero.png"),
                )
            image_root = root / "outputs" / "images"
            self.assertFalse((image_root / "hero.png").exists())
            self.assertFalse((image_root / "hero.png.json").exists())

    async def test_media_type_spoof_is_rejected(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response(media_type="image/jpeg")

        with tempfile.TemporaryDirectory() as tmp:
            artist = ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            )
            with self.assertRaisesRegex(ImageGenerationError, "did not match"):
                await artist.generate(
                    ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
                )

    async def test_error_redacts_key(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return TransportResponse(
                400, {"error": {"message": "secret-key was rejected"}}
            )

        with tempfile.TemporaryDirectory() as tmp:
            artist = ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            )
            with self.assertRaises(ImageGenerationError) as context:
                await artist.generate(
                    ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
                )
        self.assertNotIn("secret-key", str(context.exception))
        self.assertIn("[REDACTED]", str(context.exception))

    async def test_observed_overage_is_preserved_and_flagged(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response(cost=0.03)

        with tempfile.TemporaryDirectory() as tmp:
            artifact = await ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            ).generate(
                ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
            )
            self.assertTrue(Path(artifact.output_path).exists())
        self.assertTrue(artifact.budget_exceeded)

    async def test_malformed_provider_cost_does_not_discard_billed_artifact(self) -> None:
        def transport(*args: object) -> TransportResponse:
            response = image_response()
            response.body["usage"]["cost"] = "unknown"
            return response

        with tempfile.TemporaryDirectory() as tmp:
            artifact = await ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            ).generate(
                ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
            )
            self.assertTrue(Path(artifact.output_path).exists())
        self.assertIsNone(artifact.observed_cost_usd)

    async def test_malformed_usage_tokens_do_not_discard_billed_artifact(self) -> None:
        def transport(*args: object) -> TransportResponse:
            response = image_response()
            response.body["usage"]["prompt_tokens"] = "unknown"
            response.body["usage"]["completion_tokens"] = {"n": 1}
            response.body["usage"]["total_tokens"] = "12"
            return response

        with tempfile.TemporaryDirectory() as tmp:
            artifact = await ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            ).generate(
                ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
            )
            self.assertTrue(Path(artifact.output_path).exists())
        self.assertEqual(artifact.usage.prompt_tokens, 0)
        self.assertEqual(artifact.usage.completion_tokens, 0)
        self.assertEqual(artifact.usage.total_tokens, 12)

    async def test_http_200_without_image_data_fails_without_retry(self) -> None:
        calls = {"n": 0}

        def transport(*args: object) -> TransportResponse:
            calls["n"] += 1
            return TransportResponse(
                200, {"error": {"message": "OpenRouter returned a non-JSON response"}}
            )

        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ImageGenerationError):
            await ImageArtist(
                self.settings(tmp, max_retries=1),
                workspace_root=Path(tmp),
                transport=transport,
            ).generate(
                ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
            )
        self.assertEqual(calls["n"], 1)

    async def test_overwrite_does_not_clobber_alternate_format_sibling(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response(media_type="image/jpeg", payload=JPEG_STUB)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "outputs" / "images"
            image_root.mkdir(parents=True)
            (image_root / "hero.png").write_bytes(PNG_1X1)
            sibling = image_root / "hero.jpg"
            sibling.write_bytes(b"\xff\xd8\xff preexisting-jpeg")
            artifact = await ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            ).generate(
                ImageRequest(need="Hero"),
                allow_paid=True,
                max_cost_usd=0.02,
                output=Path("hero.png"),
                overwrite=True,
            )
            self.assertEqual(sibling.read_bytes(), b"\xff\xd8\xff preexisting-jpeg")
            self.assertNotEqual(Path(artifact.output_path), sibling)
            self.assertTrue(artifact.output_path.endswith(".jpg"))
            self.assertTrue((image_root / "hero.png").exists())
            self.assertTrue(Path(artifact.output_path).exists())

    async def test_alternate_format_manifest_collision_keeps_billed_image(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response(media_type="image/jpeg", payload=JPEG_STUB)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "outputs" / "images"
            image_root.mkdir(parents=True)
            (image_root / "hero.jpg.json").write_text("{}")  # stale alternate-format manifest
            artifact = await ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            ).generate(
                ImageRequest(need="Hero"),
                allow_paid=True,
                max_cost_usd=0.02,
                output=Path("hero.png"),
            )
            self.assertTrue(Path(artifact.output_path).exists())
        self.assertNotEqual(Path(artifact.output_path).name, "hero.jpg")
        self.assertTrue(artifact.output_path.endswith(".jpg"))

    async def test_overwrite_fresh_image_removed_when_manifest_write_fails(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "outputs" / "images"
            image_root.mkdir(parents=True)
            (image_root / "hero.png.json").mkdir()  # no old image; manifest write will raise
            artist = ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            )
            with self.assertRaises(OSError):
                await artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=Path("hero.png"),
                    overwrite=True,
                )
            self.assertFalse((image_root / "hero.png").exists())  # not left orphaned

    async def test_destination_race_preserves_competitor_and_still_saves(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artist = ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            )
            real_resolve = artist._resolve_destination
            seen: dict[str, Path] = {}

            def racing_resolve(path: Path, extension: str, overwrite: bool):
                image, manifest = real_resolve(path, extension, overwrite)
                if "image" not in seen:  # simulate a competitor claiming the pair
                    seen["image"] = image
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"competitor-image")
                    manifest.write_text("competitor-manifest")
                return image, manifest

            artist._resolve_destination = racing_resolve  # type: ignore[method-assign]
            artifact = await artist.generate(
                ImageRequest(need="Hero"),
                allow_paid=True,
                max_cost_usd=0.02,
                output=Path("hero.png"),
            )
            self.assertEqual(seen["image"].read_bytes(), b"competitor-image")
            self.assertNotEqual(artifact.output_path, str(seen["image"]))
            self.assertTrue(Path(artifact.output_path).exists())

    async def test_overwrite_manifest_failure_preserves_old_image(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "outputs" / "images"
            image_root.mkdir(parents=True)
            old_bytes = b"\x89PNG\r\n\x1a\n" + b"old artifact"
            (image_root / "hero.png").write_bytes(old_bytes)
            (image_root / "hero.png.json").mkdir()  # manifest replace will raise
            artist = ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            )
            with self.assertRaises(OSError):
                await artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=Path("hero.png"),
                    overwrite=True,
                )
            self.assertEqual((image_root / "hero.png").read_bytes(), old_bytes)

    async def test_configured_free_image_route_needs_no_paid_authorization(self) -> None:
        def transport(*args: object) -> TransportResponse:
            response = image_response(cost=0.0)
            response.body["model"] = "example/image:free"
            return response

        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(
                tmp,
                image_models=("example/image:free",),
                image_estimated_cost_usd=0.0,
            )
            artifact = await ImageArtist(
                settings,
                workspace_root=Path(tmp),
                transport=transport,
                model_capabilities={
                    "example/image:free": PINNED_IMAGE_CAPABILITIES[
                        "krea/krea-2-medium-turbo"
                    ]
                },
            ).generate(
                ImageRequest(need="Hero"), allow_paid=False, max_cost_usd=None
            )
        self.assertEqual(artifact.authorized_max_cost_usd, 0.0)

    async def test_non_finite_max_cost_usd_is_rejected(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            artist = ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            )
            for bad_cost in (float("nan"), float("inf"), float("-inf"), -0.05):
                with self.subTest(cost=bad_cost), self.assertRaisesRegex(
                    ImageAuthorizationError, "finite non-negative"
                ):
                    await artist.generate(
                        ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=bad_cost
                    )
        self.assertFalse(called)

    async def test_non_finite_provider_cost_does_not_corrupt_manifest(self) -> None:
        def transport(*args: object) -> TransportResponse:
            response = image_response()
            response.body["usage"]["cost"] = float("nan")
            return response

        with tempfile.TemporaryDirectory() as tmp:
            artifact = await ImageArtist(
                self.settings(tmp), workspace_root=Path(tmp), transport=transport
            ).generate(
                ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
            )
            self.assertTrue(Path(artifact.output_path).exists())
            self.assertIsNone(artifact.observed_cost_usd)
            manifest = json.loads(Path(artifact.manifest_path).read_text())
            self.assertIsNone(manifest["observed_cost_usd"])

    async def test_cli_image_telemetry_failure_preserves_artifact(self) -> None:
        import argparse
        import sqlite3

        from engine.output_store import IterationStore
        from ming.cli import command_image

        def transport(*args: object) -> TransportResponse:
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self.settings(tmp, output_dir=root / "outputs")
            args = argparse.Namespace(
                need="Hero poster",
                purpose="homepage",
                audience="all",
                style="clean",
                avoid="none",
                exact_text=None,
                aspect_ratio="1:1",
                model=None,
                output=None,
                dry_run=False,
                allow_paid=True,
                max_cost_usd=0.02,
                overwrite=False,
                json=False,
            )
            with patch.object(
                TelemetryStore,
                "record_image",
                side_effect=sqlite3.OperationalError("database is locked"),
            ), patch("ming.cli.build_image_artist") as mock_builder:
                telemetry = TelemetryStore(root / "outputs" / "logs" / "telemetry.db")
                iterations = IterationStore(root / "outputs" / "iterations")
                artist = ImageArtist(settings, workspace_root=root, transport=transport)
                mock_builder.return_value = (telemetry, iterations, artist)
                ret = await command_image(settings, args)
                self.assertEqual(ret, 3)
                saved_images = list((root / "outputs" / "images").glob("*.png"))
                self.assertEqual(len(saved_images), 1)
                saved_manifests = list((root / "outputs" / "images").glob("*.png.json"))
                self.assertEqual(len(saved_manifests), 1)

    async def test_cli_image_iteration_failure_preserves_artifact(self) -> None:
        import argparse

        from engine.output_store import IterationStore
        from ming.cli import command_image

        def transport(*args: object) -> TransportResponse:
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self.settings(tmp, output_dir=root / "outputs")
            args = argparse.Namespace(
                need="Hero poster",
                purpose="homepage",
                audience="all",
                style="clean",
                avoid="none",
                exact_text=None,
                aspect_ratio="1:1",
                model=None,
                output=None,
                dry_run=False,
                allow_paid=True,
                max_cost_usd=0.02,
                overwrite=False,
                json=False,
            )
            with patch("ming.cli.build_image_artist") as mock_builder:
                telemetry = TelemetryStore(root / "outputs" / "logs" / "telemetry.db")
                iterations = IterationStore(root / "outputs" / "iterations")
                with patch.object(iterations, "write", side_effect=OSError("disk full")):
                    artist = ImageArtist(settings, workspace_root=root, transport=transport)
                    mock_builder.return_value = (telemetry, iterations, artist)
                    ret = await command_image(settings, args)
                    self.assertEqual(ret, 3)
                    saved_images = list((root / "outputs" / "images").glob("*.png"))
                    self.assertEqual(len(saved_images), 1)
                    saved_manifests = list((root / "outputs" / "images").glob("*.png.json"))
                    self.assertEqual(len(saved_manifests), 1)

    async def test_cancel_after_dispatch_persists_billed_image(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def transport(*args: object) -> TransportResponse:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test gate timed out")
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artist = ImageArtist(
                self.settings(tmp, max_retries=0),
                workspace_root=root,
                transport=transport,
            )
            task = asyncio.create_task(
                artist.generate(
                    ImageRequest(need="Hero"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                    output=Path("hero.png"),
                )
            )
            await asyncio.to_thread(started.wait, 5)
            task.cancel()
            release.set()
            with self.assertRaises(ImageGenerationCancelled) as context:
                await task
            image_root = root / "outputs" / "images"
            self.assertTrue((image_root / "hero.png").exists())
            self.assertTrue((image_root / "hero.png.json").exists())
            self.assertIsNotNone(context.exception.artifact)

    async def test_selected_paid_model_uses_its_own_estimate(self) -> None:
        called = False
        caps = PINNED_IMAGE_CAPABILITIES["krea/krea-2-medium-turbo"]

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(
                tmp,
                image_models=("cheap/image", "expensive/image"),
                image_model_estimates={"cheap/image": 0.015, "expensive/image": 0.50},
            )
            artist = ImageArtist(
                settings,
                workspace_root=Path(tmp),
                transport=transport,
                model_capabilities={"cheap/image": caps, "expensive/image": caps},
            )
            with self.assertRaisesRegex(ImageAuthorizationError, "below"):
                await artist.generate(
                    ImageRequest(need="Hero", model="expensive/image"),
                    allow_paid=True,
                    max_cost_usd=0.02,
                )
        self.assertFalse(called)

    async def test_unpinned_image_model_is_rejected_before_transport(self) -> None:
        called = False

        def transport(*args: object) -> TransportResponse:
            nonlocal called
            called = True
            return image_response()

        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(tmp, image_models=("mystery/image",))
            artist = ImageArtist(
                settings, workspace_root=Path(tmp), transport=transport
            )
            with self.assertRaisesRegex(ImageGenerationError, "pinned image capabilities"):
                await artist.generate(
                    ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
                )
        self.assertFalse(called)

    async def test_negative_provider_cost_is_not_recorded_as_spend(self) -> None:
        def transport(*args: object) -> TransportResponse:
            return image_response(cost=-0.25)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = await ImageArtist(
                self.settings(tmp), workspace_root=root, transport=transport
            ).generate(
                ImageRequest(need="Hero"), allow_paid=True, max_cost_usd=0.02
            )
            telemetry = TelemetryStore(root / "outputs" / "logs" / "telemetry.db")
            telemetry.record_image(artifact)
            summary = telemetry.summary()
            manifest = json.loads(Path(artifact.manifest_path).read_text())
            self.assertTrue(Path(artifact.output_path).exists())
        self.assertIsNone(artifact.observed_cost_usd)
        self.assertFalse(artifact.budget_exceeded)
        self.assertIsNotNone(artifact.cost_warning)
        self.assertIsNone(manifest["observed_cost_usd"])
        self.assertEqual(summary["image_observed_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
