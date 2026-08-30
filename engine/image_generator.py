from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .contracts import Usage, usage_from_provider, utc_now
from .llm import (
    NETWORK_ERRORS,
    GatewayError,
    MissingAPIKeyError,
    SlidingWindowLimiter,
    Transport,
    _urllib_transport,
    dispatch_transport,
)
from .output_store import resolve_output_root

SUPPORTED_ASPECT_RATIOS = ("1:1", "4:3", "3:2", "16:9", "4:5", "2:3", "9:16")
MAX_PROMPT_CHARACTERS = 8_000
MAX_ENCODED_IMAGE_BYTES = 48 * 1024 * 1024
MAX_DECODED_IMAGE_BYTES = 32 * 1024 * 1024


class ImageGenerationError(GatewayError):
    pass


class ImageAuthorizationError(ImageGenerationError):
    pass


class ImageGenerationCancelled(asyncio.CancelledError):
    def __init__(self, artifact: ImageArtifact | None = None) -> None:
        super().__init__("image generation cancelled")
        self.artifact = artifact


@dataclass(frozen=True, slots=True)
class ImageModelCapabilities:
    aspect_ratios: tuple[str, ...]
    resolutions: tuple[str, ...]
    parameters: frozenset[str]


# Pinned from OpenRouter GET /api/v1/images/models on 2026-08-29.
PINNED_IMAGE_CAPABILITIES: dict[str, ImageModelCapabilities] = {
    "krea/krea-2-medium-turbo": ImageModelCapabilities(
        aspect_ratios=("1:1", "4:3", "3:2", "16:9", "4:5", "2:3", "9:16"),
        resolutions=("1K",),
        parameters=frozenset({"resolution", "aspect_ratio", "input_references", "seed"}),
    )
}


@dataclass(frozen=True, slots=True)
class ImageRequest:
    need: str
    purpose: str = "general visual support"
    audience: str = "the intended audience"
    style: str = "clear, intentional, context-appropriate visual design"
    avoid: str = "generic stock imagery, clutter, illegible details, accidental text"
    exact_text: str | None = None
    aspect_ratio: str = "1:1"
    resolution: str = "1K"
    model: str | None = None


@dataclass(slots=True)
class ImageArtifact:
    generation_id: str
    need: str
    prompt: str
    requested_model: str
    actual_model: str
    output_path: str
    manifest_path: str
    media_type: str
    sha256: str
    byte_count: int
    aspect_ratio: str
    resolution: str
    estimated_cost_usd: float
    authorized_max_cost_usd: float
    observed_cost_usd: float | None
    budget_exceeded: bool
    usage: Usage
    latency_ms: float
    created_at: str
    human_review_required: bool = True
    pricing_source: str = "configured_estimate"
    cost_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized[:48] or "ming-image"


def _is_free_model(model: str) -> bool:
    return model.endswith(":free")


def _observed_cost(raw: object) -> tuple[float | None, str | None]:
    if raw is None:
        return None, None
    try:
        cost = float(raw)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(cost) or cost < 0:
        return None, "provider cost was non-finite or negative; treated as unavailable"
    return cost, None


def _extension_aliases(extension: str) -> set[str]:
    return {extension, ".jpeg" if extension == ".jpg" else extension}


def _destination_matches_request(path: Path, extension: str) -> bool:
    suffix = path.suffix.lower()
    return suffix in _extension_aliases(extension) or suffix == ""


def _sniff_media_type(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    raise ImageGenerationError("image response was not a supported PNG, JPEG, or WebP raster")


def _atomic_write(path: Path, content: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _swap_aside(path: Path) -> Path | None:
    """Move an existing regular file to a temp sibling so it can be restored on failure."""
    if not path.is_file():
        return None
    aside = path.with_name(f".{path.name}.bak-{uuid.uuid4().hex[:8]}")
    os.replace(path, aside)
    return aside


def _persist_pair(
    image_path: Path,
    manifest_path: Path,
    image_bytes: bytes,
    manifest_bytes: bytes,
    *,
    overwrite: bool,
) -> None:
    """Write the image and its manifest as an all-or-nothing pair.

    Rollback is ownership-aware: it restores a predecessor when this call swapped
    one aside, removes a file only when this call created it, and never touches a
    path this call neither created nor replaced (e.g. a concurrent writer's).
    """
    steps: list[tuple[Path, bytes]] = [
        (image_path, image_bytes),
        (manifest_path, manifest_bytes),
    ]
    # [path, backup_or_None, created_by_this_call]; recorded before each write so
    # a swapped-aside predecessor is restorable even if its own write then fails.
    state: list[list] = []
    try:
        for path, content in steps:
            entry: list = [path, _swap_aside(path) if overwrite else None, False]
            state.append(entry)
            _atomic_write(path, content, overwrite=overwrite)
            entry[2] = True
    except Exception:
        for path, backup, created in reversed(state):
            if backup is not None:
                path.unlink(missing_ok=True)
                os.replace(backup, path)
            elif created:
                path.unlink(missing_ok=True)
        raise
    for _, backup, _ in state:
        if backup is not None:
            backup.unlink(missing_ok=True)


class ImageArtist:
    def __init__(
        self,
        settings: Settings,
        *,
        workspace_root: Path | None = None,
        transport: Transport = _urllib_transport,
        timeout: float = 180.0,
        model_capabilities: Mapping[str, ImageModelCapabilities] | None = None,
    ) -> None:
        self.settings = settings
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.transport = transport
        self.timeout = timeout
        self.limiter = SlidingWindowLimiter(settings.requests_per_minute)
        self.model_capabilities = {
            **PINNED_IMAGE_CAPABILITIES,
            **dict(model_capabilities or {}),
        }

    def build_prompt(self, request: ImageRequest) -> str:
        need = request.need.strip()
        if not need:
            raise ValueError("image need must not be empty")
        if request.aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
            raise ValueError(
                f"unsupported aspect ratio {request.aspect_ratio!r}; choose from "
                f"{', '.join(SUPPORTED_ASPECT_RATIOS)}"
            )
        if request.resolution != "1K":
            raise ValueError("the default cheap image route is intentionally limited to 1K")
        text_direction = (
            f'Render this exact text once, spelled exactly: "{request.exact_text.strip()}".'
            if request.exact_text and request.exact_text.strip()
            else "Do not include words, letters, logos, captions, watermarks, or interface chrome."
        )
        prompt = (
            "Create one finished image that solves the communication need below.\n\n"
            f"NEED: {need}\n"
            f"USE: {request.purpose.strip()}\n"
            f"AUDIENCE: {request.audience.strip()}\n"
            f"ART DIRECTION: {request.style.strip()}\n"
            f"COMPOSITION: Design specifically for a {request.aspect_ratio} frame. Make the "
            "primary subject and visual hierarchy immediately legible at thumbnail size.\n"
            f"TEXT: {text_direction}\n"
            f"AVOID: {request.avoid.strip()}\n\n"
            "Return a cohesive final image, not a mood board, contact sheet, split screen, "
            "or mockup."
        )
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(
                f"composed image prompt exceeds {MAX_PROMPT_CHARACTERS} characters"
            )
        return prompt

    def preview(self, request: ImageRequest) -> dict[str, Any]:
        model = request.model or self.settings.image_models[0]
        self._validate_model(model)
        capabilities = self._capabilities_for(model)
        self._validate_request_shape(request, capabilities)
        is_free = _is_free_model(model)
        estimated_cost = self.settings.estimate_for_image_model(model)
        return {
            "need": request.need.strip(),
            "prompt": self.build_prompt(request),
            "model": model,
            "cost_class": "free" if is_free else "paid-cheap",
            "estimated_cost_usd": estimated_cost,
            "pricing_source": "configured_estimate",
            "aspect_ratio": request.aspect_ratio,
            "resolution": request.resolution,
            "image_count": 1,
            "api_request_will_run": False,
            "human_review_required": True,
        }

    async def generate(
        self,
        request: ImageRequest,
        *,
        allow_paid: bool,
        max_cost_usd: float | None,
        output: Path | None = None,
        overwrite: bool = False,
    ) -> ImageArtifact:
        if not self.settings.api_key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY is not configured; use 'ming image --dry-run' or add it to .env"
            )
        model = request.model or self.settings.image_models[0]
        self._validate_model(model)
        capabilities = self._capabilities_for(model)
        self._validate_request_shape(request, capabilities)
        prompt = self.build_prompt(request)
        estimated_cost = self.settings.estimate_for_image_model(model)
        authorized_cost = self._authorize_cost(
            model, estimated_cost, allow_paid=allow_paid, max_cost_usd=max_cost_usd
        )
        requested_path = self._requested_output_path(output, request.need)
        self._reject_existing_outputs(requested_path, overwrite=overwrite)

        payload = self._generation_payload(model, prompt, request, capabilities)
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Ming Image Artist",
            "X-OpenRouter-Metadata": "enabled",
        }
        response_body: dict[str, Any] | None = None
        latency_ms = 0.0
        failures: list[str] = []
        cancelled = False
        for retry in range(self.settings.max_retries + 1):
            await self.limiter.acquire()
            started = time.perf_counter()
            try:
                outcome = await dispatch_transport(
                    self.transport,
                    f"{self.settings.base_url}/images",
                    headers,
                    json.dumps(payload).encode("utf-8"),
                    self.timeout,
                )
                response = outcome.response
                cancelled = outcome.cancelled
            except asyncio.CancelledError:
                raise ImageGenerationCancelled() from None
            except NETWORK_ERRORS as exc:
                failures.append(f"network {type(exc).__name__}")
                if retry < self.settings.max_retries:
                    await asyncio.sleep(min(2**retry, 4))
                    continue
                break
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status == 200 and isinstance(response.body.get("data"), list):
                response_body = response.body
                break
            if cancelled:
                raise ImageGenerationCancelled()
            message = self._safe_error(response.body)
            status_label = f"HTTP {response.status}"
            failures.append(f"{status_label} {message}")
            retryable = response.status == 429 or response.status >= 500
            if retryable and retry < self.settings.max_retries:
                await asyncio.sleep(min(2**retry, 4))
                continue
            break
        if response_body is None:
            if cancelled:
                raise ImageGenerationCancelled()
            detail = "; ".join(failures[-3:]) or "no response"
            raise ImageGenerationError(f"OpenRouter image generation failed: {detail}")

        artifact = self._persist_billed_response(
            request,
            model=model,
            prompt=prompt,
            estimated_cost=estimated_cost,
            authorized_cost=authorized_cost,
            requested_path=requested_path,
            overwrite=overwrite,
            response_body=response_body,
            latency_ms=latency_ms,
        )
        if cancelled:
            raise ImageGenerationCancelled(artifact)
        return artifact

    def _validate_model(self, model: str) -> None:
        if model not in self.settings.image_models:
            raise ValueError(
                f"image model {model!r} is not in MING_IMAGE_MODELS; configured models: "
                f"{', '.join(self.settings.image_models)}"
            )

    def _capabilities_for(self, model: str) -> ImageModelCapabilities:
        try:
            return self.model_capabilities[model]
        except KeyError as exc:
            raise ImageGenerationError(
                f"no pinned image capabilities for {model!r}; Ming only sends "
                "model-specific parameters for known endpoints"
            ) from exc

    @staticmethod
    def _validate_request_shape(
        request: ImageRequest, capabilities: ImageModelCapabilities
    ) -> None:
        if (
            "aspect_ratio" in capabilities.parameters
            and request.aspect_ratio not in capabilities.aspect_ratios
        ):
            raise ValueError(
                f"aspect ratio {request.aspect_ratio!r} is not supported by this image model"
            )
        if (
            "resolution" in capabilities.parameters
            and request.resolution not in capabilities.resolutions
        ):
            raise ValueError(
                f"resolution {request.resolution!r} is not supported by the selected image model"
            )
        if "aspect_ratio" not in capabilities.parameters and request.aspect_ratio != "1:1":
            raise ValueError(
                "the selected image model does not accept aspect_ratio; omit a custom ratio"
            )
        if "resolution" not in capabilities.parameters and request.resolution != "1K":
            raise ValueError(
                "the selected image model does not accept resolution; omit a custom resolution"
            )

    @staticmethod
    def _generation_payload(
        model: str,
        prompt: str,
        request: ImageRequest,
        capabilities: ImageModelCapabilities,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "provider": {"allow_fallbacks": True, "sort": "price"},
        }
        if "n" in capabilities.parameters:
            payload["n"] = 1
        if "resolution" in capabilities.parameters:
            payload["resolution"] = request.resolution
        if "aspect_ratio" in capabilities.parameters:
            payload["aspect_ratio"] = request.aspect_ratio
        return payload

    def _persist_billed_response(
        self,
        request: ImageRequest,
        *,
        model: str,
        prompt: str,
        estimated_cost: float,
        authorized_cost: float,
        requested_path: Path,
        overwrite: bool,
        response_body: dict[str, Any],
        latency_ms: float,
    ) -> ImageArtifact:
        image_bytes, response_media_type = self._decode_image(response_body)
        sniffed_media_type, extension = _sniff_media_type(image_bytes)
        if response_media_type and response_media_type != sniffed_media_type:
            raise ImageGenerationError(
                "image response media type did not match the decoded file signature"
            )
        usage = usage_from_provider(response_body.get("usage"))
        raw_usage = response_body.get("usage")
        usage_data = raw_usage if isinstance(raw_usage, dict) else {}
        observed_cost, cost_warning = _observed_cost(usage_data.get("cost"))
        sha256 = hashlib.sha256(image_bytes).hexdigest()
        generation_id = uuid.uuid4().hex[:12]
        actual_model = str(response_body.get("model") or model)
        created_at = utc_now()

        # The generation is already billed. If a concurrent Ming claims the
        # resolved pair between _resolve_destination() and the write, pick another
        # free pair and retry persistence rather than discard the received image.
        for attempt in range(3):
            output_path, manifest_path = self._resolve_destination(
                requested_path, extension, overwrite
            )
            replace = overwrite and _destination_matches_request(requested_path, extension)
            artifact = ImageArtifact(
                generation_id=generation_id,
                need=request.need.strip(),
                prompt=prompt,
                requested_model=model,
                actual_model=actual_model,
                output_path=str(output_path),
                manifest_path=str(manifest_path),
                media_type=sniffed_media_type,
                sha256=sha256,
                byte_count=len(image_bytes),
                aspect_ratio=request.aspect_ratio,
                resolution=request.resolution,
                estimated_cost_usd=estimated_cost,
                authorized_max_cost_usd=authorized_cost,
                observed_cost_usd=observed_cost,
                budget_exceeded=(
                    observed_cost is not None and observed_cost > authorized_cost
                ),
                usage=usage,
                latency_ms=latency_ms,
                created_at=created_at,
                pricing_source="configured_estimate",
                cost_warning=cost_warning,
            )
            manifest_bytes = json.dumps(
                artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False
            ).encode("utf-8")
            try:
                _persist_pair(
                    output_path,
                    manifest_path,
                    image_bytes,
                    manifest_bytes,
                    overwrite=replace,
                )
            except FileExistsError:
                if replace or attempt == 2:
                    raise
                continue
            return artifact
        raise ImageGenerationError(
            "could not claim a free output path for the generated image"
        )

    @staticmethod
    def _authorize_cost(
        model: str,
        estimated_cost: float,
        *,
        allow_paid: bool,
        max_cost_usd: float | None,
    ) -> float:
        if not math.isfinite(estimated_cost) or estimated_cost < 0:
            raise ImageAuthorizationError(
                "configured estimated cost must be a finite non-negative number"
            )
        if _is_free_model(model):
            return 0.0
        if not allow_paid:
            raise ImageAuthorizationError(
                "the configured image model is paid; pass --allow-paid to authorize one image"
            )
        if max_cost_usd is None:
            raise ImageAuthorizationError(
                "paid image generation requires --max-cost-usd"
            )
        if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
            raise ImageAuthorizationError(
                "authorized maximum cost must be a finite non-negative number"
            )
        if max_cost_usd < estimated_cost:
            raise ImageAuthorizationError(
                f"authorized maximum ${max_cost_usd:.4f} is below the configured "
                f"${estimated_cost:.4f} estimate for {model}"
            )
        return max_cost_usd

    @staticmethod
    def _reject_existing_outputs(requested_path: Path, *, overwrite: bool) -> None:
        if overwrite:
            return
        candidates = [requested_path]
        if requested_path.suffix:
            candidates.append(
                requested_path.with_suffix(f"{requested_path.suffix}.json")
            )
        else:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                raster = requested_path.with_suffix(ext)
                candidates.append(raster)
                candidates.append(raster.with_suffix(f"{ext}.json"))
        for path in candidates:
            if not path.exists():
                continue
            kind = "provenance manifest" if path.name.endswith(".json") else "output"
            raise FileExistsError(
                f"{kind} already exists: {path}; pass --overwrite to replace it"
            )

    def _requested_output_path(self, output: Path | None, need: str) -> Path:
        resolved_outputs = resolve_output_root(
            self.settings.output_dir, self.workspace_root
        )
        resolved_output_root = resolved_outputs / "images"
        if output is None:
            requested = resolved_output_root / f"{_slug(need)}-{int(time.time())}.png"
        else:
            requested = output if output.is_absolute() else resolved_output_root / output
        resolved = requested.resolve(strict=False)
        try:
            resolved.relative_to(resolved_output_root)
        except ValueError as exc:
            raise ValueError("image output must remain inside outputs/images") from exc
        if resolved.suffix.lower() not in {"", ".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("image output suffix must be PNG, JPEG, or WebP")
        return resolved

    @staticmethod
    def _resolve_destination(
        path: Path, extension: str, overwrite: bool
    ) -> tuple[Path, Path]:
        if path.suffix.lower() in _extension_aliases(extension):
            image = path
        else:
            image = path.with_suffix(extension)
        manifest = image.with_suffix(f"{image.suffix}.json")
        replace = overwrite and _destination_matches_request(path, extension)
        while not replace and (image.exists() or manifest.exists()):
            image = image.with_name(f"{path.stem}-{uuid.uuid4().hex[:8]}{extension}")
            manifest = image.with_suffix(f"{image.suffix}.json")
        return image, manifest

    def _safe_error(self, body: dict[str, Any]) -> str:
        error = body.get("error") or {}
        message = (
            str(error.get("message", "request failed"))
            if isinstance(error, dict)
            else "request failed"
        )
        if self.settings.api_key:
            message = message.replace(self.settings.api_key, "[REDACTED]")
        return message.replace("\n", " ")[:240]

    @staticmethod
    def _decode_image(body: dict[str, Any]) -> tuple[bytes, str | None]:
        try:
            item = body["data"][0]
            encoded = item["b64_json"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ImageGenerationError(
                "OpenRouter image response did not contain base64 image data"
            ) from exc
        if not isinstance(encoded, str) or len(encoded) > MAX_ENCODED_IMAGE_BYTES:
            raise ImageGenerationError("OpenRouter image response exceeded the safe size limit")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageGenerationError("OpenRouter returned invalid base64 image data") from exc
        if not decoded or len(decoded) > MAX_DECODED_IMAGE_BYTES:
            raise ImageGenerationError("decoded image was empty or exceeded the safe size limit")
        media_type = item.get("media_type")
        return decoded, str(media_type) if media_type else None
