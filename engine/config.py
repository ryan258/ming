from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

OFFICIAL_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter's free catalog rotates. These named routes were verified against the
# live catalog for the V1 release; openrouter/free remains the final dynamic
# fallback in every chain.
DEFAULT_ORCHESTRATOR_MODELS = (
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid/lfm-2.5-2.6b:free",
    "openrouter/free",
)
DEFAULT_CODER_MODELS = (
    "poolside/laguna-s-2.1:free",
    "cohere/north-mini-code:free",
    "openrouter/free",
)
DEFAULT_FAST_MODELS = (
    "minimax/minimax-m3:free",
    "google/gemma-4-26b-a4b-it:free",
    "liquid/lfm-2.5-2.6b:free",
    "openrouter/free",
)


class ConfigurationError(ValueError):
    """Raised when configuration would violate a Ming trust boundary."""


def _load_dotenv(path: Path) -> None:
    """Load a small, predictable subset of dotenv without overriding the shell."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        quoted = (
            len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}
        )
        if not quoted and "#" in value:
            value = value.split("#", 1)[0].rstrip()
        value = value.strip().strip("\"").strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(value):
        raise ConfigurationError(f"{name} must be a finite number")
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _models_env(name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    models = tuple(item.strip() for item in raw.split(",") if item.strip()) or defaults
    for model in models:
        if model != "openrouter/free" and not model.endswith(":free"):
            raise ConfigurationError(
                f"{name} contains non-free model {model!r}; Ming accepts only ':free' "
                "models or 'openrouter/free'"
            )
    if "openrouter/free" not in models:
        models = (*models, "openrouter/free")
    return tuple(dict.fromkeys(models))


def _image_models_env(name: str, defaults: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    models = tuple(item.strip() for item in raw.split(",") if item.strip()) or defaults
    if not models:
        raise ConfigurationError(f"{name} must contain at least one image model")
    return tuple(dict.fromkeys(models))


def _image_model_costs_env(name: str) -> dict[str, float]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return {}
    result: dict[str, float] = {}
    for item in raw.split(","):
        piece = item.strip()
        if not piece:
            continue
        model, sep, price = piece.rpartition(":")
        if not sep or not model.strip():
            raise ConfigurationError(f"{name} entries must be model:price")
        try:
            value = float(price)
        except ValueError as exc:
            raise ConfigurationError(f"{name} prices must be numeric") from exc
        if not math.isfinite(value) or value < 0:
            raise ConfigurationError(
                f"{name} prices must be finite and cannot be negative"
            )
        result[model.strip()] = value
    return result


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str | None = None
    base_url: str = OFFICIAL_OPENROUTER_BASE_URL
    allow_custom_base_url: bool = False
    orchestrator_models: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_ORCHESTRATOR_MODELS
    )
    coder_models: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_CODER_MODELS
    )
    fast_models: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_FAST_MODELS
    )
    image_models: tuple[str, ...] = ("krea/krea-2-medium-turbo",)
    image_estimated_cost_usd: float = 0.015
    image_model_estimates: dict[str, float] = field(default_factory=dict)
    output_dir: Path = Path("outputs")
    requests_per_minute: int = 20
    max_retries: int = 2
    max_specialists: int = 3
    human_rate: float = 75.0
    data_dir: Path = Path("data")

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ConfigurationError("OPENROUTER_BASE_URL must be an absolute HTTPS URL")
        normalized = self.base_url.rstrip("/")
        object.__setattr__(self, "base_url", normalized)
        if normalized != OFFICIAL_OPENROUTER_BASE_URL and not self.allow_custom_base_url:
            raise ConfigurationError(
                "custom OPENROUTER_BASE_URL requires MING_ALLOW_CUSTOM_BASE_URL=true"
            )
        if self.max_specialists < 1 or self.max_specialists > 8:
            raise ConfigurationError("MING_MAX_SPECIALISTS must be between 1 and 8")
        for chain in (self.orchestrator_models, self.coder_models, self.fast_models):
            for model in chain:
                if model != "openrouter/free" and not model.endswith(":free"):
                    raise ConfigurationError(f"non-free model route rejected: {model}")
        if not self.image_models:
            raise ConfigurationError("at least one image model must be configured")
        if (
            not math.isfinite(self.image_estimated_cost_usd)
            or self.image_estimated_cost_usd < 0
        ):
            raise ConfigurationError(
                "MING_IMAGE_ESTIMATED_COST_USD must be finite and cannot be negative"
            )
        if not math.isfinite(self.human_rate) or self.human_rate < 0:
            raise ConfigurationError(
                "MING_HUMAN_RATE must be finite and cannot be negative"
            )
        estimates = dict(self.image_model_estimates)
        paid_models = [model for model in self.image_models if not model.endswith(":free")]
        if len(paid_models) == 1 and paid_models[0] not in estimates:
            estimates[paid_models[0]] = self.image_estimated_cost_usd
        object.__setattr__(self, "image_model_estimates", estimates)
        missing = [model for model in paid_models if model not in estimates]
        if missing:
            raise ConfigurationError(
                "each paid image model needs its own estimate; set "
                f"MING_IMAGE_MODEL_COSTS for {', '.join(missing)}"
            )

    @classmethod
    def from_env(cls, env_file: Path | None = Path(".env")) -> Settings:
        if env_file is not None:
            _load_dotenv(env_file)
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY") or None,
            base_url=os.getenv("OPENROUTER_BASE_URL", OFFICIAL_OPENROUTER_BASE_URL),
            allow_custom_base_url=_bool_env("MING_ALLOW_CUSTOM_BASE_URL"),
            orchestrator_models=_models_env(
                "MING_ORCHESTRATOR_MODELS",
                DEFAULT_ORCHESTRATOR_MODELS,
            ),
            coder_models=_models_env(
                "MING_CODER_MODELS",
                DEFAULT_CODER_MODELS,
            ),
            fast_models=_models_env(
                "MING_FAST_MODELS",
                DEFAULT_FAST_MODELS,
            ),
            image_models=_image_models_env(
                "MING_IMAGE_MODELS", ("krea/krea-2-medium-turbo",)
            ),
            image_estimated_cost_usd=_float_env(
                "MING_IMAGE_ESTIMATED_COST_USD", 0.015
            ),
            image_model_estimates=_image_model_costs_env("MING_IMAGE_MODEL_COSTS"),
            output_dir=Path(os.getenv("MING_OUTPUT_DIR", "outputs")),
            requests_per_minute=_int_env("MING_REQUESTS_PER_MINUTE", 20, minimum=1),
            max_retries=_int_env("MING_MAX_RETRIES", 2, minimum=0),
            max_specialists=_int_env("MING_MAX_SPECIALISTS", 3, minimum=1),
            human_rate=_float_env("MING_HUMAN_RATE", 75.0),
            data_dir=Path(os.getenv("MING_DATA_DIR", "data")),
        )

    def models_for(self, role: str) -> tuple[str, ...]:
        if role == "orchestrator":
            return self.orchestrator_models
        if role == "coder":
            return self.coder_models
        return self.fast_models

    def estimate_for_image_model(self, model: str) -> float:
        if model.endswith(":free"):
            return 0.0
        try:
            return self.image_model_estimates[model]
        except KeyError as exc:
            raise ConfigurationError(
                f"no estimate configured for paid image model {model!r}"
            ) from exc

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def iterations_dir(self) -> Path:
        return self.output_dir / "iterations"

    @property
    def image_output_dir(self) -> Path:
        return self.output_dir / "images"

    def public_status(self) -> dict[str, object]:
        return {
            "provider": "openrouter",
            "base_url": self.base_url,
            "custom_origin": self.base_url != OFFICIAL_OPENROUTER_BASE_URL,
            "api_key_configured": bool(self.api_key),
            "free_only": True,
            "requests_per_minute": self.requests_per_minute,
            "max_specialists": self.max_specialists,
            "models": {
                "orchestrator": list(self.orchestrator_models),
                "coder": list(self.coder_models),
                "fast": list(self.fast_models),
            },
            "image_generation": {
                "models": list(self.image_models),
                "estimated_cost_usd": self.image_estimated_cost_usd,
                "model_estimates": dict(self.image_model_estimates),
                "paid_opt_in_required": any(
                    not model.endswith(":free") for model in self.image_models
                ),
                "output_dir": str(self.image_output_dir),
            },
            "outputs": {
                "root": str(self.output_dir),
                "logs": str(self.logs_dir),
                "iterations": str(self.iterations_dir),
                "images": str(self.image_output_dir),
            },
        }
