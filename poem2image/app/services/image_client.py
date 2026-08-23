from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from app.config import get_settings
from app.providers.base import ProviderUnavailableError
from app.providers.registry import get_image_provider_chain

logger = logging.getLogger(__name__)


class ModelLoadingError(RuntimeError):
    pass


class ImageGenerationError(RuntimeError):
    pass


@dataclass
class GeneratedImageResult:
    image_bytes: bytes
    model_used: str
    mime_type: str = "image/png"
    image_provider: str = ""

    @property
    def base64(self) -> str:
        return base64.b64encode(self.image_bytes).decode("utf-8")


async def generate_image(
    prompt: str,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> GeneratedImageResult:
    """Generate image using the provider fallback chain (HF -> Pollinations)."""
    settings = get_settings()
    width = width or settings.image_width
    height = height or settings.image_height

    chain = get_image_provider_chain()
    if not chain:
        raise ImageGenerationError("No image providers configured.")

    failures: list[str] = []
    for provider in chain:
        try:
            result = await provider.generate_image(prompt, negative_prompt, width, height)
            return GeneratedImageResult(
                image_bytes=result.image_bytes,
                model_used=result.model_name or provider.name,
                mime_type=result.mime_type,
                image_provider=result.provider_name,
            )
        except ProviderUnavailableError as exc:
            logger.warning("Image provider '%s' failed (%s); trying next.", provider.name, exc)
            failures.append(f"{provider.name}: {exc}")
            continue

    raise ImageGenerationError("All image providers failed: " + " | ".join(failures))
