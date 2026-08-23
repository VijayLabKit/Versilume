from __future__ import annotations

import functools
import logging

from app.config import get_settings
from app.providers.base import ImageProvider, LLMProvider
from app.providers.image_providers import HuggingFaceImageProvider, PollinationsImageProvider
from app.providers.llm_providers import GeminiProvider, HuggingFaceLLMProvider, NullLLMProvider

logger = logging.getLogger(__name__)


def _build_llm_provider(provider_name: str) -> LLMProvider:
    settings = get_settings()
    provider_name = (provider_name or "none").strip().lower()

    if provider_name == "gemini":
        return GeminiProvider(model_name=settings.model_name, api_key=settings.gemini_api_key)
    if provider_name == "huggingface":
        return HuggingFaceLLMProvider(model_name=settings.model_name, api_key=settings.hf_api_token)
    if provider_name in ("none", "", "null", "off"):
        return NullLLMProvider()

    logger.warning("Unknown MODEL_PROVIDER='%s' — falling back to NullLLMProvider.", provider_name)
    return NullLLMProvider()


@functools.lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    try:
        provider = _build_llm_provider(settings.model_provider)
        logger.info("LLM provider: %s (model=%s)", provider.name, settings.model_name)
        return provider
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build LLM provider (%s). Falling back to NullLLMProvider.", exc)
        return NullLLMProvider()


def refresh_llm_provider() -> LLMProvider:
    get_llm_provider.cache_clear()
    return get_llm_provider()


@functools.lru_cache(maxsize=1)
def get_image_provider_chain() -> tuple[ImageProvider, ...]:
    settings = get_settings()
    requested = [p.strip().lower() for p in (settings.image_model_provider or "").split(",") if p.strip()]
    if not requested:
        requested = ["huggingface"]

    chain: list[ImageProvider] = []
    for name in requested:
        try:
            if name == "huggingface":
                chain.append(
                    HuggingFaceImageProvider(
                        model_name=settings.image_model_name,
                        api_key=settings.hf_api_token,
                        timeout_s=settings.hf_request_timeout_s,
                    )
                )
            elif name == "pollinations":
                chain.append(PollinationsImageProvider())
            else:
                logger.warning("Unknown IMAGE_MODEL_PROVIDER entry '%s' — skipping.", name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to build image provider '%s' (%s) — skipping.", name, exc)

    # Pollinations is always appended as a keyless last-resort fallback
    if not any(isinstance(p, PollinationsImageProvider) for p in chain):
        chain.append(PollinationsImageProvider())

    logger.info("Image provider chain: %s", [p.name for p in chain])
    return tuple(chain)


def refresh_image_provider_chain() -> tuple[ImageProvider, ...]:
    get_image_provider_chain.cache_clear()
    return get_image_provider_chain()
