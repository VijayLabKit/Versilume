from __future__ import annotations

import abc
from dataclasses import dataclass


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider cannot complete a request."""


@dataclass
class LLMResult:
    text: str
    provider_name: str
    model_name: str
    used_fallback: bool = False


class LLMProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        raise NotImplementedError


@dataclass
class GeneratedImageResult:
    image_bytes: bytes
    provider_name: str
    model_name: str
    mime_type: str = "image/png"

    @property
    def base64(self) -> str:
        import base64 as _b64
        return _b64.b64encode(self.image_bytes).decode("utf-8")

    @property
    def model_used(self) -> str:
        return f"{self.provider_name}:{self.model_name}"


class ImageProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
    ) -> GeneratedImageResult:
        raise NotImplementedError
