from __future__ import annotations

import asyncio
import io
import logging
import random
import urllib.parse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.providers.base import GeneratedImageResult, ImageProvider, ProviderUnavailableError

logger = logging.getLogger(__name__)


class _ModelLoadingRetry(RuntimeError):
    pass


class HuggingFaceImageProvider(ImageProvider):
    name = "huggingface"

    def __init__(self, model_name: str, api_key: str, timeout_s: float = 120.0):
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_s = timeout_s

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
    ) -> GeneratedImageResult:
        if not self.api_key:
            raise ProviderUnavailableError("HF_API_TOKEN is not set.")

        @retry(
            retry=retry_if_exception_type(_ModelLoadingRetry),
            stop=stop_after_attempt(5),
            wait=wait_fixed(8),
            reraise=True,
        )
        async def _call() -> tuple[bytes, str]:
            try:
                from huggingface_hub import InferenceClient

                client = InferenceClient(token=self.api_key, timeout=self.timeout_s)
                kwargs: dict = {}
                if negative_prompt:
                    kwargs["negative_prompt"] = negative_prompt
                if width and height:
                    kwargs["width"] = width
                    kwargs["height"] = height

                image = await asyncio.to_thread(
                    client.text_to_image, prompt, model=self.model_name or None, **kwargs
                )
                if hasattr(image, "save"):
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    return buf.getvalue(), "image/png"
                if isinstance(image, bytes):
                    return image, "image/png"
                raise ProviderUnavailableError(f"Unexpected image type: {type(image)}")
            except ProviderUnavailableError:
                raise
            except Exception as exc:
                msg = str(exc)
                if "503" in msg or "currently loading" in msg.lower():
                    raise _ModelLoadingRetry(msg) from exc
                raise ProviderUnavailableError(f"Hugging Face image generation failed: {msg}") from exc

        try:
            image_bytes, mime_type = await _call()
        except _ModelLoadingRetry as exc:
            raise ProviderUnavailableError(f"Hugging Face model kept loading: {exc}") from exc

        return GeneratedImageResult(
            image_bytes=image_bytes,
            provider_name=self.name,
            model_name=self.model_name,
            mime_type=mime_type,
        )


class PollinationsImageProvider(ImageProvider):
    name = "pollinations"

    def __init__(self, model_name: str = "flux"):
        self.model_name = model_name

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
    ) -> GeneratedImageResult:
        seed = random.randint(1, 99_999_999)
        full_prompt = prompt if not negative_prompt else f"{prompt} (avoid: {negative_prompt})"
        encoded = urllib.parse.quote(full_prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
        )
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.get(url, follow_redirects=True)
            if resp.status_code != 200:
                raise ProviderUnavailableError(f"Pollinations returned HTTP {resp.status_code}")
            mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip() or "image/jpeg"
            return GeneratedImageResult(
                image_bytes=resp.content,
                provider_name=self.name,
                model_name=self.model_name,
                mime_type=mime,
            )
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(f"Pollinations request failed: {exc}") from exc
