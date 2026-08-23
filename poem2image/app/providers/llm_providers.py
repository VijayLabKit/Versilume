from __future__ import annotations

import asyncio
import logging

from app.providers.base import LLMProvider, ProviderUnavailableError

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        if not self.api_key:
            raise ProviderUnavailableError("GEMINI_API_KEY is not set.")
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderUnavailableError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            ) from exc

        try:
            genai.configure(api_key=self.api_key)
            models_to_try = [
                self.model_name,
                "gemini-3.7-flash",
                "gemini-3.6-flash",
                "gemini-flash-latest",
                "gemini-2.5-flash",
            ]
            last_err = None
            for model_id in models_to_try:
                try:
                    model = genai.GenerativeModel(model_id, system_instruction=system or None)

                    def _call():
                        return model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=temperature,
                                max_output_tokens=max_tokens,
                            ),
                        )

                    response = await asyncio.to_thread(_call)
                    text = getattr(response, "text", None)
                    if text:
                        return text.strip()
                except Exception as exc:
                    last_err = exc
                    continue

            raise ProviderUnavailableError(f"Gemini request failed on all candidate models: {last_err}")
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Gemini call failed: %s", exc)
            raise ProviderUnavailableError(f"Gemini request failed: {exc}") from exc


class HuggingFaceLLMProvider(LLMProvider):
    name = "huggingface"

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        if not self.api_key:
            raise ProviderUnavailableError("HF_API_TOKEN is not set.")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ProviderUnavailableError("huggingface_hub not installed.") from exc

        try:
            client = InferenceClient(token=self.api_key)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            def _call():
                return client.chat_completion(
                    messages=messages,
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            response = await asyncio.to_thread(_call)
            content = response.choices[0].message.content
            if not content:
                raise ProviderUnavailableError("HuggingFace returned an empty response.")
            return content.strip()
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            logger.warning("HuggingFace LLM call failed: %s", exc)
            raise ProviderUnavailableError(f"HuggingFace request failed: {exc}") from exc


class NullLLMProvider(LLMProvider):
    """No-op fallback — always succeeds by returning an empty string."""
    name = "none"

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        return ""
