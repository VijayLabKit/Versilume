from __future__ import annotations

import logging
from dataclasses import dataclass

from app.data.languages import SUPPORTED_LANGUAGES, is_supported, language_name
from app.providers.base import ProviderUnavailableError
from app.providers.registry import get_llm_provider

logger = logging.getLogger(__name__)

_TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert literary translator and cultural poet. When translating a poem, "
    "preserve its literal and metaphorical visual imagery, emotional cadence, characters, "
    "and cultural context. "
    "Return ONLY the translated poem in clear English, with no preamble, no commentary, "
    "and no quotation marks."
)


@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_language_code: str
    source_language_name: str
    was_translated: bool
    provider_used: str


def detect_language(text: str) -> str:
    """Detect language code, defaulting to 'en' on failure."""
    try:
        from langdetect import detect

        code = detect(text)
        code = {"zh-cn": "zh", "zh-tw": "zh"}.get(code, code)
        return code if is_supported(code) else "en"
    except Exception as exc:  # noqa: BLE001
        logger.info("Language detection failed (%s); defaulting to English.", exc)
        return "en"


async def translate_poem(text: str, source_language: str = "auto") -> TranslationResult:
    """Translate poem text into English via the configured LLM provider."""
    lang_code = source_language.lower().strip()
    if lang_code == "auto":
        lang_code = detect_language(text)

    if lang_code not in SUPPORTED_LANGUAGES:
        logger.warning("Language code '%s' not supported; treating as English.", lang_code)
        lang_code = "en"

    if lang_code == "en":
        return TranslationResult(
            original_text=text,
            translated_text=text,
            source_language_code="en",
            source_language_name="English",
            was_translated=False,
            provider_used="none",
        )

    provider = get_llm_provider()
    lang_name = language_name(lang_code)
    prompt = (
        f"Translate the following {lang_name} poem into English, preserving its "
        f"poetic essence rather than translating literally:\n\n{text}"
    )

    try:
        translated = await provider.generate(
            prompt=prompt,
            system=_TRANSLATION_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=800,
        )
        if not translated:
            raise ProviderUnavailableError("Empty translation result.")
        return TranslationResult(
            original_text=text,
            translated_text=translated,
            source_language_code=lang_code,
            source_language_name=lang_name,
            was_translated=True,
            provider_used=provider.name,
        )
    except ProviderUnavailableError as exc:
        logger.warning("Translation unavailable (%s); keeping original text.", exc)
        return TranslationResult(
            original_text=text,
            translated_text=text,
            source_language_code=lang_code,
            source_language_name=lang_name,
            was_translated=False,
            provider_used="none",
        )
