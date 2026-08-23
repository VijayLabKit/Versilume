from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.config import get_settings
from app.providers.base import ProviderUnavailableError

logger = logging.getLogger(__name__)

_EMOTION_LABELS = [
    "joy", "serenity", "wonder", "awe", "nostalgia", "melancholy", "longing",
    "tenderness", "grief", "dread", "fear", "anger", "triumph", "hope",
    "tranquility", "euphoria", "loneliness", "bittersweetness", "reverence",
    "playfulness", "defiance", "resignation", "ecstasy", "wistfulness",
]

_SYSTEM_PROMPT = (
    "You are an expert literary analyst specialising in the emotional register of poetry. "
    "Respond with valid JSON only."
)

_USER_PROMPT = """\
Analyse the emotional register of this poem or poem segment.

Poem:
{poem}

Respond with a JSON object with exactly these keys:
- "emotion": one label from: {labels}
- "nuance": one sentence (max 20 words) describing the specific emotional tone
- "intensity": float 0.0-1.0

Example: {{"emotion": "melancholy", "nuance": "A quiet ache for a home that can never be returned to.", "intensity": 0.75}}

Respond with ONLY the JSON object."""


@dataclass
class EmotionResult:
    emotion: str
    nuance: str
    intensity: float
    source: str = "agent"


async def _call_agent(poem_text: str) -> EmotionResult:
    settings = get_settings()
    try:
        from app.providers.llm_providers import HuggingFaceLLMProvider
    except ImportError as exc:
        raise ProviderUnavailableError("HuggingFaceLLMProvider unavailable") from exc

    provider = HuggingFaceLLMProvider(
        model_name=settings.emotion_agent_model,
        api_key=settings.hf_api_token,
    )
    prompt = _USER_PROMPT.format(poem=poem_text, labels=", ".join(_EMOTION_LABELS))
    raw = await provider.generate(prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.2, max_tokens=settings.agent_max_tokens)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(cleaned)

    emotion = str(data.get("emotion", "")).strip().lower()
    if emotion not in _EMOTION_LABELS:
        raise ValueError(f"Unknown emotion label: {emotion!r}")

    return EmotionResult(
        emotion=emotion,
        nuance=str(data.get("nuance", "")).strip(),
        intensity=float(data.get("intensity", 0.7)),
        source="agent",
    )


async def analyse_emotion(poem_text: str) -> EmotionResult:
    settings = get_settings()
    if not settings.agents_enabled or not settings.hf_api_token:
        return _heuristic_fallback(poem_text)

    try:
        result = await _call_agent(poem_text)
        logger.info("EmotionAgent: '%s' (%.2f) source=agent", result.emotion, result.intensity)
        return result
    except ProviderUnavailableError as exc:
        logger.warning("EmotionAgent unavailable (%s); using heuristic.", exc)
    except (json.JSONDecodeError, KeyError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("EmotionAgent parse failed (%s); using heuristic.", exc)

    return _heuristic_fallback(poem_text)


def _heuristic_fallback(poem_text: str) -> EmotionResult:
    from app.services.nlp_pipeline import classify_emotion_heuristic
    label, scores = classify_emotion_heuristic(poem_text)
    return EmotionResult(
        emotion=label,
        nuance=f"Dominant emotional register: {label}.",
        intensity=float(scores.get(label, 0.6)),
        source="heuristic",
    )
