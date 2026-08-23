from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

from app.config import get_settings
from app.providers.base import ProviderUnavailableError

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a visual storytelling expert who translates poetry into concrete paintable imagery. "
    "Resolve ALL metaphors into their literal visual equivalent. "
    "Respond with valid JSON only."
)

_USER_PROMPT = """\
Read this poem and translate it into paintable visual imagery.

Poem:
{poem}

Respond with a JSON object with exactly these keys:
- "visuals": array of 3-6 short concrete visual phrases (resolve ALL metaphors into literal scenes)
- "style_cues": array of 2-3 composition/lighting cues (e.g. "golden hour side-lighting")
- "palette": one string describing the dominant colour palette

Example: {{"visuals": ["a glowing star in dark sky", "child lying in grass gazing upward"], "style_cues": ["soft ambient lighting", "low-angle upward composition"], "palette": "midnight blue and silver starlight"}}

Respond with ONLY the JSON object."""


@dataclass
class MetaphorResult:
    visuals: List[str] = field(default_factory=list)
    style_cues: List[str] = field(default_factory=list)
    palette: str = ""
    source: str = "agent"


async def _call_agent(poem_text: str) -> MetaphorResult:
    settings = get_settings()
    try:
        from app.providers.llm_providers import HuggingFaceLLMProvider
    except ImportError as exc:
        raise ProviderUnavailableError("HuggingFaceLLMProvider unavailable") from exc

    provider = HuggingFaceLLMProvider(
        model_name=settings.metaphor_agent_model,
        api_key=settings.hf_api_token,
    )
    raw = await provider.generate(
        prompt=_USER_PROMPT.format(poem=poem_text),
        system=_SYSTEM_PROMPT,
        temperature=0.5,
        max_tokens=settings.agent_max_tokens,
    )
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(cleaned)

    visuals = [str(v).strip() for v in data.get("visuals", []) if v][:6]
    style_cues = [str(s).strip() for s in data.get("style_cues", []) if s][:3]

    if not visuals:
        raise ValueError("MetaphorAgent returned empty visuals list")

    return MetaphorResult(
        visuals=visuals,
        style_cues=style_cues,
        palette=str(data.get("palette", "")).strip(),
        source="agent",
    )


async def analyse_metaphors(poem_text: str) -> MetaphorResult:
    settings = get_settings()
    if not settings.agents_enabled or not settings.hf_api_token:
        return _heuristic_fallback(poem_text)

    try:
        result = await _call_agent(poem_text)
        logger.info("MetaphorAgent: %d visuals, %d style_cues, source=agent", len(result.visuals), len(result.style_cues))
        return result
    except ProviderUnavailableError as exc:
        logger.warning("MetaphorAgent unavailable (%s); using heuristic.", exc)
    except (json.JSONDecodeError, KeyError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("MetaphorAgent parse failed (%s); using heuristic.", exc)

    return _heuristic_fallback(poem_text)


def _heuristic_fallback(poem_text: str) -> MetaphorResult:
    from app.services.nlp_pipeline import extract_visual_elements_heuristic
    return MetaphorResult(
        visuals=extract_visual_elements_heuristic(poem_text, max_elements=6),
        style_cues=[],
        palette="",
        source="heuristic",
    )
