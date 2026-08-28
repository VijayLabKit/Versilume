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
    "You are an expert visual director and cultural ethnographer translating world poetry into vivid, "
    "concrete, paintable imagery. Resolve ALL abstract metaphors into literal, tangible physical scenes. "
    "Respond with valid JSON only."
)

_USER_PROMPT = """\
Translate this {language} poem into paintable, evocative visual imagery.

Poem:
{poem}

CRITICAL RULES:
1. "visuals": Array of 3-6 concrete visual elements (who is in the scene, what are they doing, what is the environment, what are the key physical objects).
2. CULTURAL & REGIONAL GROUNDING: If the poem is in {language} (e.g. Japanese, Bengali, Hindi, Persian, Spanish, French, Chinese, Arabic, etc.), ALL human figures, clothing, houses/architecture, flora, and landscape MUST authentically reflect {language} regional heritage (e.g., traditional Japanese wooden structures, tatami, kimonos, stone lanterns; or rural Bengal village paths, monsoon greenery, cotton kurtas/dhotis; or Indian fire-lit paths and banyan trees; or classical European stone buildings).
3. NO modern anachronisms (NO modern paved highways, NO modern cars or Western suits unless specifically written in the poem).
4. "style_cues": Array of 2-3 composition/lighting cues (e.g., "dramatic firelight illumination", "wide-angle misty riverbank composition").
5. "palette": One string with the dominant color palette.

Respond with ONLY this JSON format:
{{"visuals": ["...", "..."], "style_cues": ["...", "..."], "palette": "..."}}"""


@dataclass
class MetaphorResult:
    visuals: List[str] = field(default_factory=list)
    style_cues: List[str] = field(default_factory=list)
    palette: str = ""
    source: str = "agent"


async def _call_agent(poem_text: str, language: str = "English") -> MetaphorResult:
    from app.providers.registry import get_llm_provider

    provider = get_llm_provider()
    if provider.name == "none":
        raise ProviderUnavailableError("No LLM provider available")

    settings = get_settings()
    raw = await provider.generate(
        prompt=_USER_PROMPT.format(poem=poem_text, language=language),
        system=_SYSTEM_PROMPT,
        temperature=0.4,
        max_tokens=settings.agent_max_tokens,
    )
    # Robust multi-pattern JSON parsing
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    json_str = json_match.group(0) if json_match else cleaned
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback regex extraction for visuals
        visuals_m = re.findall(r'"([^"]{4,})"', raw)
        data = {
            "visuals": [v for v in visuals_m if v not in ("visuals", "style_cues", "palette")][:5],
            "style_cues": [],
            "palette": ""
        }

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


async def analyse_metaphors(poem_text: str, language: str = "English") -> MetaphorResult:
    settings = get_settings()
    if not settings.agents_enabled or not settings.hf_api_token:
        return _heuristic_fallback(poem_text)

    try:
        result = await _call_agent(poem_text, language=language)
        logger.info("MetaphorAgent (%s): %d visuals, %d style_cues, source=agent", language, len(result.visuals), len(result.style_cues))
        return result
    except ProviderUnavailableError as exc:
        logger.warning("MetaphorAgent unavailable (%s); using heuristic.", exc)
    except (json.JSONDecodeError, KeyError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("MetaphorAgent parse failed (%s); using heuristic.", exc)

    return _heuristic_fallback(poem_text)

    return _heuristic_fallback(poem_text)


def _heuristic_fallback(poem_text: str) -> MetaphorResult:
    from app.services.nlp_pipeline import extract_visual_elements_heuristic
    return MetaphorResult(
        visuals=extract_visual_elements_heuristic(poem_text, max_elements=6),
        style_cues=[],
        palette="",
        source="heuristic",
    )
