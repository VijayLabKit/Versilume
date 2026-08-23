from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List

from app.config import get_settings
from app.providers.base import ProviderUnavailableError

logger = logging.getLogger(__name__)

_THEME_KEYS = [
    "celestial_and_cosmos", "wonder_and_magic", "peace_and_serenity", "nature",
    "childhood_and_nostalgia", "dreams_and_fantasy", "love", "hope_and_resilience",
    "joy_and_celebration", "time_and_mortality", "solitude_and_isolation",
    "loss_and_grief", "war_and_conflict", "friendship_and_companionship",
    "family_and_home", "identity_and_self_discovery", "freedom_and_rebellion",
    "seasons_and_cycles", "urban_and_modern_life", "myth_and_folklore",
    "spirituality_and_faith", "travel_and_journey", "innocence_and_purity",
    "melancholy_and_longing", "courage_and_heroism", "beauty_and_aesthetics",
    "chaos_and_disorder", "gratitude_and_abundance", "justice_and_morality",
    "technology_and_progress",
]

_SYSTEM_PROMPT = (
    "You are a literary scholar with expertise in world poetry and symbolism. "
    "Respond with valid JSON only."
)

_USER_PROMPT = """\
Perform a semantic analysis of this poem or poem segment.

Poem:
{poem}

Respond with a JSON object with exactly these keys:
- "theme": one key from: {themes}
- "symbols": array of 2-5 short strings naming key symbolic images
- "cultural_context": one sentence (max 25 words) on the literary or cultural tradition

Example: {{"theme": "melancholy_and_longing", "symbols": ["fading rose", "empty chair"], "cultural_context": "Romantic-era European lyric poetry."}}

Respond with ONLY the JSON object."""


@dataclass
class SemanticResult:
    theme: str
    symbols: List[str] = field(default_factory=list)
    cultural_context: str = ""
    source: str = "agent"


async def _call_agent(poem_text: str) -> SemanticResult:
    settings = get_settings()
    try:
        from app.providers.llm_providers import HuggingFaceLLMProvider
    except ImportError as exc:
        raise ProviderUnavailableError("HuggingFaceLLMProvider unavailable") from exc

    provider = HuggingFaceLLMProvider(
        model_name=settings.semantic_agent_model,
        api_key=settings.hf_api_token,
    )
    prompt = _USER_PROMPT.format(poem=poem_text, themes=", ".join(_THEME_KEYS))
    raw = await provider.generate(prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.3, max_tokens=settings.agent_max_tokens)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    data = json.loads(cleaned)

    theme = str(data.get("theme", "")).strip().lower().replace(" ", "_")
    if theme not in _THEME_KEYS:
        match = next((t for t in _THEME_KEYS if t.startswith(theme[:6])), None)
        theme = match if match else "nature"

    symbols = data.get("symbols", [])
    if not isinstance(symbols, list):
        symbols = []

    return SemanticResult(
        theme=theme,
        symbols=[str(s).strip() for s in symbols if s][:5],
        cultural_context=str(data.get("cultural_context", "")).strip(),
        source="agent",
    )


async def analyse_semantics(poem_text: str) -> SemanticResult:
    settings = get_settings()
    if not settings.agents_enabled or not settings.hf_api_token:
        return _heuristic_fallback(poem_text)

    try:
        result = await _call_agent(poem_text)
        logger.info("SemanticAgent: theme='%s', %d symbols, source=agent", result.theme, len(result.symbols))
        return result
    except ProviderUnavailableError as exc:
        logger.warning("SemanticAgent unavailable (%s); using heuristic.", exc)
    except (json.JSONDecodeError, KeyError, ValueError, Exception) as exc:  # noqa: BLE001
        logger.warning("SemanticAgent parse failed (%s); using heuristic.", exc)

    return _heuristic_fallback(poem_text)


def _heuristic_fallback(poem_text: str) -> SemanticResult:
    from app.services.nlp_pipeline import assign_theme, extract_visual_elements_heuristic
    theme, _ = assign_theme(poem_text)
    return SemanticResult(
        theme=theme,
        symbols=extract_visual_elements_heuristic(poem_text, max_elements=4),
        cultural_context="",
        source="heuristic",
    )
