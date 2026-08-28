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
    "You are a literary scholar and cultural ethnographer with expertise in world poetry, symbolism, "
    "and regional heritage. Respond with valid JSON only."
)

_USER_PROMPT = """\
Perform a deep semantic and cultural analysis of this {language} poem.

Poem:
{poem}

Respond with a JSON object with exactly these keys:
- "theme": one key from: {themes}
- "symbols": array of 2-5 short strings naming key symbolic visual objects or metaphors
- "cultural_context": one vivid sentence (max 25 words) describing the authentic {language} cultural, regional, or historical setting

Example for Bengali poem: {{"theme": "joy_and_celebration", "symbols": ["sunlit rain clouds", "green monsoon meadows", "free children"], "cultural_context": "Rural Bengal monsoon landscape with joyful children in traditional cotton clothing."}}

Respond with ONLY the JSON object."""


@dataclass
class SemanticResult:
    theme: str
    symbols: List[str] = field(default_factory=list)
    cultural_context: str = ""
    source: str = "agent"


async def _call_agent(poem_text: str, language: str = "English") -> SemanticResult:
    from app.providers.registry import get_llm_provider

    provider = get_llm_provider()
    if provider.name == "none":
        raise ProviderUnavailableError("No LLM provider available")

    settings = get_settings()
    prompt = _USER_PROMPT.format(poem=poem_text, language=language, themes=", ".join(_THEME_KEYS))
    raw = await provider.generate(prompt=prompt, system=_SYSTEM_PROMPT, temperature=0.3, max_tokens=settings.agent_max_tokens)
    
    # Robust multi-pattern JSON parsing
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    json_str = json_match.group(0) if json_match else cleaned
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback regex extraction for keys
        theme_m = re.search(r'"theme"\s*:\s*"([^"]+)"', raw)
        symbols_m = re.findall(r'"([^"]+)"', raw)
        context_m = re.search(r'"cultural_context"\s*:\s*"([^"]+)"', raw)
        data = {
            "theme": theme_m.group(1) if theme_m else "nature",
            "symbols": [s for s in symbols_m if s not in ("theme", "symbols", "cultural_context")][:4],
            "cultural_context": context_m.group(1) if context_m else "",
        }

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


async def analyse_semantics(poem_text: str, language: str = "English") -> SemanticResult:
    settings = get_settings()
    if not settings.agents_enabled or not settings.hf_api_token:
        return _heuristic_fallback(poem_text)

    try:
        result = await _call_agent(poem_text, language=language)
        logger.info("SemanticAgent (%s): theme='%s', %d symbols, source=agent", language, result.theme, len(result.symbols))
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
