from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List

from app.services.agents.emotion_agent import EmotionResult, analyse_emotion
from app.services.agents.metaphor_agent import MetaphorResult, analyse_metaphors
from app.services.agents.semantic_agent import SemanticResult, analyse_semantics

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    # EmotionAgent outputs
    emotion: str = "tranquility"
    emotion_nuance: str = ""
    emotion_intensity: float = 0.6
    emotion_source: str = "heuristic"

    # SemanticAgent outputs
    theme: str = "nature"
    symbols: List[str] = field(default_factory=list)
    cultural_context: str = ""
    semantic_source: str = "heuristic"

    # MetaphorAgent outputs
    visuals: List[str] = field(default_factory=list)
    style_cues: List[str] = field(default_factory=list)
    palette: str = ""
    visual_source: str = "heuristic"

    @property
    def agents_used(self) -> List[str]:
        used = []
        if self.emotion_source == "agent":
            used.append("emotion")
        if self.semantic_source == "agent":
            used.append("semantic")
        if self.visual_source == "agent":
            used.append("metaphor")
        return used


async def run_agents(poem_text: str) -> AgentResult:
    """Run all three agents concurrently. Each agent handles its own fallback."""
    logger.info("Running 3 agents on %d-char segment.", len(poem_text))

    results = await asyncio.gather(
        analyse_emotion(poem_text),
        analyse_semantics(poem_text),
        analyse_metaphors(poem_text),
        return_exceptions=True,
    )

    emotion_res, semantic_res, metaphor_res = results

    if isinstance(emotion_res, BaseException):
        logger.error("EmotionAgent raised unexpectedly: %s", emotion_res)
        from app.services.agents.emotion_agent import _heuristic_fallback as _ef
        emotion_res = _ef(poem_text)

    if isinstance(semantic_res, BaseException):
        logger.error("SemanticAgent raised unexpectedly: %s", semantic_res)
        from app.services.agents.semantic_agent import _heuristic_fallback as _sf
        semantic_res = _sf(poem_text)

    if isinstance(metaphor_res, BaseException):
        logger.error("MetaphorAgent raised unexpectedly: %s", metaphor_res)
        from app.services.agents.metaphor_agent import _heuristic_fallback as _mf
        metaphor_res = _mf(poem_text)

    emotion_res: EmotionResult
    semantic_res: SemanticResult
    metaphor_res: MetaphorResult

    result = AgentResult(
        emotion=emotion_res.emotion,
        emotion_nuance=emotion_res.nuance,
        emotion_intensity=emotion_res.intensity,
        emotion_source=emotion_res.source,
        theme=semantic_res.theme,
        symbols=semantic_res.symbols,
        cultural_context=semantic_res.cultural_context,
        semantic_source=semantic_res.source,
        visuals=metaphor_res.visuals,
        style_cues=metaphor_res.style_cues,
        palette=metaphor_res.palette,
        visual_source=metaphor_res.source,
    )

    logger.info(
        "Agents done — emotion='%s'(%s), theme='%s'(%s), visuals=%d(%s). Active: %s",
        result.emotion, result.emotion_source,
        result.theme, result.semantic_source,
        len(result.visuals), result.visual_source,
        result.agents_used or "none",
    )
    return result
