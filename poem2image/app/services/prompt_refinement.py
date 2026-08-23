from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import get_settings
from app.providers.base import ProviderUnavailableError
from app.providers.registry import get_llm_provider
from app.services.scoring import compute_alignment_score

logger = logging.getLogger(__name__)

_REFINEMENT_SYSTEM_PROMPT = (
    "You are a visual-storytelling expert refining an image-generation prompt "
    "for a poem. Deepen the emotional and visual resonance while staying "
    "faithful to the poem's meaning. "
    "Respond with ONLY the refined prompt text, under 90 words, no preamble."
)


@dataclass
class RefinementResult:
    final_prompt: str
    rounds_run: int
    converged: bool
    score_history: list[float] = field(default_factory=list)

    @property
    def final_alignment_score(self) -> float:
        return self.score_history[-1] if self.score_history else 0.0


async def refine_prompt(base_prompt: str, poem_text: str) -> RefinementResult:
    """Iteratively refine prompt using the configured LLM until convergence or max rounds."""
    settings = get_settings()
    score0 = compute_alignment_score(base_prompt, poem_text)

    if not settings.prompt_refinement_enabled:
        return RefinementResult(final_prompt=base_prompt, rounds_run=0, converged=False, score_history=[score0])

    provider = get_llm_provider()
    if provider.name == "none":
        return RefinementResult(final_prompt=base_prompt, rounds_run=0, converged=False, score_history=[score0])

    current_prompt = base_prompt
    score_history = [score0]
    best_prompt, best_score = base_prompt, score0
    stagnant_rounds = 0

    for round_num in range(1, settings.prompt_refinement_max_rounds + 1):
        prompt_for_llm = (
            f"Original poem:\n{poem_text}\n\n"
            f"Current image prompt (round {round_num}):\n{current_prompt}\n\n"
            f"Refine this prompt further."
        )
        try:
            refined = await provider.generate(
                prompt=prompt_for_llm,
                system=_REFINEMENT_SYSTEM_PROMPT,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        except ProviderUnavailableError as exc:
            logger.info("Prompt refinement stopped early at round %d (%s).", round_num, exc)
            break

        refined = refined.strip()
        if not refined:
            break

        score = compute_alignment_score(refined, poem_text)
        score_history.append(score)
        current_prompt = refined

        if score > best_score:
            best_prompt, best_score = refined, score

        improvement = score - score_history[-2]
        if improvement < settings.prompt_refinement_convergence_epsilon:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0

        if stagnant_rounds >= settings.prompt_refinement_patience:
            logger.info("Prompt refinement converged after %d rounds.", round_num)
            return RefinementResult(
                final_prompt=best_prompt, rounds_run=round_num, converged=True, score_history=score_history
            )

    return RefinementResult(
        final_prompt=best_prompt,
        rounds_run=len(score_history) - 1,
        converged=False,
        score_history=score_history,
    )
