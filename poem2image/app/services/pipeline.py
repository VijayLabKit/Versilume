from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings
from app.models.schemas import ExtractedElements, PoemSegment, RefinementInfo, TranslationInfo
from app.providers.registry import get_llm_provider
from app.services.nlp_pipeline import extract_elements
from app.services.prompt_builder import build_image_prompt
from app.services.prompt_refinement import refine_prompt
from app.services.segmentation import segment_poem
from app.services.translation import translate_poem


@dataclass
class AnalysisResult:
    segments: list[PoemSegment]
    translation: TranslationInfo | None
    model_provider_used: str


async def analyze_poem(
    poem: str,
    force_single_image: bool = False,
    style: str | None = None,
    source_language: str = "auto",
    enable_refinement: bool = True,
) -> AnalysisResult:
    """
    Full pipeline:
      1. Gemini translates the poem to English (if needed)
      2. EPE segmentation splits it into visual scenes
      3. Three agents (emotion, semantic, metaphor) analyse each segment in parallel
      4. Prompt builder assembles the final diffusion prompt
      5. Gemini MSPR loop refines the prompt (if enabled)
    """
    settings = get_settings()

    translation_result = None
    working_poem = poem
    if settings.translation_enabled:
        translation_result = await translate_poem(poem, source_language=source_language)
        working_poem = translation_result.translated_text

    if force_single_image:
        line_segments = [[ln for ln in working_poem.splitlines() if ln.strip()]]
    else:
        line_segments = segment_poem(working_poem)

    results: list[PoemSegment] = []
    for idx, lines in enumerate(line_segments):
        segment_text = "\n".join(lines)
        elements: ExtractedElements = await extract_elements(segment_text)
        base_prompt = build_image_prompt(elements, style=style)

        refinement_info = None
        final_prompt = base_prompt
        if enable_refinement and settings.prompt_refinement_enabled:
            refinement_result = await refine_prompt(base_prompt, segment_text)
            final_prompt = refinement_result.final_prompt
            refinement_info = RefinementInfo(
                rounds_run=refinement_result.rounds_run,
                converged=refinement_result.converged,
                score_history=refinement_result.score_history,
                final_alignment_score=refinement_result.final_alignment_score,
            )

        results.append(
            PoemSegment(
                segment_id=idx,
                lines=lines,
                elements=elements,
                image_prompt=final_prompt,
                refinement=refinement_info,
            )
        )

    translation_info = None
    if translation_result is not None:
        translation_info = TranslationInfo(
            was_translated=translation_result.was_translated,
            source_language_code=translation_result.source_language_code,
            source_language_name=translation_result.source_language_name,
            original_text=translation_result.original_text,
            provider_used=translation_result.provider_used,
        )

    return AnalysisResult(
        segments=results,
        translation=translation_info,
        model_provider_used=get_llm_provider().name,
    )
