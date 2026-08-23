from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.data.languages import SUPPORTED_LANGUAGES
from app.models.schemas import (
    GeneratedImage,
    LanguageEntry,
    LanguagesResponse,
    PoemAnalysisResponse,
    PoemToImageRequest,
    PoemToImageResponse,
)
from app.providers.registry import get_image_provider_chain, get_llm_provider
from app.services.image_client import ImageGenerationError, ModelLoadingError, generate_image
from app.services.pipeline import analyze_poem
from app.services.prompt_builder import build_negative_prompt

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health() -> dict:
    llm = get_llm_provider()
    image_chain = get_image_provider_chain()
    return {
        "status": "ok",
        "model_provider": llm.name,
        "image_provider_chain": [p.name for p in image_chain],
    }


@router.get("/languages", response_model=LanguagesResponse)
async def list_languages() -> LanguagesResponse:
    """List all supported source languages."""
    entries = [LanguageEntry(code=code, name=name) for code, name in sorted(SUPPORTED_LANGUAGES.items())]
    return LanguagesResponse(count=len(entries), languages=entries)


@router.post("/analyze", response_model=PoemAnalysisResponse)
async def analyze(request: PoemToImageRequest) -> PoemAnalysisResponse:
    """Run poem analysis and prompt building without image generation."""
    result = await analyze_poem(
        request.poem,
        force_single_image=request.force_single_image,
        style=request.style,
        source_language=request.source_language,
        enable_refinement=request.enable_refinement,
    )
    return PoemAnalysisResponse(
        poem=request.poem,
        num_segments=len(result.segments),
        segments=result.segments,
        translation=result.translation,
        model_provider_used=result.model_provider_used,
    )


@router.post("/generate", response_model=PoemToImageResponse)
async def generate(request: PoemToImageRequest) -> PoemToImageResponse:
    """Run full pipeline and generate images per segment."""
    result = await analyze_poem(
        request.poem,
        force_single_image=request.force_single_image,
        style=request.style,
        source_language=request.source_language,
        enable_refinement=request.enable_refinement,
    )
    if not result.segments:
        raise HTTPException(status_code=422, detail="Poem is empty.")

    negative_prompt = build_negative_prompt()
    images: list[GeneratedImage] = []

    for segment in result.segments:
        try:
            image_result = await generate_image(
                prompt=segment.image_prompt,
                negative_prompt=negative_prompt,
            )
        except ModelLoadingError as exc:
            logger.warning("ModelLoadingError: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ImageGenerationError as exc:
            logger.error("ImageGenerationError: %s", exc, exc_info=True)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        images.append(
            GeneratedImage(
                segment_id=segment.segment_id,
                image_prompt=segment.image_prompt,
                image_base64=image_result.base64,
                mime_type=image_result.mime_type,
                model_used=image_result.model_used,
                image_provider=image_result.image_provider,
            )
        )

    return PoemToImageResponse(
        poem=request.poem,
        num_segments=len(result.segments),
        segments=result.segments,
        images=images,
        translation=result.translation,
    )
