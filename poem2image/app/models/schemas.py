from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedElements(BaseModel):
    summary: str
    dominant_emotion: str
    emotion_scores: dict[str, float]
    theme: str
    theme_score: float
    visual_elements: List[str]

    # Source tracking: "heuristic" | "agent"
    emotion_source: str = "heuristic"
    visual_elements_source: str = "heuristic"

    # Fields populated by the 3 agents
    emotion_nuance: str = ""
    emotion_intensity: float = 0.6
    symbols: List[str] = []
    cultural_context: str = ""
    style_cues: List[str] = []
    palette: str = ""
    agents_used: List[str] = []


class RefinementInfo(BaseModel):
    rounds_run: int
    converged: bool
    score_history: List[float]
    final_alignment_score: float


class TranslationInfo(BaseModel):
    was_translated: bool
    source_language_code: str
    source_language_name: str
    original_text: str
    provider_used: str = "none"


class PoemSegment(BaseModel):
    segment_id: int
    lines: List[str]
    elements: ExtractedElements
    image_prompt: str
    refinement: Optional[RefinementInfo] = None


class PoemAnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    poem: str
    num_segments: int
    segments: List[PoemSegment]
    translation: Optional[TranslationInfo] = None
    model_provider_used: str = "none"


class GeneratedImage(BaseModel):
    model_config = {"protected_namespaces": ()}

    segment_id: int
    image_prompt: str
    image_base64: str
    mime_type: str
    model_used: str
    image_provider: str = ""


class PoemToImageRequest(BaseModel):
    poem: str = Field(..., min_length=1, description="Full poem text")
    force_single_image: bool = Field(False, description="Generate one image for the whole poem")
    style: Optional[str] = Field(None, description="Visual style keyword (e.g. watercolor, oil-painting)")
    num_inference_steps: Optional[int] = Field(None, description="Diffusion steps override")
    source_language: str = Field("auto", description="ISO language code or auto")
    enable_refinement: bool = Field(True, description="Enable prompt refinement loop")


class PoemToImageResponse(BaseModel):
    poem: str
    num_segments: int
    segments: List[PoemSegment]
    images: List[GeneratedImage]
    translation: Optional[TranslationInfo] = None


class LanguageEntry(BaseModel):
    code: str
    name: str


class LanguagesResponse(BaseModel):
    count: int
    languages: List[LanguageEntry]
