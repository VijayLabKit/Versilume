from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # Text model — Gemini for translation and prompt refinement
    model_provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    gemini_api_key: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 400

    # Image generation fallback chain
    image_model_provider: str = "huggingface,pollinations"
    image_model_name: str = "black-forest-labs/FLUX.1-schnell"

    # Hugging Face — used by agents and image generation
    hf_api_token: str = ""
    hf_request_timeout_s: float = 120.0

    # Local NLP models (downloaded on first run)
    summarizer_model: str = "facebook/bart-large-cnn"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    sentence_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"

    # Multi-Agent Configuration (Emotion, Semantic, Metaphor)
    agents_enabled: bool = True
    emotion_agent_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    semantic_agent_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    metaphor_agent_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    agent_max_tokens: int = 450

    # Optional: path to a calibrated theme bank JSON file
    theme_bank_path: str = ""

    use_expanded_emotion_palette: bool = True

    # Translation
    translation_enabled: bool = True
    default_source_language: str = "auto"

    # Segmentation
    max_words_per_summary: int = 60
    segment_on_emotion_shift: bool = True
    segment_on_entity_shift: bool = True
    max_segments_per_poem: int = 5
    min_lines_per_segment: int = 2

    # Prompt refinement
    prompt_refinement_enabled: bool = True
    prompt_refinement_max_rounds: int = 10
    prompt_refinement_convergence_epsilon: float = 0.01
    prompt_refinement_patience: int = 3

    # Image output
    image_width: int = 1024
    image_height: int = 1024
    default_num_images: int = 1
    output_dir: str = "generated_images"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def refresh_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
