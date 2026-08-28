from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path
from typing import List

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

# 30-category theme bank (used for cosine-similarity theme assignment)
THEME_BANK: dict[str, str] = {
    "celestial_and_cosmos": "stars, night sky, starry sky, cosmos, glowing star, diamond in the sky, celestial light, moon, nocturnal beauty, galaxies, planets",
    "wonder_and_magic": "wonder, curiosity, magical mystery, enchanting beauty, imagination, sparkling awe, magical light, spellbinding scenes",
    "peace_and_serenity": "calm, tranquility, quiet stillness, soothing peace, serene atmosphere, gentle slumber, restful quiet",
    "nature": "landscapes, seasons, trees, plants, mountains, ocean, waters, weather, the natural world",
    "childhood_and_nostalgia": "memories of childhood, lullaby, nursery rhyme, home, innocence, sweet nostalgia",
    "dreams_and_fantasy": "dreamworld, imagination, surreal visions, mystical dreamscape, ethereal glow, fantasy",
    "love": "romantic love, affection, longing for a partner, devotion, warmth, tenderness",
    "hope_and_resilience": "hope, perseverance, overcoming hardship, sunrise, optimism, dawn breaking",
    "joy_and_celebration": "happiness, celebration, festivity, delight, radiance, cheerful song",
    "time_and_mortality": "the passage of time, aging, mortality, impermanence, twilight, seasons changing",
    "solitude_and_isolation": "loneliness, isolation, solitude, quiet contemplation, being apart, peaceful wandering",
    "loss_and_grief": "death, mourning, loss of a loved one, grief, farewell, sorrowful parting",
    "war_and_conflict": "war, violence, battle, conflict, soldiers, tempest, storm",
    "friendship_and_companionship": "friendship, camaraderie, loyal companionship, shared journeys, trust between friends",
    "family_and_home": "family bonds, ancestral home, parents and children, hearth and household warmth",
    "identity_and_self_discovery": "self-reflection, personal identity, finding oneself, inner journey, self-realisation",
    "freedom_and_rebellion": "liberty, breaking free, defiance, rebellion, unshackled spirit, open horizons",
    "seasons_and_cycles": "spring bloom, summer heat, autumn decay, winter frost, cyclical renewal of nature",
    "urban_and_modern_life": "city streets, skyscrapers, traffic, modern urban rhythm, neon lights, metropolitan bustle",
    "myth_and_folklore": "legendary figures, folk tales, ancient myths, gods and heroes, fables, nursery-rhyme characters",
    "spirituality_and_faith": "prayer, the divine, sacred ritual, transcendence, faith, spiritual devotion",
    "travel_and_journey": "roads, voyages, wandering, exploration, distant lands, the open path",
    "innocence_and_purity": "childlike wonder, untouched purity, gentle naivety, unspoiled beauty",
    "melancholy_and_longing": "wistful yearning, bittersweet ache, quiet sadness, unfulfilled desire",
    "courage_and_heroism": "bravery, valor, heroic deeds, standing firm against adversity",
    "beauty_and_aesthetics": "physical beauty, art, elegance, aesthetic appreciation, graceful form",
    "chaos_and_disorder": "turmoil, upheaval, storm of confusion, unraveling order, frantic energy",
    "gratitude_and_abundance": "thankfulness, plenty, harvest, blessings, generous abundance",
    "justice_and_morality": "right and wrong, fairness, ethical struggle, moral reckoning",
    "technology_and_progress": "machines, invention, futuristic imagery, scientific progress, industrial change",
}


# 24-label poetic emotion palette (expands raw 7-label DistilRoBERTa output)
POETIC_EMOTION_LABELS: list[str] = [
    "joy", "serenity", "wonder", "awe", "nostalgia", "melancholy", "longing",
    "tenderness", "grief", "dread", "fear", "anger", "triumph", "hope",
    "tranquility", "euphoria", "loneliness", "bittersweetness", "reverence",
    "playfulness", "defiance", "resignation", "ecstasy", "wistfulness",
]

# Keyword overrides for common classifier errors in poetic context
_CELESTIAL_OR_WONDER_CUES = [
    "star", "twinkle", "wonder", "sky", "moon", "lullaby", "sparkle", "glow",
    "diamond in the sky", "daffodil", "woods", "dream", "stream", "meadow",
    "lake", "ocean", "peace", "galaxy", "cosmos", "horizon", "pond", "frog",
    "stillness", "silent", "quiet", "bamboo", "autumn", "zen", "temple", "garden",
    "blossom", "cherry", "ripples", "stone", "mist", "breeze", "valley", "mountain",
]
_DEFIANCE_OR_TRIUMPH_CUES = [
    "path of fire", "fire", "flames", "oath", "pledge", "swear", "struggle",
    "unbroken", "never stop", "never yield", "agneepath", "battle", "courage",
    "brave", "march", "conquer", "victory", "won", "overcame", "stand firm",
]
_GRIEF_CUES = ["mourn", "grave", "funeral", "farewell", "gone forever", "buried", "wept", "tears fell"]
_NOSTALGIA_CUES = ["remember when", "childhood", "used to", "long ago", "old photograph", "faded memory"]
_REVERENCE_CUES = ["sacred", "divine", "prayer", "blessed", "holy", "temple"]
_DREAD_CUES = ["shadow looms", "creeping dark", "something waits", "unseen threat"]


def _harmonize_emotion(raw_label: str, raw_scores: dict[str, float], text_lower: str) -> tuple[str, float]:
    """Map raw 7-label classifier output to the 24-label poetic palette using keyword cues."""

    def has_any(cues: list[str]) -> bool:
        return any(c in text_lower for c in cues)

    if has_any(_DEFIANCE_OR_TRIUMPH_CUES):
        return "defiance", 0.9
    if has_any(_GRIEF_CUES):
        return "grief", 0.9
    if has_any(_REVERENCE_CUES):
        return "reverence", 0.85
    if has_any(_NOSTALGIA_CUES):
        return "nostalgia", 0.85
    if has_any(_DREAD_CUES) and raw_label in ("fear", "anger", "disgust"):
        return "dread", 0.8

    if has_any(_CELESTIAL_OR_WONDER_CUES):
        if raw_label in ("anger", "fear", "disgust"):
            return "wonder", 0.9
        if raw_label == "surprise":
            return "awe", raw_scores.get("surprise", 0.8)
        if raw_label == "joy":
            return "serenity", raw_scores.get("joy", 0.75)

    fallback_map = {
        "joy": "joy", "sadness": "melancholy", "anger": "anger", "fear": "fear",
        "surprise": "wonder", "disgust": "resignation", "neutral": "tranquility",
    }
    refined = fallback_map.get(raw_label, "tranquility")
    confidence = raw_scores.get(raw_label, 0.5)
    return refined, confidence


@functools.lru_cache(maxsize=1)
def _get_summarizer():
    from transformers import pipeline

    settings = get_settings()
    logger.info("Loading summarization model: %s", settings.summarizer_model)
    return pipeline("summarization", model=settings.summarizer_model)


@functools.lru_cache(maxsize=1)
def _get_emotion_classifier():
    from transformers import pipeline

    settings = get_settings()
    logger.info("Loading emotion classification model: %s", settings.emotion_model)
    return pipeline("text-classification", model=settings.emotion_model, top_k=None)


@functools.lru_cache(maxsize=1)
def _get_sentence_encoder():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    logger.info("Loading sentence embedding model: %s", settings.sentence_embedding_model)
    return SentenceTransformer(settings.sentence_embedding_model)


def _load_theme_bank() -> dict[str, str]:
    """Load built-in theme bank, or a calibrated one from Settings.theme_bank_path if configured."""
    settings = get_settings()
    if settings.theme_bank_path:
        path = Path(settings.theme_bank_path)
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    logger.info("Loaded calibrated theme bank from %s (%d themes)", path, len(data))
                    return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load theme bank override at %s (%s); using built-in bank.", path, exc)
    return THEME_BANK


@functools.lru_cache(maxsize=1)
def _get_theme_embeddings() -> dict[str, np.ndarray]:
    """Build theme → embedding lookup. Supports both text descriptions and precomputed centroid vectors."""
    bank = _load_theme_bank()
    encoder = _get_sentence_encoder()

    text_items: dict[str, str] = {}
    vector_items: dict[str, np.ndarray] = {}
    for name, value in bank.items():
        if isinstance(value, str):
            text_items[name] = value
        elif isinstance(value, list):
            vec = np.array(value, dtype=np.float32)
            norm = np.linalg.norm(vec)
            vector_items[name] = vec / norm if norm > 0 else vec
        else:
            logger.warning("Theme bank entry '%s' has unsupported type %s; skipping.", name, type(value))

    if text_items:
        names = list(text_items.keys())
        vectors = encoder.encode(list(text_items.values()), normalize_embeddings=True)
        vector_items.update(dict(zip(names, vectors)))

    return vector_items


@functools.lru_cache(maxsize=1)
def _get_spacy_nlp():
    import spacy

    settings = get_settings()
    try:
        return spacy.load(settings.spacy_model)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{settings.spacy_model}' is not installed. "
            f"Run: python -m spacy download {settings.spacy_model}"
        ) from exc


def summarize(text: str) -> str:
    """Extract or summarize key lines from the poem text."""
    text_clean = text.strip()
    if not text_clean:
        return ""
    lines = [ln.strip() for ln in text_clean.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return " ".join(lines)
    return " ".join(lines[:2])


def classify_emotion_heuristic(text: str) -> tuple[str, dict[str, float]]:
    """Synchronous emotion classification using DistilRoBERTa + poetic harmonizer."""
    text_clean = text.strip()
    if not text_clean:
        return "tranquility", {"tranquility": 1.0}

    try:
        classifier = _get_emotion_classifier()
        raw = classifier(text_clean)[0]
        raw_scores = {item["label"]: float(item["score"]) for item in raw}
        raw_dominant = max(raw_scores, key=raw_scores.get)
    except Exception as exc:
        logger.warning("Emotion classifier fallback triggered: %s", exc)
        raw_scores = {"neutral": 0.6, "joy": 0.2, "surprise": 0.2}
        raw_dominant = "neutral"

    settings = get_settings()
    if not settings.use_expanded_emotion_palette:
        return raw_dominant, raw_scores

    refined_label, confidence = _harmonize_emotion(raw_dominant, raw_scores, text_clean.lower())
    scores = dict(raw_scores)
    scores[refined_label] = confidence
    return refined_label, scores


# Alias kept for backward compatibility with segmentation.py
classify_emotion = classify_emotion_heuristic


def assign_theme(text: str) -> tuple[str, float]:
    """Fast lexical theme assignment against THEME_BANK with zero memory overhead."""
    text = (text or "").strip().lower()
    if not text:
        return "nature", 0.0

    words = set(re.findall(r"\w+", text))
    best_theme, best_score = "nature", 0.1

    for theme_name, keywords_str in THEME_BANK.items():
        kw_set = set(re.findall(r"\w+", keywords_str.lower()))
        common = len(words & kw_set)
        if common > 0:
            score = common / (len(words) + len(kw_set))
            if score > best_score:
                best_theme, best_score = theme_name, score

    return best_theme, min(0.95, 0.4 + (best_score * 3.0))


def extract_visual_elements_heuristic(text: str, max_elements: int = 6) -> List[str]:
    """Extract concrete visual subjects via spaCy NER and noun-chunk analysis."""
    text = text.strip()
    if not text:
        return []

    try:
        nlp = _get_spacy_nlp()
        doc = nlp(text)

        elements: list[str] = []
        seen: set[str] = set()

        stop_nouns = {
            "world", "thing", "way", "place", "part", "side", "time", "day", "night",
            "someone", "everyone", "nobody", "bit", "lot", "kind", "line", "word", "hand", "foot",
            "how", "what", "when", "where", "twinkle", "fall", "great fall", "action",
        }

        for ent in doc.ents:
            raw_ent = re.sub(r"[\r\n,;:\.\?\!]+", " ", ent.text).strip()
            raw_ent = re.sub(r"\s+", " ", raw_ent)
            key = raw_ent.lower()
            if key and key not in seen and key not in stop_nouns and len(key) > 2:
                elements.append(raw_ent)
                seen.add(key)

        for chunk in doc.noun_chunks:
            head = chunk.root
            if head.pos_ in ("NOUN", "PROPN"):
                raw_chunk = re.sub(r"[\r\n,;:\.\?\!]+", " ", chunk.text).strip()
                raw_chunk = re.sub(r"\s+", " ", raw_chunk)
                key = raw_chunk.lower()
                key = re.sub(r"^(the|a|an|your|my|his|her|its|our|their|this|that)\s+", "", key)
                key = re.sub(r"\s+(how|when|what|where|who|why|if|then|and|or|but|as|so)$", "", key).strip()
                if key and key not in seen and len(key) > 2 and key not in stop_nouns:
                    elements.append(key)
                    seen.add(key)

        return elements[:max_elements] if elements else [w for w in re.findall(r"\w+", text) if len(w) > 3][:4]
    except Exception as exc:
        logger.warning("spaCy extraction fallback: %s", exc)
        return [w for w in re.findall(r"\w+", text) if len(w) > 3][:4]


# Backward-compatible alias
extract_visual_elements = extract_visual_elements_heuristic


async def extract_elements(text: str, language: str = "English") -> "ExtractedElements":
    """
    Full extraction pipeline for one segment.
    Runs cloud LLM agents first (fast, rich, 0MB RAM).
    Falls back to lightweight heuristics only if an agent fails.
    """
    from app.models.schemas import ExtractedElements
    from app.services.agents.orchestrator import run_agents

    summary = summarize(text)

    # ── Step 1: Run agents concurrently (fast cloud LLMs) ─────────────────
    agent_result = await run_agents(text, language=language)

    # ── Step 2: Resolve emotion ───────────────────────────────────────────
    if agent_result.emotion_source == "agent":
        final_emotion = agent_result.emotion
        final_scores = {final_emotion: agent_result.emotion_intensity}
    else:
        final_emotion, final_scores = classify_emotion_heuristic(text)

    # ── Step 3: Resolve theme ─────────────────────────────────────────────
    if agent_result.semantic_source == "agent":
        final_theme = agent_result.theme
        theme_score = 0.88
    else:
        final_theme, theme_score = assign_theme(text)

    # ── Step 4: Resolve visuals ───────────────────────────────────────────
    final_visuals = agent_result.visuals if agent_result.visual_source == "agent" else extract_visual_elements_heuristic(text)

    return ExtractedElements(
        summary=summary,
        dominant_emotion=final_emotion,
        emotion_scores=final_scores,
        theme=final_theme,
        theme_score=theme_score,
        visual_elements=final_visuals,
        # Source tracing
        emotion_source=agent_result.emotion_source,
        visual_elements_source=agent_result.visual_source,
        # Agent enrichments
        emotion_nuance=agent_result.emotion_nuance,
        emotion_intensity=agent_result.emotion_intensity,
        symbols=agent_result.symbols,
        cultural_context=agent_result.cultural_context,
        style_cues=agent_result.style_cues,
        palette=agent_result.palette,
        agents_used=agent_result.agents_used,
    )

