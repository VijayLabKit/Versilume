from __future__ import annotations

from app.models.schemas import ExtractedElements

_THEME_MOOD_HINTS: dict[str, str] = {
    "celestial_and_cosmos": "deep night sky, glowing starry atmosphere, luminous moonlight radiance",
    "wonder_and_magic": "magical glowing particles, enchanting twilight atmosphere, sparkling wonder",
    "peace_and_serenity": "gentle tranquil ambient light, soothing serene atmosphere, crystal clear water reflections",
    "nature": "golden natural sunlight, lush vibrant scenery, scenic landscape composition",
    "childhood_and_nostalgia": "soft golden-hour storybook light, nostalgic warm tone",
    "dreams_and_fantasy": "ethereal mystical lighting, dreamlike floating atmosphere, surreal beauty",
    "love": "warm intimate lighting, soft romantic focus, tender golden glow",
    "hope_and_resilience": "golden dawn light breaking through, uplifting vibrant composition",
    "joy_and_celebration": "bright vibrant cheerful colors, radiant joyous composition",
    "time_and_mortality": "long dramatic shadows, warm autumnal or twilight tones",
    "solitude_and_isolation": "expansive landscape, peaceful stillness, quiet contemplative tone",
    "loss_and_grief": "muted soft colors, quiet atmospheric mist, somber stillness",
    "war_and_conflict": "dramatic shadow and smoke, dynamic tense composition",
    "friendship_and_companionship": "warm shared light, close natural composition, inviting camaraderie",
    "family_and_home": "cozy interior warmth, soft hearth light, intimate domestic detail",
    "identity_and_self_discovery": "reflective luminous atmosphere, introspective mood",
    "freedom_and_rebellion": "wide open sky, wind-swept motion, dynamic liberated composition",
    "seasons_and_cycles": "seasonal color palette, transitional light, lush natural detail",
    "urban_and_modern_life": "neon-lit street atmosphere, reflective glass, vibrant city night",
    "myth_and_folklore": "epic legendary lighting, storybook grandeur, timeless mythic atmosphere",
    "spirituality_and_faith": "soft sacred light rays, reverent atmosphere, gentle divine glow",
    "travel_and_journey": "scenic open pathway, distant horizon, wandering sunlight",
    "innocence_and_purity": "soft pastel light, gentle untouched clarity, delicate composition",
    "melancholy_and_longing": "muted cool tones, wistful hazy light, quiet yearning atmosphere",
    "courage_and_heroism": "bold dramatic lighting, heroic silhouette, resolute composition",
    "beauty_and_aesthetics": "elegant refined lighting, graceful composition, exquisite detail",
    "chaos_and_disorder": "swirling dynamic energy, turbulent atmosphere",
    "gratitude_and_abundance": "warm golden harvest light, generous rich composition",
    "justice_and_morality": "stark contrasting light and shadow, balanced symbolic composition",
    "technology_and_progress": "sleek glowing illumination, modern aesthetics",
}

_STYLE_SUFFIX = (
    "masterpiece, award-winning visual art, sharp focus, crisp details, natural proportions, "
    "elegant cinematic lighting, 8k resolution"
)


_STYLE_KEYWORD_MAP: dict[str, str] = {
    "watercolor": "exquisite storybook watercolor painting, delicate translucent washes of color, soft paper texture, luminous lighting, masterpiece illustration, clean detailed linework",
    "oil-painting": "classical fine art oil painting, impasto brushwork, rich layered pigments, dramatic chiaroscuro lighting, canvas texture, museum fine art masterpiece",
    "ink-sketch": "intricate fine ink etching, delicate cross-hatching, fine fountain pen linework, subtle sepia wash shading, handcrafted masterpiece illustration",
    "photorealistic": "stunning 8k cinematic photography, 50mm f/1.4 lens, natural lifelike facial details, realistic skin texture, soft ambient bokeh, studio lighting, hyper-realistic masterpiece",
    "surreal": "surrealist art masterpiece, dreamlike ethereal composition, poetic symbolism, floating glowing elements, Salvador Dali and René Magritte aesthetic, razor-sharp details",
    "ukiyo-e": "authentic traditional Japanese ukiyo-e woodblock print, elegant black keylines, harmonious mineral pigments, delicate washi paper texture, classic Hokusai aesthetic",
}


def build_image_prompt(
    elements: ExtractedElements,
    style: str | None = None,
    language: str = "English",
    style_suffix: str = _STYLE_SUFFIX,
) -> str:
    """Build a concrete diffusion prompt with authentic cultural grounding from extracted poem elements and agent outputs."""
    visuals = ", ".join(elements.visual_elements) if elements.visual_elements else "an evocative poetic scene"
    mood_hint = _THEME_MOOD_HINTS.get(elements.theme, "atmospheric, emotionally resonant lighting")
    style_hint = _STYLE_KEYWORD_MAP.get((style or "").lower().strip(), "")

    # Cultural grounding anchor
    cultural_parts = []
    if elements.cultural_context:
        cultural_parts.append(f"Authentic {elements.cultural_context}.")
    elif language and language.lower() != "english":
        cultural_parts.append(
            f"Authentic {language} regional setting with traditional {language} architecture, clothing, and natural landscape."
        )

    style_cue_part = "; ".join(elements.style_cues[:2]) if elements.style_cues else ""
    palette_part = f"Colour palette of {elements.palette}." if elements.palette else ""
    emotion_part = f"Atmosphere: {elements.emotion_nuance}." if elements.emotion_nuance else f"Mood: {elements.dominant_emotion}."

    prompt_parts = [
        f"{style_hint}," if style_hint else "",
        f"A breathtaking, detailed visual scene depicting {visuals}.",
        " ".join(cultural_parts),
        emotion_part,
        mood_hint,
        palette_part,
        style_cue_part,
        style_suffix,
    ]
    return " ".join(p.strip() for p in prompt_parts if p.strip()).strip()


def build_negative_prompt() -> str:
    """
    Standard negative prompt to suppress common diffusion failure modes
    (distorted faces, extra fingers, text artifacts, watermarks).
    """
    return (
        "deformed face, blurry eyes, distorted anatomy, warped face, extra limbs, extra fingers, "
        "mutated hands, bad proportions, unnatural skin texture, poorly drawn face, poorly drawn hands, "
        "disfigured, text, watermark, signature, logo, low quality, jpeg artifacts, oversaturated, plastic skin"
    )
