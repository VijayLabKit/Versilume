from __future__ import annotations

from typing import List

from app.config import get_settings
from app.services.nlp_pipeline import _get_spacy_nlp, classify_emotion


def _line_entities(line: str) -> set[str]:
    nlp = _get_spacy_nlp()
    doc = nlp(line)
    ents = {ent.text.lower().strip() for ent in doc.ents}
    if not ents:
        ents = {tok.text.lower() for tok in doc if tok.pos_ in ("PROPN", "NOUN")}
    return ents


def segment_poem(poem: str) -> List[List[str]]:
    """Split poem into segments based on stanzas or entity/emotion transitions."""
    settings = get_settings()
    raw_text = poem.strip()
    if not raw_text:
        return []

    # Segment by stanza if double newlines are present
    raw_stanzas = [st.strip().splitlines() for st in raw_text.split("\n\n") if st.strip()]
    if len(raw_stanzas) > 1 and len(raw_stanzas) <= settings.max_segments_per_poem:
        return [[ln.strip() for ln in stanza if ln.strip()] for stanza in raw_stanzas]

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return []

    # Keep short poems as a single segment
    if len(lines) <= 8 or not (settings.segment_on_emotion_shift or settings.segment_on_entity_shift):
        return [lines]

    segments: list[list[str]] = []
    current: list[str] = [lines[0]]
    prev_entities = _line_entities(lines[0])
    prev_emotion, _ = classify_emotion(lines[0])

    min_lines = max(settings.min_lines_per_segment, 3)

    for line in lines[1:]:
        entities = _line_entities(line)
        emotion, _ = classify_emotion(line)

        entity_shift = settings.segment_on_entity_shift and (
            bool(prev_entities) and bool(entities) and prev_entities.isdisjoint(entities)
        )
        emotion_shift = settings.segment_on_emotion_shift and (emotion != prev_emotion)

        long_enough = len(current) >= min_lines
        room_for_more = len(segments) < settings.max_segments_per_poem - 1

        if long_enough and room_for_more and (entity_shift and emotion_shift):
            segments.append(current)
            current = [line]
        else:
            current.append(line)

        prev_entities = entities or prev_entities
        prev_emotion = emotion

    segments.append(current)
    return segments
