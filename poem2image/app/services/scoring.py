from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


def compute_alignment_score(candidate_text: str, reference_text: str) -> float:
    """Compute cosine-similarity alignment between candidate text and reference text (0.0 to 1.0)."""
    candidate_text = (candidate_text or "").strip()
    reference_text = (reference_text or "").strip()
    if not candidate_text or not reference_text:
        return 0.0

    try:
        from app.services.nlp_pipeline import _get_sentence_encoder

        encoder = _get_sentence_encoder()
        vecs = encoder.encode([candidate_text, reference_text], normalize_embeddings=True)
        cosine = float(np.dot(vecs[0], vecs[1]))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_alignment_score failed (%s); returning 0.0.", exc)
        return 0.0


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t.strip()]


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l(candidate: str, reference: str) -> dict[str, float]:
    """Compute ROUGE-L precision, recall, and F1 score."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def bleu_1(candidate: str, reference: str) -> float:
    """Compute unigram BLEU-1 with brevity penalty."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)
    if not cand_tokens or not ref_tokens:
        return 0.0

    cand_counts = Counter(cand_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum(min(count, ref_counts[tok]) for tok, count in cand_counts.items())
    precision = overlap / len(cand_tokens)

    brevity_penalty = 1.0 if len(cand_tokens) >= len(ref_tokens) else np.exp(1 - len(ref_tokens) / max(1, len(cand_tokens)))
    return float(precision * brevity_penalty)


@dataclass
class LabelScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationReport:
    per_label: list[LabelScore] = field(default_factory=list)
    accuracy: float = 0.0
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0


def classification_report(predictions: list[str], gold_labels: list[str]) -> ClassificationReport:
    """Generate precision/recall/F1 metrics across labels."""
    if len(predictions) != len(gold_labels):
        raise ValueError("predictions and gold_labels must be the same length")
    if not predictions:
        return ClassificationReport()

    labels = sorted(set(gold_labels) | set(predictions))
    per_label: list[LabelScore] = []

    for label in labels:
        tp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label)
        support = sum(1 for g in gold_labels if g == label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
        per_label.append(LabelScore(label=label, precision=precision, recall=recall, f1=f1, support=support))

    correct = sum(1 for p, g in zip(predictions, gold_labels) if p == g)
    accuracy = correct / len(predictions)

    labels_with_support = [ls for ls in per_label if ls.support > 0]
    macro_f1 = sum(ls.f1 for ls in labels_with_support) / len(labels_with_support) if labels_with_support else 0.0
    macro_precision = sum(ls.precision for ls in labels_with_support) / len(labels_with_support) if labels_with_support else 0.0
    macro_recall = sum(ls.recall for ls in labels_with_support) / len(labels_with_support) if labels_with_support else 0.0

    return ClassificationReport(
        per_label=per_label,
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
    )
