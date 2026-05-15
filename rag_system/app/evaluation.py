"""Simple RAG evaluation metrics."""

from __future__ import annotations

from statistics import mean
from typing import Iterable, List, Set

from app.schemas import EvaluationResult, EvaluationSample


def _token_set(text: str) -> Set[str]:
    return {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()}


def _jaccard(a: str, b: str) -> float:
    left, right = _token_set(a), _token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def evaluate_samples(samples: Iterable[EvaluationSample]) -> EvaluationResult:
    """Compute lightweight evaluation metrics for local validation."""

    faithfulness_scores: List[float] = []
    answer_relevancy_scores: List[float] = []
    context_precision_scores: List[float] = []
    context_recall_scores: List[float] = []

    for sample in samples:
        combined_context = " ".join(sample.retrieved_contexts)
        reference_context = " ".join(sample.reference_contexts)
        faithfulness_scores.append(_jaccard(sample.answer, combined_context))
        answer_relevancy_scores.append(_jaccard(sample.answer, sample.ground_truth))
        context_precision_scores.append(_jaccard(combined_context, reference_context))
        context_recall_scores.append(_jaccard(reference_context, combined_context))

    return EvaluationResult(
        faithfulness=round(mean(faithfulness_scores or [0.0]), 4),
        answer_relevancy=round(mean(answer_relevancy_scores or [0.0]), 4),
        context_precision=round(mean(context_precision_scores or [0.0]), 4),
        context_recall=round(mean(context_recall_scores or [0.0]), 4),
    )
