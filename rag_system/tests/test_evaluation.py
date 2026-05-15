from app.evaluation import evaluate_samples
from app.schemas import EvaluationSample


def test_evaluation_scores_are_bounded():
    sample = EvaluationSample(
        question="What is RAG?",
        answer="RAG combines retrieval and generation.",
        ground_truth="RAG combines retrieval and generation.",
        retrieved_contexts=["RAG combines retrieval and generation for grounded answering."],
        reference_contexts=["RAG combines retrieval and generation for grounded answering."],
    )

    result = evaluate_samples([sample])

    assert 0.0 <= result.faithfulness <= 1.0
    assert 0.0 <= result.answer_relevancy <= 1.0
    assert 0.0 <= result.context_precision <= 1.0
    assert 0.0 <= result.context_recall <= 1.0
