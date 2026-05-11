from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from eval_framework.cases.llm_case import LLMEvalCase
from eval_framework.schemas.rag_schema import RAGCaseSchema


class RAGEvalCase(LLMEvalCase):
    config: RAGCaseSchema

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrieval_context: list[str] = []

    async def run(self):
        response = await self.executor.execute(
            input_text=self.config.input,
            context=self.config.metadata.get("context", {}),
        )
        self.actual_responses = [response.content]
        self.retrieval_context = response.retrieval_context or self.config.retrieval_context or []

    def build_test_case(self) -> LLMTestCase:
        case = super().build_test_case()
        case.retrieval_context = self.retrieval_context
        return case

    def get_default_metrics(self) -> list:
        return [
            AnswerRelevancyMetric(threshold=0.7, include_reason=True),
            FaithfulnessMetric(threshold=0.7, include_reason=True),
            ContextualPrecisionMetric(threshold=0.7, include_reason=True),
            ContextualRecallMetric(threshold=0.7, include_reason=True),
            ContextualRelevancyMetric(threshold=0.7, include_reason=True),
        ]
