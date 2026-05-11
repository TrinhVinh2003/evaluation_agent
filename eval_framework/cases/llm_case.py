from deepeval.test_case import LLMTestCase
from eval_framework.core.base_case import BaseEvalCase
from eval_framework.schemas.llm_schema import LLMCaseSchema


class LLMEvalCase(BaseEvalCase):
    config: LLMCaseSchema

    async def run(self):
        response = await self.executor.execute(
            input_text=self.config.input,
            context=self.config.metadata.get("context", {}),
        )
        self.actual_responses = [response.content]

    def build_test_case(self) -> LLMTestCase:
        return LLMTestCase(
            input=self.config.input,
            actual_output=self.actual_responses[0],
            expected_output=self.config.expected_output,
        )

    def get_default_metrics(self) -> list:
        from deepeval.metrics import AnswerRelevancyMetric

        return [AnswerRelevancyMetric(threshold=0.7)]
