from deepeval.test_case import ConversationalTestCase, LLMTestCase, Turn, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric, ConversationCompletenessMetric

from eval_framework.core.base_case import BaseEvalCase
from eval_framework.schemas.multi_turn_schema import MultiTurnCaseSchema


class MultiTurnEvalCase(BaseEvalCase):
    config: MultiTurnCaseSchema

    async def run(self):
        context = self.config.metadata.get("context", {})
        for turn in self.config.turns:
            response = await self.executor.execute(
                input_text=turn.content, context=context, session_id=self.session_id
            )
            self.actual_responses.append(response.content)
            self.actual_tool_calls.append(response.tool_calls)
            self.trace_id = response.trace_id

    def build_test_case(self) -> LLMTestCase:
        """
        Multi-turn tool evaluation: gom tất cả tool calls từ mọi lượt
        vào một LLMTestCase để ToolCorrectnessMetric đánh giá toàn bộ.
        """
        all_tools_called = [
            ToolCall(name=tc["name"], input_parameters=tc.get("arguments", {}))
            for turn_calls in self.actual_tool_calls
            for tc in turn_calls
        ]

        all_expected_tools = [
            ToolCall(name=et.name, input_parameters=et.arguments)
            for turn in self.config.turns
            for et in turn.expected_tools
        ]

        final_output = self.actual_responses[-1] if self.actual_responses else ""

        return LLMTestCase(
            input=self.config.turns[0].content if self.config.turns else "",
            actual_output=final_output,
            tools_called=all_tools_called,
            expected_tools=all_expected_tools,
        )

    def build_conversational_test_case(self) -> ConversationalTestCase:
        """
        Tạo ConversationalTestCase cho các metric đánh giá hội thoại
        (VD: ConversationRelevancyMetric, ConversationCompletenessMetric).
        """
        turns = []
        for i, turn_config in enumerate(self.config.turns):
            turns.append(Turn(role="user", content=turn_config.content))
            actual = self.actual_responses[i] if i < len(self.actual_responses) else ""
            turns.append(
                Turn(
                    role="assistant",
                    content=actual,
                    expected_output=turn_config.expected_output,
                    retrieval_context=turn_config.retrieval_context,
                )
            )
        return ConversationalTestCase(turns=turns)

    def get_default_metrics(self) -> list:
        return [
            ToolCorrectnessMetric(threshold=0.8, include_reason=True),
            ArgumentCorrectnessMetric(threshold=0.8, include_reason=True),
            ConversationCompletenessMetric(threshold=0.8, include_reason=True),
        ]
