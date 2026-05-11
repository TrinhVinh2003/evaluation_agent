from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric

from eval_framework.core.base_case import BaseEvalCase
from eval_framework.schemas.agent_schema import AgentCaseSchema


class AgentEvalCase(BaseEvalCase):
    config: AgentCaseSchema

    async def run(self):
        response = await self.executor.execute(
            input_text=self.config.input,
            context=self.config.metadata.get("context", {}),
            session_id=self.session_id,
        )
        self.actual_responses = [response.content]
        self.actual_tool_calls = [response.tool_calls]
        self.trace_id = response.trace_id

    def build_test_case(self) -> LLMTestCase:
        tools_called = [
            ToolCall(
                name=tc["name"],
                input_parameters=tc.get("arguments", {}),
            )
            for tc in (self.actual_tool_calls[0] if self.actual_tool_calls else [])
        ]

        expected_tools = [
            ToolCall(
                name=t.name,
                input_parameters=t.arguments,
            )
            for t in self.config.expected_tools
        ]

        return LLMTestCase(
            input=self.config.input,
            actual_output=self.actual_responses[0] if self.actual_responses else "",
            expected_output=self.config.expected_output,
            tools_called=tools_called,
            expected_tools=expected_tools,
        )

    def get_default_metrics(self) -> list:
        return [
            ToolCorrectnessMetric(threshold=0.8, include_reason=True),
            ArgumentCorrectnessMetric(threshold=0.8, include_reason=True),
        ]
