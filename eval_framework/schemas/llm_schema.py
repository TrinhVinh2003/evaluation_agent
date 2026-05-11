from eval_framework.core.base_schema import BaseCaseSchema


class LLMCaseSchema(BaseCaseSchema):
    input: str
    expected_output: str | None = None
