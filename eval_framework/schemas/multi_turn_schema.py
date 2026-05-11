from pydantic import BaseModel, Field
from eval_framework.core.base_schema import BaseCaseSchema
from eval_framework.schemas.agent_schema import ExpectedToolCall

class EvalTurn(BaseModel):
    """Một lượt hội thoại."""
    role: str | None = None  # "user" hoặc "assistant"
    content: str
    expected_tools: list[ExpectedToolCall] = Field(default_factory=list)
    expected_output: str | None = None
    retrieval_context: list[str] | None = None

class MultiTurnCaseSchema(BaseCaseSchema):
    """Hội thoại đa lượt."""
    turns: list[EvalTurn]
