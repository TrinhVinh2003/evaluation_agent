from typing import Any
from pydantic import BaseModel, Field
from eval_framework.core.base_schema import BaseCaseSchema

class ExpectedToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class AgentCaseSchema(BaseCaseSchema):
    """Single-turn agent."""
    input: str
    expected_output: str | None = None
    expected_tools: list[ExpectedToolCall] = Field(default_factory=list)
