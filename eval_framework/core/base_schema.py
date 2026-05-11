from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field



class MetricConfig(BaseModel):
    """Cấu hình cho một Metric cụ thể."""

    name: str
    threshold: float = 0.7
    model: str | None = None      # LLM judge model, VD: "gpt-4o", "gpt-4.1"
    criteria: str | None = None   # Dành cho g_eval


class BaseCaseSchema(BaseModel):
    """Schema cơ sở cho mọi loại test case."""

    id: str
    name: str
    type: str = ""  # Sẽ được inject từ Suite
    metrics: list[MetricConfig] | None = Field(default=None)
    criteria: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        populate_by_name = True




class EvalSuiteConfig(BaseModel):
    """Cấu hình cho cả một bộ test (suite) theo chuẩn mới."""

    eval_suite: str
    type: str  # "llm", "rag", "agent", "multi_turn", hoặc custom type đã đăng ký
    default_model: str | None = None
    default_threshold: float = 0.7
    default_metrics: list[MetricConfig] = Field(default_factory=list)
    test_cases: list[dict]  # Để parse thô trước khi validate từng case theo type
