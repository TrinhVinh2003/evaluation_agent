"""
JudgeModel — cấu hình LLM dùng để chấm điểm DeepEval metrics.

Thứ tự ưu tiên:
  1. Tham số truyền thẳng vào EvalPipeline(judge_model=...)
  2. Biến môi trường VLM_* trong .env
  3. Biến môi trường OPENAI_* chuẩn (fallback)

Team mới muốn dùng model riêng:
  from deepeval.models import GPTModel
  my_model = GPTModel(model="gpt-4o", api_key="sk-...")
  pipeline = EvalPipeline(..., judge_model=my_model)
"""

import os
from typing import Any, Optional


def get_judge_model(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """
    Tạo DeepEval GPTModel từ tham số hoặc env vars.
    Trả về None nếu không có cấu hình nào — DeepEval sẽ báo lỗi khi measure().

    Tham số truyền trực tiếp luôn ưu tiên hơn env vars.
    """
    from deepeval.models import GPTModel

    resolved_model   = model   or os.getenv("VLM_MODEL")   or os.getenv("OPENAI_MODEL")
    resolved_api_key = api_key or os.getenv("VLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    resolved_base_url = base_url or os.getenv("VLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")

    if not resolved_model:
        return None

    kwargs: dict = {"model": resolved_model}
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url

    return GPTModel(**kwargs)
