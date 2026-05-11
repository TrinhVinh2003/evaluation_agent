import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langfuse import get_client

from eval_framework.core.base_schema import BaseCaseSchema, EvalSuiteConfig
from eval_framework.factory.schema_factory import SchemaFactory
from eval_framework.schemas.agent_schema import AgentCaseSchema
from eval_framework.schemas.llm_schema import LLMCaseSchema
from eval_framework.schemas.multi_turn_schema import MultiTurnCaseSchema
from eval_framework.schemas.rag_schema import RAGCaseSchema

langfuse = get_client()


class LangfuseDatasetSync:
    """
    Service đồng bộ dữ liệu test case lên Langfuse Dataset theo chuẩn SOLID.
    Sử dụng Pydantic models để validate và handlers chuyên biệt cho từng type.
    """

    def __init__(self):
        self.langfuse = langfuse
        # Đăng ký handlers cho từng loại case
        self._handlers: dict[str, Callable[[str, Any, EvalSuiteConfig], None]] = {
            "llm": self._upload_llm_items,
            "rag": self._upload_rag_items,
            "agent": self._upload_agent_items,
            "multi_turn": self._upload_multi_turn_items,
        }

    def sync_from_json(self, suite_path: str):
        path = Path(suite_path)
        if not path.exists():
            print(f"❌ File không tồn tại: {suite_path}")
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # 1. Validate Suite Config
        try:
            suite_config = SchemaFactory.validate_suite(data)
        except Exception as e:
            print(f"❌ Lỗi cấu hình Suite khi sync: {e}")
            return

        suite_name = suite_config.eval_suite
        suite_type = suite_config.type

        print(f"🚀 Syncing {suite_type.upper()} dataset '{suite_name}' to Langfuse...")

        # 2. Khởi tạo dataset (idempotent)
        try:
            self.langfuse.create_dataset(
                name=suite_name,
                description=f"Type: {suite_type} | Source: {path.name}",
                metadata={
                    "type": suite_type,
                    "default_model": suite_config.default_model,
                    "source_file": suite_path,
                },
            )
        except Exception:
            # Dataset có thể đã tồn tại
            pass

        # 3. Dispatch tới handler tương ứng cho từng case
        handler = self._handlers.get(suite_type)
        if not handler:
            print(
                f"❌ Unknown type '{suite_type}'. Valid: {list(self._handlers.keys())}"
            )
            return

        uploaded_count = 0
        for case_raw in suite_config.test_cases:
            try:
                # Validate case (injects defaults từ suite)
                case = SchemaFactory.validate_case(case_raw, suite_config)
                handler(suite_name, case, suite_config)
                uploaded_count += 1
            except Exception as e:
                print(f"  ❌ Error uploading case {case_raw.get('id', 'unknown')}: {e}")

        print(
            f"✅ Done — {uploaded_count}/{len(suite_config.test_cases)} items uploaded to '{suite_name}'\n"
        )

    def _upload_llm_items(
        self, suite_name: str, case: LLMCaseSchema, suite_config: EvalSuiteConfig
    ):
        self.langfuse.create_dataset_item(
            id=case.id,
            dataset_name=suite_name,
            input=case.input,
            expected_output=case.expected_output,
            metadata=self._build_metadata(case, suite_config),
        )
        print(f"  ↑ LLM   [{case.id}] {case.name}")

    def _upload_rag_items(
        self, suite_name: str, case: RAGCaseSchema, suite_config: EvalSuiteConfig
    ):
        self.langfuse.create_dataset_item(
            id=case.id,
            dataset_name=suite_name,
            input={
                "id": case.id,
                "type": "rag",
                "input": case.input,
                "retrieval_context": case.retrieval_context,
            },
            expected_output=case.expected_output,
            metadata=self._build_metadata(case, suite_config),
        )
        print(f"  ↑ RAG   [{case.id}] {case.name}")

    def _upload_agent_items(
        self, suite_name: str, case: AgentCaseSchema, suite_config: EvalSuiteConfig
    ):
        self.langfuse.create_dataset_item(
            id=case.id,
            dataset_name=suite_name,
            input={
                "id": case.id,
                "type": "agent",
                "input": case.input,
            },
            expected_output={
                "expected_tools": [
                    t.model_dump() if hasattr(t, "model_dump") else t
                    for t in case.expected_tools
                ],
                "expected_output": case.expected_output,
            },
            metadata=self._build_metadata(case, suite_config),
        )
        print(f"  ↑ AGENT [{case.id}] {case.name}")

    def _upload_multi_turn_items(
        self, suite_name: str, case: MultiTurnCaseSchema, suite_config: EvalSuiteConfig
    ):
        self.langfuse.create_dataset_item(
            id=case.id,
            dataset_name=suite_name,
            input={
                "id": case.id,
                "type": "multi_turn",
                "turns": [{"content": t.content} for t in case.turns],
            },
            expected_output={
                "criteria": case.criteria,
                "turns": [
                    {
                        "expected_tools": [
                            tool.model_dump() if hasattr(tool, "model_dump") else tool
                            for tool in (t.expected_tools or [])
                        ],
                        "expected_output": t.expected_output,
                    }
                    for t in case.turns
                ],
            },
            metadata={
                **self._build_metadata(case, suite_config),
                "turn_count": len(case.turns),
            },
        )
        print(f"  ↑ MT    [{case.id}] {case.name} ({len(case.turns)} turns)")

    def _build_metadata(
        self, case: BaseCaseSchema, suite_config: EvalSuiteConfig
    ) -> dict[str, Any]:
        """Build metadata chung cho các loại case."""
        return {
            **case.metadata,
            "name": case.name,
            "metrics": [m.model_dump(exclude_none=True) for m in (case.metrics or [])],
        }


def create_dataset_from_json(suite_path: str):
    """Dispatcher function for external use."""
    sync_service = LangfuseDatasetSync()
    sync_service.sync_from_json(suite_path)
