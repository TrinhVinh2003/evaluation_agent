from typing import Any, Dict, List
from eval_framework.core.base_schema import EvalSuiteConfig, BaseCaseSchema
from eval_framework.core.schema_registry import CASE_TYPE_MAP

class SchemaFactory:
    """
    Factory để parse và validate dữ liệu JSON thành các Pydantic Schema.
    Hỗ trợ cấu trúc Suite mới với type và metrics ở root.
    """
    
    @staticmethod
    def validate_suite(suite_data: Dict[str, Any]) -> EvalSuiteConfig:
        """Validate toàn bộ bộ test suite."""
        return EvalSuiteConfig.model_validate(suite_data)

    @staticmethod
    def validate_case(case_data: Dict[str, Any], suite_config: EvalSuiteConfig) -> BaseCaseSchema:
        """Validate một test case, inject defaults từ suite config."""
        case_data["type"] = suite_config.type

        if not case_data.get("metrics"):
            case_data["metrics"] = [m.model_dump() for m in suite_config.default_metrics]

        # Inject judge model vào metadata để MetricBuilder dùng khi build metrics
        metadata = case_data.setdefault("metadata", {})
        if suite_config.default_model and "_judge_model" not in metadata:
            metadata["_judge_model"] = suite_config.default_model

        schema_class = CASE_TYPE_MAP.get(suite_config.type)
        if not schema_class:
            raise ValueError(f"Không tìm thấy Schema cho loại: {suite_config.type}")

        return schema_class.model_validate(case_data)
