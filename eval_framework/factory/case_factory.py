from eval_framework.core.base_executor import BaseTaskExecutor
from eval_framework.core.base_case import BaseEvalCase
from eval_framework.core.base_schema import BaseCaseSchema
from eval_framework.core.schema_registry import _get_impl_map
from eval_framework.factory.schema_factory import SchemaFactory


class EvalCaseFactory:
    @classmethod
    def get_impl(cls, case_type: str) -> type[BaseEvalCase]:
        impl = _get_impl_map().get(case_type)
        if not impl:
            raise ValueError(
                f"Không tìm thấy implementation cho type '{case_type}'. "
                f"Dùng register_type() để thêm loại mới."
            )
        return impl

    @classmethod
    def create_from_config(cls, config: BaseCaseSchema, executor: BaseTaskExecutor) -> BaseEvalCase:
        impl_class = cls.get_impl(config.type)
        return impl_class(config=config, executor=executor)

    @classmethod
    def create(cls, case_data: dict, executor: BaseTaskExecutor) -> BaseEvalCase:
        """Tạo case từ raw dict (không có suite context)."""
        config = SchemaFactory.validate_case(case_data, None)
        return cls.create_from_config(config, executor)
