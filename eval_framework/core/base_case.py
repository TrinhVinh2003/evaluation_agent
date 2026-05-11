from abc import ABC, abstractmethod
from typing import Any, Optional

from deepeval.metrics import BaseMetric

from eval_framework.core.base_schema import BaseCaseSchema
from eval_framework.core.metric_builder import MetricBuilder
from eval_framework.integrations.langfuse_reporter import LangfuseReporter
from eval_framework.core.base_executor import BaseTaskExecutor


class BaseEvalCase(ABC):
    def __init__(self, config: BaseCaseSchema, executor: BaseTaskExecutor):
        self.config = config
        self.executor = executor
        self.session_id: str | None = None
        self.trace_id: str | None = None
        self.actual_responses: list[str] = []
        self.actual_tool_calls: list[list[dict]] = []

    @abstractmethod
    async def run(self):
        """Thực thi logic gọi mô hình/agent."""
        pass

    @abstractmethod
    def build_test_case(self) -> Any:
        """Đóng gói dữ liệu thành DeepEval test case."""
        pass

    @abstractmethod
    def get_default_metrics(self) -> list[BaseMetric]:
        """Trả về metrics mặc định nếu config không khai báo."""
        pass

    def get_metrics(self, judge_model: Any = None) -> list[BaseMetric]:
        """
        Lấy danh sách metrics từ config.
        judge_model: GPTModel object từ EvalPipeline (ưu tiên hơn _judge_model trong metadata).
        """
        if self.config.metrics:
            # judge_model object ưu tiên nhất; fallback về tên model trong metadata
            model = judge_model or self.config.metadata.get("_judge_model")
            return MetricBuilder.build(self.config.metrics, default_model=model)
        return self.get_default_metrics()

    async def execute(
        self,
        override_trace_id: Optional[str] = None,
        dataset_item_id: Optional[str] = None,
        judge_model: Any = None,
    ):
        """
        Template Method định nghĩa luồng chạy chuẩn.

        override_trace_id: trace ID của Langfuse Experiment run.
        judge_model:       GPTModel object dùng để chấm điểm DeepEval metrics.
        """
        from uuid import uuid4

        self.session_id = f"eval-{self.config.id}-{uuid4().hex[:8]}"
        print(f"\n▶ [{self.config.id}] {self.config.name} (Session: {self.session_id})")

        await self.run()
        test_case = self.build_test_case()
        metrics = self.get_metrics(judge_model=judge_model)

        if not metrics:
            print("  ⚠️  Không có metric nào khả dụng (thiếu judge model/API key?).")
        else:
            for metric in metrics:
                score = metric.measure(test_case)
                if not hasattr(metric, "score") or metric.score is None:
                    metric.score = score

                status = (
                    "✅" if metric.score >= getattr(metric, "threshold", 0.5) else "❌"
                )
                print(
                    f"  {status} {getattr(metric, 'metric_name', metric.__class__.__name__)}: {metric.score:.2f}"
                )

        report_trace_id = override_trace_id or self.trace_id
        if report_trace_id:
            LangfuseReporter.report_scores(
                report_trace_id, metrics, dataset_item_id=dataset_item_id
            )

        else:
            LangfuseReporter.report_case(self.config.id, metrics, self.config.metadata)

        return {
            "metrics": metrics,
            "actual_responses": self.actual_responses,
            "trace_id": report_trace_id,
        }
