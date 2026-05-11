import json
from pathlib import Path
from typing import Any
from eval_framework.core.base_executor import BaseTaskExecutor
from eval_framework.factory.case_factory import EvalCaseFactory
from eval_framework.factory.schema_factory import SchemaFactory


class EvalSummary:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.metric_scores = {}

    def add_result(self, result_dict: dict):
        self.total += 1
        metrics = result_dict.get("metrics", [])

        case_passed = True
        for m in metrics:
            name = getattr(m, "metric_name", m.__class__.__name__)
            score = getattr(m, "score", 0)
            threshold = getattr(m, "threshold", 0.5)

            if score < threshold:
                case_passed = False

            if name not in self.metric_scores:
                self.metric_scores[name] = []
            self.metric_scores[name].append(score)

        if case_passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_report(self):
        print("\n" + "=" * 50)
        print("📊 FINAL EVALUATION REPORT")
        print("=" * 50)
        print(f"✅ Total Cases:  {self.total}")
        print(f"🟢 Passed:       {self.passed}")
        print(f"🔴 Failed:       {self.failed}")
        print(f"⚠️ Errors:       {self.errors}")
        print("-" * 50)
        print("📈 Average Metric Scores:")
        for name, scores in self.metric_scores.items():
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  - {name:<25}: {avg:.2f}")
        print("=" * 50 + "\n")


class EvalRunner:
    def __init__(self, executor: BaseTaskExecutor):
        self.executor = executor
        self.summary = EvalSummary()

    async def run_suite(self, suite_path: str):
        print(f"📂 Running Evaluation Suite: {suite_path}")

        path = Path(suite_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # 1. Validate Suite Config
        try:
            suite_config = SchemaFactory.validate_suite(data)
        except Exception as e:
            print(f"❌ Lỗi cấu hình Suite: {e}")
            return

        # 2. Chạy từng case
        for case_raw_data in suite_config.test_cases:
            try:
                # Validate case với defaults từ suite
                case_config = SchemaFactory.validate_case(case_raw_data, suite_config)

                # Tạo và thực thi Case thông qua Factory
                case = EvalCaseFactory.create_from_config(case_config, self.executor)
                result = await case.execute()
                self.summary.add_result(result)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"❌ Error in case: {str(e)}")
                self.summary.errors += 1

        self.summary.print_report()

    async def run_case_from_item(self, item: Any, executor: BaseTaskExecutor) -> dict:
        """
        Chạy một test case từ một item của Langfuse Dataset (dành cho Experiment).
        """
        # Reconstruct case_data từ item metadata
        raw_data = {}
        if isinstance(item.metadata, dict):
            raw_data = item.metadata.copy()

        if item.expected_output and "expected_output" not in raw_data:
            raw_data["expected_output"] = item.expected_output

        raw_data["id"] = raw_data.get("id") or item.id

        # Vì Langfuse Item thường đã lưu full config, ta có thể validate trực tiếp
        # Tuy nhiên để an toàn, ta giả định nó tuân theo BaseCaseSchema
        from eval_framework.factory.case_factory import EvalCaseFactory

        # Mock suite_config nếu cần hoặc lấy từ metadata
        # Ở đây ta giả định item.metadata đã có đủ type
        case_type = raw_data.get("type", "agent")
        from eval_framework.core.schema_registry import CASE_TYPE_MAP

        schema_class = CASE_TYPE_MAP.get(case_type)
        case_config = schema_class.model_validate(raw_data)

        case = EvalCaseFactory.create_from_config(case_config, executor)
        result = await case.execute()

        actual_output = (
            result["actual_responses"][-1] if result["actual_responses"] else ""
        )
        scores = {
            getattr(m, "metric_name", m.__class__.__name__): getattr(m, "score", 0)
            for m in result["metrics"]
        }

        return {
            "actual_output": actual_output,
            "scores": scores,
            "trace_id": result["trace_id"],
        }
