"""
EvalPipeline — điểm vào duy nhất của framework.

Cách dùng cơ bản:
    from eval_framework import EvalPipeline
    from eval_framework.core.base_executor import BaseTaskExecutor, ExecutorResponse

    class MyExecutor(BaseTaskExecutor):
        async def execute(self, input_text, context=None, session_id=None, **kwargs):
            result = await my_agent.run(input_text)
            return ExecutorResponse(content=result.text, tool_calls=result.tools)

    pipeline = EvalPipeline("my_suite.json", MyExecutor())
    asyncio.run(pipeline.run())
"""

import json
from datetime import datetime
from typing import Any, Optional

from eval_framework.core.base_executor import BaseTaskExecutor
from eval_framework.core.base_schema import EvalSuiteConfig
from eval_framework.core.judge_model import get_judge_model
from eval_framework.factory.case_factory import EvalCaseFactory
from eval_framework.factory.schema_factory import SchemaFactory
from eval_framework.integrations.langfuse_dataset import LangfuseDatasetSync
from eval_framework.runner import EvalSummary


from langfuse import Evaluation
import nest_asyncio


class _MutableDatasetItem:
    """Wrapper cho DatasetItem để vượt qua lỗi frozen_instance và hỗ trợ truy cập dạng object."""

    def __init__(self, item):
        self.id = getattr(item, "id", None)
        self.input = getattr(item, "input", None)
        self.expected_output = getattr(item, "expected_output", None)
        self.dataset_id = getattr(item, "dataset_id", None)
        metadata = getattr(item, "metadata", {}) or {}
        # Lọc bớt metadata để tránh cảnh báo 200 ký tự
        if isinstance(metadata, dict) and "metrics" in metadata:
            metadata = {k: v for k, v in metadata.items() if k != "metrics"}
        self.metadata = metadata


class EvalPipeline:
    """
    Chạy toàn bộ quy trình eval từ 1 file JSON:
      Phase 1 — Sync: upload test cases lên Langfuse Dataset
      Phase 2 — Experiment: pull items, chạy executor, chấm điểm DeepEval, push scores

    Args:
        suite_path:   đường dẫn tới file JSON suite
        executor:     implementation của BaseTaskExecutor cho hệ thống AI cần eval
        run_name:     tên experiment run trên Langfuse (tự sinh nếu không truyền)
        mode:         "local"           — chạy DeepEval cục bộ, không cần Langfuse
                      "full"            — sync lên Langfuse + chạy experiment + chấm điểm
                      "sync_only"       — chỉ upload dataset lên Langfuse
                      "experiment_only" — chỉ chạy experiment (dataset đã có trên Langfuse)
        judge_model:  DeepEval model dùng để chấm điểm (GPTModel object).
                      Mặc định: đọc từ VLM_* trong .env.
                      Override: truyền GPTModel object của bạn.
    """

    def __init__(
        self,
        suite_path: str,
        executor: BaseTaskExecutor,
        run_name: Optional[str] = None,
        mode: str = "full",
        judge_model: Any = None,
    ):
        self.suite_path = suite_path
        self.executor = executor
        self.run_name = run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.mode = mode
        # judge_model: tham số trực tiếp > VLM_* env > None
        self.judge_model = judge_model if judge_model is not None else get_judge_model()
        self._langfuse = None  # lazy — chỉ khởi tạo khi cần Langfuse

    @property
    def langfuse(self):
        if self._langfuse is None:
            from langfuse import get_client

            self._langfuse = get_client()
        return self._langfuse

    async def run(self) -> EvalSummary:
        suite_config = self._load_suite()

        if self.mode == "local":
            return await self._run_local(suite_config)

        if self.mode in ("full", "sync_only"):
            self._sync_dataset()

        summary = EvalSummary()
        if self.mode in ("full", "experiment_only"):
            summary = await self._run_experiment(suite_config)

        return summary

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _run_local(self, suite_config: EvalSuiteConfig) -> EvalSummary:
        """Chạy DeepEval cục bộ từ JSON — không cần Langfuse."""
        print(f"\n{'=' * 55}")
        print(
            f"🏃 Local Mode: '{suite_config.eval_suite}' ({len(suite_config.test_cases)} cases)"
        )
        if self.judge_model:
            name = getattr(self.judge_model, "name", None) or getattr(
                self.judge_model, "model_name", str(self.judge_model)
            )
            print(f"⚖️  Judge model: {name}")

        summary = EvalSummary()

        for case_raw in suite_config.test_cases:
            try:
                case_config = SchemaFactory.validate_case(dict(case_raw), suite_config)
                case = EvalCaseFactory.create_from_config(case_config, self.executor)
                result = await case.execute(judge_model=self.judge_model)
                summary.add_result(result)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"❌ Error in case: {e}")
                summary.errors += 1

        summary.print_report()
        return summary

    def _load_suite(self) -> EvalSuiteConfig:
        with open(self.suite_path, encoding="utf-8") as f:
            data = json.load(f)
        return SchemaFactory.validate_suite(data)

    def _sync_dataset(self):
        print(f"\n{'=' * 55}")
        print("📤 Phase 1: Syncing dataset to Langfuse...")
        LangfuseDatasetSync().sync_from_json(self.suite_path)

    async def _run_experiment(self, suite_config: EvalSuiteConfig) -> EvalSummary:
        print(f"\n{'=' * 55}")
        print(
            f"🧪 Phase 2: Experiment '{self.run_name}' on '{suite_config.eval_suite}'"
        )

        try:
            nest_asyncio.apply()
            dataset = self.langfuse.get_dataset(suite_config.eval_suite)
            # Dùng wrapper để tránh frozen_instance và lọc metadata
            items = [_MutableDatasetItem(item) for item in dataset.items]
        except Exception as e:
            print(f"❌ Không tìm thấy dataset '{suite_config.eval_suite}': {e}")
            return EvalSummary()

        judge_model = self.judge_model
        executor = self.executor
        reconstruct_fn = self._reconstruct_case

        async def task(*, item, **kwargs):
            try:
                case_config = reconstruct_fn(item, suite_config)
                case = EvalCaseFactory.create_from_config(case_config, executor)
                print(f"\n▶ [{item.id[:8]}…] {case_config.name}")
                await case.run()
                return {
                    "test_case": case.build_test_case(),
                    "actual_responses": case.actual_responses,
                }
            except Exception as e:
                import traceback

                traceback.print_exc()
                item_id = getattr(item, "id", str(item))
                print(f"❌ Task thất bại item {item_id}: {e}")
                return None

        # Một evaluator riêng cho mỗi metric trong suite defaults
        from eval_framework.core.metric_builder import MetricBuilder

        metrics_configs = suite_config.default_metrics or []

        def make_evaluator(metric_cfg):
            metric_name = getattr(metric_cfg, "name", str(metric_cfg))

            def evaluator(
                *, input, output, expected_output=None, metadata=None, **kwargs
            ):
                if not isinstance(output, dict):
                    return None
                test_case = output.get("test_case")
                if test_case is None:
                    return None
                try:
                    metric = MetricBuilder._build_one(metric_cfg, judge_model)
                    if metric is None:
                        return None
                    score_val = metric.measure(test_case)
                    value = float(getattr(metric, "score", score_val) or 0)
                    reason = getattr(metric, "reason", "") or ""
                    status = (
                        "✅" if value >= getattr(metric, "threshold", 0.5) else "❌"
                    )
                    print(f"  {status} {metric_name}: {value:.2f}")
                    return Evaluation(name=metric_name, value=value, comment=reason)

                except Exception as e:
                    print(f"  ⚠️  Metric '{metric_name}': {e}")
                    return None

            return evaluator

        evaluators = [make_evaluator(m) for m in metrics_configs]

        try:
            # nest_asyncio cho phép chạy run_experiment trực tiếp trong main loop
            exp_result = self.langfuse.run_experiment(
                name=self.run_name,
                data=items,  # Dùng wrapper đã lọc metadata
                task=task,
                evaluators=evaluators,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"❌ Experiment thất bại: {e}")
            return EvalSummary()

        # Chuyển ExperimentResult → EvalSummary
        from types import SimpleNamespace

        summary = EvalSummary()
        for ir in exp_result.item_results:
            fake_metrics = []
            for ev in ir.evaluations or []:
                # ir.evaluations là list các Evaluation object
                ev_name = getattr(ev, "name", "")
                ev_value = getattr(ev, "value", 0)
                m = SimpleNamespace(
                    metric_name=ev_name, score=ev_value or 0, threshold=0.5
                )
                fake_metrics.append(m)

            summary.add_result({
                "metrics": fake_metrics,
                "actual_responses": (ir.output or {}).get("actual_responses", []),
                "trace_id": ir.trace_id,
            })

        if getattr(exp_result, "dataset_run_url", None):
            print(f"\n🔗 Langfuse: {exp_result.dataset_run_url}")

        summary.print_report()
        return summary

    def _reconstruct_case(self, item, suite_config: EvalSuiteConfig):
        """Tái tạo CaseSchema từ một Langfuse Dataset Item (Robust version)."""
        metadata = getattr(item, "metadata", {}) or {}

        inp = item.input
        exp = item.expected_output
        suite_type = suite_config.type

        # Helpers trích xuất dữ liệu an toàn
        def get_input_text(v):
            if isinstance(v, str):
                return v
            if isinstance(v, dict):
                return v.get("input", str(v))
            return str(v)

        def get_id(v, fallback):
            if isinstance(v, dict):
                return v.get("id", fallback)
            return fallback

        common = {
            "metrics": metadata.get("metrics")
            or [m.model_dump() for m in suite_config.default_metrics],
        }

        if suite_type == "llm":
            case_data = {
                "id": get_id(inp, getattr(item, "id", None)),
                "name": metadata.get("name", getattr(item, "id", None)),
                "input": get_input_text(inp),
                "expected_output": exp,
                **common,
            }

        elif suite_type == "rag":
            case_data = {
                "id": get_id(inp, getattr(item, "id", None)),
                "name": metadata.get("name", getattr(item, "id", None)),
                "input": get_input_text(inp),
                "retrieval_context": (
                    inp.get("retrieval_context", []) if isinstance(inp, dict) else []
                ),
                "expected_output": exp,
                **common,
            }

        elif suite_type == "agent":
            exp_dict = exp if isinstance(exp, dict) else {}
            case_data = {
                "id": get_id(inp, getattr(item, "id", None)),
                "name": metadata.get("name", getattr(item, "id", None)),
                "input": get_input_text(inp),
                "expected_output": exp_dict.get("expected_output", ""),
                "expected_tools": exp_dict.get("expected_tools", []),
                **common,
            }

        elif suite_type == "multi_turn":
            exp_dict = exp if isinstance(exp, dict) else {}
            input_turns = inp.get("turns", []) if isinstance(inp, dict) else []
            exp_turns = exp_dict.get("turns", [])
            turns = [
                {
                    "content": it.get("content", str(it))
                    if isinstance(it, dict)
                    else str(it),
                    "expected_tools": et.get("expected_tools", [])
                    if isinstance(et, dict)
                    else [],
                    "expected_output": et.get("expected_output", "")
                    if isinstance(et, dict)
                    else "",
                }
                for it, et in zip(input_turns, exp_turns)
            ]
            case_data = {
                "id": get_id(inp, getattr(item, "id", None)),
                "name": metadata.get("name", getattr(item, "id", None)),
                "turns": turns,
                "criteria": exp_dict.get("criteria", []),
                **common,
            }

        else:
            # Custom type
            case_data = {
                "id": getattr(item, "id", None),
                "name": metadata.get("name", getattr(item, "id", None)),
                "input": inp,
                "expected_output": exp,
                **metadata,
                **common,
            }

        return SchemaFactory.validate_case(case_data, suite_config)
