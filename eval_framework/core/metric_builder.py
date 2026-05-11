"""
MetricBuilder — ánh xạ từ tên metric trong JSON sang DeepEval metric objects.

Metrics theo loại task:
  LLM          : answer_relevancy, hallucination, bias, toxicity, g_eval, summarization, exact_match
  RAG          : answer_relevancy, faithfulness, contextual_precision, contextual_recall,
                 contextual_relevancy, hallucination
  Agent        : tool_correctness, argument_correctness, task_completion, goal_accuracy
  Multi-turn   : conversation_completeness, knowledge_retention, role_adherence
  Chung        : g_eval, exact_match, json_correctness, prompt_alignment, topic_adherence

Thêm metric mới của project: dùng register_metric()
"""

from typing import Any, Dict, List, Optional
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import (
    # RAG
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    # Agent
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
    TaskCompletionMetric,
    GoalAccuracyMetric,
    # Multi-turn
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    # LLM / General
    GEval,
    BiasMetric,
    ToxicityMetric,
    SummarizationMetric,
    ExactMatchMetric,
    JsonCorrectnessMetric,
    PromptAlignmentMetric,
    TopicAdherenceMetric,
    BaseMetric,
)
from eval_framework.core.base_schema import MetricConfig

# Registry cho custom metrics của project — dùng register_metric() để mở rộng
_CUSTOM_METRIC_REGISTRY: dict = {}


def register_metric(name: str, factory_fn) -> None:
    """
    Đăng ký một custom metric factory.

    factory_fn(threshold, model, extra_params) → BaseMetric

    Ví dụ:
        register_metric("my_metric", lambda t, m, p: MyMetric(threshold=t, model=m))
    """
    _CUSTOM_METRIC_REGISTRY[name.lower()] = factory_fn


class MetricBuilder:
    """Xây dựng DeepEval metric objects từ MetricConfig."""

    @classmethod
    def build(
        cls,
        configs: List[MetricConfig],
        default_model: Any = None,  # str hoặc GPTModel object
    ) -> List[BaseMetric]:
        metrics = []
        for cfg in configs:
            try:
                metric = cls._build_one(cfg, default_model)
                if metric is not None:
                    metrics.append(metric)
            except Exception as e:
                # LLM-based metrics cần API key — bỏ qua khi chạy local mode
                print(f"  ⚠️  Bỏ qua metric '{cfg.name}': {e}")
        return metrics

    @classmethod
    def _build_one(
        cls,
        cfg: MetricConfig,
        default_model: Any = None,  # str hoặc GPTModel object
    ) -> Optional[BaseMetric]:
        name = cfg.name.lower()
        # cfg.model (string tên model) chỉ dùng khi không có judge_model object
        model = default_model if default_model is not None else cfg.model
        t = cfg.threshold

        # Shorthand builders
        def llm_kwargs(**extra) -> Dict[str, Any]:
            kw = {"threshold": t, "include_reason": True, **extra}
            if model:
                kw["model"] = model
            return kw

        # ── RAG ──────────────────────────────────────────────────────────────
        if name == "answer_relevancy":
            return AnswerRelevancyMetric(**llm_kwargs())

        if name == "faithfulness":
            return FaithfulnessMetric(**llm_kwargs())

        if name == "contextual_precision":
            return ContextualPrecisionMetric(**llm_kwargs())

        if name == "contextual_recall":
            return ContextualRecallMetric(**llm_kwargs())

        if name == "contextual_relevancy":
            return ContextualRelevancyMetric(**llm_kwargs())

        if name == "hallucination":
            return HallucinationMetric(**llm_kwargs())

        # ── Agent ─────────────────────────────────────────────────────────────
        if name == "tool_correctness":
            return ToolCorrectnessMetric(**llm_kwargs())

        if name == "argument_correctness":
            return ArgumentCorrectnessMetric(**llm_kwargs())

        if name == "task_completion":
            # criteria dùng để mô tả task nếu metric cần (optional trong DeepEval)
            kw = llm_kwargs()
            if cfg.criteria:
                kw["task"] = cfg.criteria
            return TaskCompletionMetric(**kw)

        if name == "goal_accuracy":
            return GoalAccuracyMetric(**llm_kwargs())

        # ── Multi-turn ────────────────────────────────────────────────────────
        if name == "conversation_completeness":
            return ConversationCompletenessMetric(**llm_kwargs())

        if name == "knowledge_retention":
            return KnowledgeRetentionMetric(**llm_kwargs())

        if name == "role_adherence":
            return RoleAdherenceMetric(**llm_kwargs())

        # ── LLM / General ─────────────────────────────────────────────────────
        if name == "g_eval":
            if not cfg.criteria:
                print("  ⚠️  g_eval cần 'criteria' — bỏ qua.")
                return None
            kw = {
                "name": cfg.name,
                "criteria": cfg.criteria,
                "evaluation_params": [
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                "threshold": t,
            }
            if model:
                kw["model"] = model
            return GEval(**kw)

        if name == "bias":
            return BiasMetric(**llm_kwargs())

        if name == "toxicity":
            return ToxicityMetric(**llm_kwargs())

        if name == "summarization":
            return SummarizationMetric(**llm_kwargs())

        if name == "exact_match":
            return ExactMatchMetric(threshold=t)

        if name == "json_correctness":
            # criteria truyền expected_schema dạng string mô tả
            if not cfg.criteria:
                print(
                    "  ⚠️  json_correctness cần 'criteria' (expected_schema) — bỏ qua."
                )
                return None
            kw = {
                "expected_schema": cfg.criteria,
                "threshold": t,
                "include_reason": True,
            }
            if model:
                kw["model"] = model
            return JsonCorrectnessMetric(**kw)

        if name == "prompt_alignment":
            if not cfg.criteria:
                print(
                    "  ⚠️  prompt_alignment cần 'criteria' (prompt_instructions) — bỏ qua."
                )
                return None
            kw = {
                "prompt_instructions": cfg.criteria.split("\n"),
                "threshold": t,
                "include_reason": True,
            }
            if model:
                kw["model"] = model
            return PromptAlignmentMetric(**kw)

        if name == "topic_adherence":
            if not cfg.criteria:
                print(
                    "  ⚠️  topic_adherence cần 'criteria' (danh sách topic cách nhau bởi dấu phẩy) — bỏ qua."
                )
                return None
            topics = [s.strip() for s in cfg.criteria.split(",")]
            kw = {"relevant_topics": topics, "threshold": t, "include_reason": True}
            if model:
                kw["model"] = model
            return TopicAdherenceMetric(**kw)

        # ── Custom metrics của project ─────────────────────────────────────────
        factory = _CUSTOM_METRIC_REGISTRY.get(name)
        if factory:
            return factory(t, model, cfg)

        print(
            f"  ⚠️  Metric '{cfg.name}' không được hỗ trợ — bỏ qua. Dùng register_metric() để thêm."
        )
        return None
