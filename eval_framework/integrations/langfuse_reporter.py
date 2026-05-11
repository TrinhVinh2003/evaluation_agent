from typing import Dict, List

from deepeval.metrics import BaseMetric


class LangfuseReporter:
    @staticmethod
    def report_case(case_id: str, metrics: List[BaseMetric], metadata: Dict = None):
        """No-op khi chạy local (không có trace_id)."""
        pass

    @staticmethod
    def report_scores(
        trace_id: str, metric_results: List[BaseMetric], dataset_item_id: str = None
    ):
        """Gắn DeepEval scores vào Langfuse trace (Langfuse SDK v3/v4)."""
        if not trace_id or not metric_results:
            return

        from langfuse import get_client

        langfuse = get_client()

        for m in metric_results:
            score = getattr(m, "score", 0) or 0
            name = getattr(m, "metric_name", m.__class__.__name__)
            reason = getattr(m, "reason", "") or ""

            langfuse.create_score(
                trace_id=trace_id,
                name=name,
                value=float(score),
                comment=reason,
                data_type="NUMERIC",
                dataset_item_id=dataset_item_id,
            )
            print(f"  📊 [{trace_id[:8]}…] {name} = {score:.2f}")
