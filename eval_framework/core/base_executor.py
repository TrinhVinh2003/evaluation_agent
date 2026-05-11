from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutorResponse:
    """
    Standardized response from any Task Executor.
    This ensures the EvalCase doesn't need to know the internals of the project.
    """
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_context: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None
    raw_response: Any = None  # To keep anything else for specific metrics


class BaseTaskExecutor(ABC):
    """
    Abstract Base Class for all Task Executors.
    An Executor knows HOW to run a specific task (DMS, RAG, LLM, etc.)
    """

    @abstractmethod
    async def execute(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ExecutorResponse:
        """
        Execute the task and return a standardized ExecutorResponse.
        """
        pass
