from eval_framework.schemas.agent_schema import AgentCaseSchema
from eval_framework.schemas.llm_schema import LLMCaseSchema
from eval_framework.schemas.multi_turn_schema import MultiTurnCaseSchema
from eval_framework.schemas.rag_schema import RAGCaseSchema

# Schema registry: type -> Pydantic schema class
CASE_TYPE_MAP: dict = {
    "llm": LLMCaseSchema,
    "rag": RAGCaseSchema,
    "agent": AgentCaseSchema,
    "multi_turn": MultiTurnCaseSchema,
}

# Impl registry: type -> EvalCase implementation class
CASE_IMPL_MAP: dict = {}

# Populated lazily to avoid circular imports
def _get_impl_map() -> dict:
    if not CASE_IMPL_MAP:
        from eval_framework.cases.agent_case import AgentEvalCase
        from eval_framework.cases.llm_case import LLMEvalCase
        from eval_framework.cases.multi_turn_case import MultiTurnEvalCase
        from eval_framework.cases.rag_case import RAGEvalCase
        CASE_IMPL_MAP.update({
            "llm": LLMEvalCase,
            "rag": RAGEvalCase,
            "agent": AgentEvalCase,
            "multi_turn": MultiTurnEvalCase,
        })
    return CASE_IMPL_MAP


def register_type(type_name: str, schema_class, impl_class) -> None:
    """
    Đăng ký một loại eval case mới vào framework.

    Ví dụ:
        from eval_framework.core.schema_registry import register_type
        register_type("chatbot", ChatbotCaseSchema, ChatbotEvalCase)
    """
    CASE_TYPE_MAP[type_name] = schema_class
    _get_impl_map()[type_name] = impl_class
