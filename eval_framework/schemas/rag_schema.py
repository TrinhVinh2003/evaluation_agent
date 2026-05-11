from eval_framework.core.base_schema import BaseCaseSchema

class RAGCaseSchema(BaseCaseSchema):
    input: str
    expected_output: str
    retrieval_context: list[str]
