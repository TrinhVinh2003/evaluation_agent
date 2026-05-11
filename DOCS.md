# Hướng dẫn sử dụng Eval Framework

Framework đánh giá AI dành cho các bài toán LLM, RAG, Agent và hội thoại đa lượt.  
Tích hợp **DeepEval** để chấm điểm và **Langfuse** để theo dõi thí nghiệm.

---

## Mục lục

1. [Cài đặt](#1-cài-đặt)
2. [Cấu hình môi trường](#2-cấu-hình-môi-trường)
3. [Khái niệm cốt lõi](#3-khái-niệm-cốt-lõi)
4. [Bước 1 — Viết file Suite JSON](#4-bước-1--viết-file-suite-json)
5. [Bước 2 — Implement Executor](#5-bước-2--implement-executor)
6. [Bước 3 — Chạy evaluation](#6-bước-3--chạy-evaluation)
7. [Danh sách Metrics](#7-danh-sách-metrics)
8. [Xem kết quả](#8-xem-kết-quả)
9. [Mở rộng framework](#9-mở-rộng-framework)

---

## 1. Cài đặt

```bash
pip install deepeval langfuse langchain-openai langgraph python-dotenv nest-asyncio
```

---

## 2. Cấu hình môi trường

Copy file `.env.example` thành `.env` và điền các giá trị:

```env
# Judge model — LLM dùng để chấm điểm DeepEval
VLM_MODEL=gpt-4o              # hoặc gemini-1.5-flash, claude-3-5-sonnet, v.v.
VLM_API_KEY=sk-...
VLM_BASE_URL=                 # để trống nếu dùng OpenAI chuẩn

# Langfuse — chỉ cần khi chạy mode "full" hoặc "sync_only"
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

> **Chạy local không cần Langfuse hay judge model** nếu dùng `--executor mock`.  
> Judge model chỉ cần thiết khi chạy LLM-based metrics (answer_relevancy, faithfulness, v.v.)

---

## 3. Khái niệm cốt lõi

```
Suite JSON  →  EvalPipeline  →  Executor  →  DeepEval Metrics  →  Kết quả / Langfuse
```

| Thành phần       | Vai trò                                                            |
| ---------------- | ------------------------------------------------------------------ |
| **Suite JSON**   | Khai báo bộ test: loại bài toán, metrics, danh sách test cases     |
| **Executor**     | Lớp bạn tự viết — gọi AI system của bạn, trả về `ExecutorResponse` |
| **EvalPipeline** | Điều phối toàn bộ quy trình: đọc suite → chạy executor → chấm điểm |
| **Judge model**  | LLM dùng để chấm điểm (đọc từ `.env`, có thể override)             |

**Bốn loại bài toán được hỗ trợ:**

| Type         | Dùng cho                                                      |
| ------------ | ------------------------------------------------------------- |
| `llm`        | Đánh giá câu trả lời đơn của LLM (Q&A, chatbot đơn giản)      |
| `rag`        | Đánh giá hệ thống RAG (câu trả lời + ngữ cảnh đã retrieve)    |
| `agent`      | Đánh giá agent gọi tool (single-turn, kiểm tra tool đúng/sai) |
| `multi_turn` | Đánh giá hội thoại nhiều lượt, agent nhớ context qua các lượt |

---

## 4. Bước 1 — Viết file Suite JSON

Tạo file JSON trong thư mục `suites/<tên-project>/`.

### 4.1 Cấu trúc chung

```json
{
  "eval_suite": "Tên hiển thị trên Langfuse",
  "type": "llm | rag | agent | multi_turn",
  "default_model": "gpt-4o",
  "default_threshold": 0.7,
  "default_metrics": [
    { "name": "tên_metric", "threshold": 0.8 }
  ],
  "test_cases": [ ... ]
}
```

- `default_model` — judge model dùng để chấm điểm (có thể override trong `EvalPipeline`)
- `default_threshold` — ngưỡng pass/fail mặc định
- `default_metrics` — áp dụng cho tất cả cases; từng case có thể ghi đè bằng field `metrics` riêng

---

### 4.2 Type: `llm`

Dùng khi cần đánh giá câu trả lời tự do của một LLM (chatbot, Q&A, tóm tắt…).

```json
{
  "eval_suite": "My LLM Suite",
  "type": "llm",
  "default_model": "gpt-4o",
  "default_threshold": 0.7,
  "default_metrics": [
    { "name": "answer_relevancy", "threshold": 0.7 },
    { "name": "bias", "threshold": 0.8 }
  ],
  "test_cases": [
    {
      "id": "LLM-001",
      "name": "Câu hỏi chính sách",
      "input": "Chính sách đổi trả hàng hóa là gì?",
      "expected_output": "Hàng đổi trả trong 7 ngày, còn nguyên seal"
    },
    {
      "id": "LLM-002",
      "name": "Tóm tắt văn bản",
      "input": "Tóm tắt đoạn văn sau: ...",
      "expected_output": "Bản tóm tắt ngắn gọn",
      "metrics": [
        { "name": "summarization", "threshold": 0.75 },
        {
          "name": "g_eval",
          "criteria": "Bản tóm tắt có đầy đủ ý chính và không sai sự thật",
          "threshold": 0.8
        }
      ]
    }
  ]
}
```

**Trường bắt buộc:** `id`, `name`, `input`  
**Trường tùy chọn:** `expected_output` (cần cho một số metrics như `exact_match`)

---

### 4.3 Type: `rag`

Dùng khi hệ thống của bạn retrieve các đoạn văn bản rồi tổng hợp câu trả lời.

```json
{
  "eval_suite": "My RAG Suite",
  "type": "rag",
  "default_model": "gpt-4o",
  "default_threshold": 0.75,
  "default_metrics": [
    { "name": "answer_relevancy", "threshold": 0.7 },
    { "name": "faithfulness", "threshold": 0.8 },
    { "name": "contextual_precision", "threshold": 0.7 },
    { "name": "contextual_recall", "threshold": 0.7 }
  ],
  "test_cases": [
    {
      "id": "RAG-001",
      "name": "Hỏi về quy trình hoàn tiền",
      "input": "Quy trình hoàn tiền mất bao lâu?",
      "expected_output": "Hoàn tiền trong 5-7 ngày làm việc",
      "retrieval_context": [
        "Yêu cầu hoàn tiền được xử lý trong 2 ngày làm việc",
        "Tiền hoàn về tài khoản sau 3-5 ngày tùy ngân hàng",
        "Khách hàng nhận email xác nhận khi hoàn tiền thành công"
      ]
    }
  ]
}
```

**Trường bắt buộc:** `id`, `name`, `input`, `expected_output`, `retrieval_context`

> **Lưu ý:** `retrieval_context` trong JSON là fallback — nếu `ExecutorResponse.retrieval_context` có dữ liệu thì framework ưu tiên dùng giá trị đó.

---

### 4.4 Type: `agent`

Dùng khi agent gọi tool để hoàn thành nhiệm vụ. Framework kiểm tra agent gọi đúng tool với đúng arguments.

```json
{
  "eval_suite": "My Agent Suite",
  "type": "agent",
  "default_model": "gpt-4o",
  "default_threshold": 0.85,
  "default_metrics": [
    { "name": "tool_correctness", "threshold": 0.9 },
    { "name": "argument_correctness", "threshold": 0.85 },
    { "name": "task_completion", "threshold": 0.8 }
  ],
  "test_cases": [
    {
      "id": "AGENT-001",
      "name": "Tra cứu đơn hàng",
      "input": "Lấy đơn hàng của khách KH007",
      "expected_output": "Danh sách đơn hàng của KH007",
      "expected_tools": [
        {
          "name": "lay_don_hang_theo_khach_hang",
          "arguments": { "customer_code": "KH007" }
        }
      ]
    },
    {
      "id": "AGENT-002",
      "name": "Agent gọi nhiều tool liên tiếp",
      "input": "Tìm khách Minh Phát rồi lấy công nợ",
      "expected_output": "Công nợ của khách Minh Phát",
      "expected_tools": [
        {
          "name": "tim_kiem_khach_hang",
          "arguments": { "keyword": "Minh Phát" }
        },
        {
          "name": "lay_cong_no_khach_hang",
          "arguments": { "customer_code": "KH012" }
        }
      ]
    }
  ]
}
```

**Trường bắt buộc:** `id`, `name`, `input`  
**Trường tùy chọn:** `expected_output`, `expected_tools` (danh sách tool theo thứ tự gọi)

---

### 4.5 Type: `multi_turn`

Dùng khi cần đánh giá agent qua nhiều lượt hội thoại liên tiếp, giữ context giữa các lượt.

```json
{
  "eval_suite": "My Multi-turn Suite",
  "type": "multi_turn",
  "default_model": "gpt-4o",
  "default_threshold": 0.8,
  "default_metrics": [
    { "name": "tool_correctness", "threshold": 0.9 },
    { "name": "argument_correctness", "threshold": 0.85 },
    { "name": "conversation_completeness", "threshold": 0.8 }
  ],
  "test_cases": [
    {
      "id": "MT-001",
      "name": "Tìm khách → lấy chi tiết → xem công nợ",
      "criteria": [
        "Lượt 1: tìm kiếm đúng từ khóa",
        "Lượt 2: tự suy luận mã khách từ lượt trước",
        "Lượt 3: giữ nguyên mã khách, gọi tool công nợ"
      ],
      "turns": [
        {
          "content": "Tìm khách hàng tên Hùng Cường",
          "expected_output": "Tìm thấy Hùng Cường, mã KH007",
          "expected_tools": [
            {
              "name": "tim_kiem_khach_hang",
              "arguments": { "keyword": "Hùng Cường" }
            }
          ]
        },
        {
          "content": "Xem chi tiết hồ sơ khách đó",
          "expected_output": "Hồ sơ đầy đủ của KH007",
          "expected_tools": [
            {
              "name": "lay_chi_tiet_khach_hang",
              "arguments": { "customer_code": "KH007" }
            }
          ]
        },
        {
          "content": "Cho xem luôn công nợ",
          "expected_output": "Công nợ hiện tại của KH007",
          "expected_tools": [
            {
              "name": "lay_cong_no_khach_hang",
              "arguments": { "customer_code": "KH007" }
            }
          ]
        }
      ]
    }
  ]
}
```

**Trường bắt buộc:** `id`, `name`, `turns` (mỗi turn có `content`)  
**Trường tùy chọn mỗi turn:** `expected_output`, `expected_tools`, `retrieval_context`

> **Cách framework xử lý multi_turn:** mỗi lượt `content` được gọi `executor.execute()` riêng với cùng `session_id` — executor phải tự quản lý session/memory giữa các lượt.

---

## 5. Bước 2 — Implement Executor

Executor là cầu nối giữa framework và AI system của bạn. Kế thừa `BaseTaskExecutor` và implement phương thức `execute()`.

```python
# my_executor.py
import asyncio
from eval_framework import EvalPipeline, BaseTaskExecutor, ExecutorResponse

class MyExecutor(BaseTaskExecutor):

    def __init__(self):
        # Khởi tạo AI client, load model, v.v.
        pass

    async def execute(
        self,
        input_text: str,
        context: dict = None,
        session_id: str = None,
        **kwargs,
    ) -> ExecutorResponse:
        # Gọi AI system của bạn ở đây
        result = await my_ai_system.run(input_text, session_id=session_id)

        return ExecutorResponse(
            content=result.answer,           # str — câu trả lời của agent
            tool_calls=result.tools_called,  # list[dict] — xem định dạng bên dưới
            retrieval_context=result.chunks, # list[str] — chỉ cần cho RAG
            trace_id=result.trace_id,        # str|None — Langfuse trace ID nếu có
        )
```

### Định dạng `tool_calls`

Mỗi phần tử trong `tool_calls` là một dict:

```python
{
    "name": "tên_tool",          # str — tên hàm được gọi
    "arguments": {               # dict — các tham số đã truyền vào
        "param1": "value1",
        "param2": 123,
    }
}
```

### Mapping theo loại bài toán

| Bài toán   | `content`               | `tool_calls`           | `retrieval_context` |
| ---------- | ----------------------- | ---------------------- | ------------------- |
| LLM        | Câu trả lời             | Không cần              | Không cần           |
| RAG        | Câu trả lời             | Không cần              | **Bắt buộc**        |
| Agent      | Câu trả lời cuối        | **Bắt buộc**           | Không cần           |
| Multi-turn | Câu trả lời của lượt đó | Tool gọi trong lượt đó | Tùy                 |

---

## 6. Bước 3 — Chạy evaluation

### Qua CLI (đơn giản nhất)

```bash
# Test nhanh không cần API key nào
python run_eval.py --suite suites/my_project/agent_suite.json --mode local --executor mock

# Chạy với AI system thật, lưu kết quả lên Langfuse
python run_eval.py --suite suites/my_project/agent_suite.json --mode full --executor real

# Đặt tên experiment trên Langfuse
python run_eval.py --suite suites/my_project/agent_suite.json --mode full --run-name "v2-hotfix"
```

### Qua Python (linh hoạt hơn)

```python
# run_my_project.py
import asyncio
from eval_framework import EvalPipeline
from my_executor import MyExecutor

async def main():
    pipeline = EvalPipeline(
        suite_path="suites/my_project/agent_suite.json",
        executor=MyExecutor(),
        run_name="my-project-v1",
        mode="local",           # "local" | "full" | "sync_only" | "experiment_only"
    )
    summary = await pipeline.run()
    print(f"Passed: {summary.passed}/{summary.total}")

asyncio.run(main())
```

#### Override judge model (không cần chỉnh .env)

```python
from deepeval.models import GPTModel

my_judge = GPTModel(
    model="gpt-4o",
    api_key="sk-...",
)

pipeline = EvalPipeline(
    suite_path="...",
    executor=MyExecutor(),
    judge_model=my_judge,   # ưu tiên hơn VLM_* trong .env
)
```

### Bảng tóm tắt các mode

| Mode              | Langfuse  | Judge model    | Dùng khi                        |
| ----------------- | --------- | -------------- | ------------------------------- |
| `local`           | Không cần | Cần (trừ mock) | Debug, CI/CD nhanh              |
| `full`            | Cần       | Cần            | Chạy thí nghiệm đầy đủ          |
| `sync_only`       | Cần       | Không cần      | Chỉ upload dataset              |
| `experiment_only` | Cần       | Cần            | Dataset đã có sẵn trên Langfuse |

---

## 7. Danh sách Metrics

Khai báo trong `default_metrics` hoặc `metrics` của từng case bằng `"name"`.

### RAG

| Tên                    | Mô tả                                             | Cần judge model |
| ---------------------- | ------------------------------------------------- | --------------- |
| `answer_relevancy`     | Câu trả lời có liên quan đến câu hỏi không        | Có              |
| `faithfulness`         | Câu trả lời có bịa đặt so với context không       | Có              |
| `contextual_precision` | Các chunk retrieve có đúng thứ tự ưu tiên không   | Có              |
| `contextual_recall`    | Tất cả thông tin cần thiết có được retrieve không | Có              |
| `contextual_relevancy` | Các chunk retrieve có liên quan đến câu hỏi không | Có              |
| `hallucination`        | Mức độ thông tin bịa đặt                          | Có              |

### Agent

| Tên                    | Mô tả                                  | Cần judge model |
| ---------------------- | -------------------------------------- | --------------- |
| `tool_correctness`     | Đúng tool được gọi không               | Có              |
| `argument_correctness` | Arguments truyền vào có đúng không     | Có              |
| `task_completion`      | Agent có hoàn thành nhiệm vụ không     | Có              |
| `goal_accuracy`        | Agent có đạt được mục tiêu đề ra không | Có              |

### Multi-turn

| Tên                         | Mô tả                                              | Cần judge model |
| --------------------------- | -------------------------------------------------- | --------------- |
| `conversation_completeness` | Tất cả yêu cầu trong hội thoại có được xử lý không | Có              |
| `knowledge_retention`       | Agent có nhớ thông tin từ lượt trước không         | Có              |
| `role_adherence`            | Agent có giữ đúng role được giao không             | Có              |

### LLM / Tổng quát

| Tên                | Mô tả                                              | Ghi chú                                                        |
| ------------------ | -------------------------------------------------- | -------------------------------------------------------------- |
| `g_eval`           | Đánh giá tự do theo tiêu chí bạn định nghĩa        | Cần thêm field `"criteria"`                                    |
| `bias`             | Phát hiện thiên kiến trong câu trả lời             |                                                                |
| `toxicity`         | Phát hiện nội dung độc hại                         |                                                                |
| `summarization`    | Chất lượng tóm tắt                                 |                                                                |
| `exact_match`      | Khớp chính xác với `expected_output`               | Không cần judge model                                          |
| `json_correctness` | Câu trả lời có đúng JSON schema không              | Cần thêm `"criteria"`                                          |
| `prompt_alignment` | Câu trả lời có tuân theo prompt instructions không | Cần thêm `"criteria"`                                          |
| `topic_adherence`  | Câu trả lời có đúng chủ đề không                   | Cần thêm `"criteria"` (danh sách topic phân cách bởi dấu phẩy) |

#### Ví dụ các metric cần `criteria`

```json
{ "name": "g_eval",
  "criteria": "Câu trả lời có chính xác, đầy đủ và không sai sự thật",
  "threshold": 0.8 }

{ "name": "json_correctness",
  "criteria": "{ \"order_id\": str, \"status\": str, \"total\": float }",
  "threshold": 0.9 }

{ "name": "prompt_alignment",
  "criteria": "Trả lời bằng tiếng Việt\nKhông đề cập đến đối thủ cạnh tranh\nTối đa 3 câu",
  "threshold": 0.85 }

{ "name": "topic_adherence",
  "criteria": "chính sách bảo hành, quy trình đổi trả, thanh toán",
  "threshold": 0.8 }
```

---

## 8. Xem kết quả

### Terminal output

```
══════════════════════════════════════════════════
📊 FINAL EVALUATION REPORT
══════════════════════════════════════════════════
✅ Total Cases:  5
🟢 Passed:       4
🔴 Failed:       1
⚠️  Errors:       0
──────────────────────────────────────────────────
📈 Average Metric Scores:
  - tool_correctness         : 0.92
  - argument_correctness     : 0.88
  - task_completion          : 0.85
══════════════════════════════════════════════════
```

### Langfuse (mode `full`)

Khi chạy `--mode full`, framework sẽ:

1. Upload tất cả test cases lên **Langfuse Dataset** (tên = `eval_suite` trong JSON)
2. Chạy thí nghiệm và gắn điểm DeepEval vào từng trace
3. In URL để xem kết quả: `🔗 Langfuse: https://cloud.langfuse.com/...`

---

## 9. Mở rộng framework

### Thêm loại bài toán mới

Tạo schema và case class, sau đó đăng ký:

```python
# custom_types.py
from pydantic import Field
from eval_framework.core.base_schema import BaseCaseSchema
from eval_framework.core.base_case import BaseEvalCase
from eval_framework.core.schema_registry import register_type

class ClassificationCaseSchema(BaseCaseSchema):
    input: str
    expected_label: str
    candidate_labels: list[str] = Field(default_factory=list)

class ClassificationEvalCase(BaseEvalCase):
    async def run(self):
        response = await self.executor.execute(self.config.input)
        self.actual_responses = [response.content]

    def build_test_case(self):
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=self.config.input,
            actual_output=self.actual_responses[0],
            expected_output=self.config.expected_label,
        )

    def get_default_metrics(self):
        from deepeval.metrics import ExactMatchMetric
        return [ExactMatchMetric(threshold=1.0)]

# Đăng ký — gọi trước khi tạo EvalPipeline
register_type("classification", ClassificationCaseSchema, ClassificationEvalCase)
```

Sau đó dùng `"type": "classification"` trong file JSON.

### Thêm metric mới

```python
from eval_framework.core.metric_builder import register_metric
from my_metrics import MyCustomMetric

# factory_fn(threshold, model, metric_config) → BaseMetric
register_metric(
    "my_custom_metric",
    lambda threshold, model, cfg: MyCustomMetric(threshold=threshold, model=model)
)
```

Sau đó khai báo `{ "name": "my_custom_metric", "threshold": 0.8 }` trong JSON như bình thường.
