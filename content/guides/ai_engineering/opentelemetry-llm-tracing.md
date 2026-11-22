---
title: "Observability & OpenTelemetry Tracing for AI Systems"
slug: "opentelemetry-llm-tracing"
topic: "ai_engineering"
topic_label: "AI Engineering"
difficulty: "Intermediate"
reading_time: "12 min"
order: 5
certification_track: "certified-ai-engineer"
milestone_id: "aie-m3"
sandbox_app:
  title: "Inspect Token & Latency Telemetry in LLM Benchmark"
  description: "Measure end-to-end execution timing across parameter configurations."
  url: "/llm-benchmark"
unlocked_roles:
  - "LLMOps / AI Platform Engineer"
  - "Production AI Engineer"
---

## Executive Summary
Standard HTTP status logs (`200 OK`) cannot tell you why an AI pipeline took 6 seconds, spent $0.14 on a single query, or hallucinated a citation. **LLM Observability** uses distributed tracing (OpenTelemetry GenAI semantic conventions) to capture nested spans across query rewriting, vector retrieval, reranking, tool calls, and model generation—linking every output to its exact prompt version, token count, latency breakdown, and user feedback.

## Distributed Trace Span Tree

```text
[Trace ID: tr_98a1f] POST /api/chat (Total: 1,420 ms | Cost: $0.0042)
 ├── [Span 1: Query Rewrite]       180 ms | gemini-3.1-flash-lite | 120 in / 35 out
 ├── [Span 2: Hybrid Vector Search] 65 ms | top_k=20 | max_score=0.84
 ├── [Span 3: Cross-Encoder Rerank] 95 ms | top_n=5
 └── [Span 4: Final Synthesis]   1,080 ms | gemini-2.5-flash | TTFT: 310 ms | 2,400 in / 380 out
```
## Production Implementation (Python + Structured JSON)

```python
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class SpanRecord:
    trace_id: str
    span_name: str
    duration_ms: float = 0.0
    attributes: dict = field(default_factory=dict)

@contextmanager
def trace_llm_step(trace_id: str, span_name: str, **initial_attrs):
    record = SpanRecord(trace_id=trace_id, span_name=span_name, attributes=dict(initial_attrs))
    start = time.perf_counter()
    try:
        yield record
    finally:
        record.duration_ms = round((time.perf_counter() - start) * 1000, 2)
        # In production, export asynchronously to OpenTelemetry collector / DB
        print(f"[TRACE {record.trace_id}] {record.span_name}: {record.duration_ms}ms | {record.attributes}")

# Example usage
tid = uuid.uuid4().hex[:8]
with trace_llm_step(tid, "synthesis_generation", model="gemini-2.5-flash") as span:
    # Simulate model call
    time.sleep(0.05)
    span.attributes["input_tokens"] = 1850
    span.attributes["output_tokens"] = 240
```
