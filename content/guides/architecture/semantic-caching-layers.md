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
import asyncio

async def fetch_vector_context(query: str) -> list[str]:
    await asyncio.sleep(0.15)  # Simulated vector search I/O
    return ["Policy Doc A: Refunds within 30 days."]

async def fetch_user_account_tier(user_id: str) -> dict:
    await asyncio.sleep(0.12)  # Simulated SQL/Redis lookup
    return {"user_id": user_id, "tier": "enterprise"}

async def gather_prompt_inputs(user_id: str, query: str) -> dict:
    # Execute independent I/O concurrently instead of sequentially
    docs, account = await asyncio.gather(
        fetch_vector_context(query),
        fetch_user_account_tier(user_id),
    )
    return {"docs": docs, "account": account}
```
