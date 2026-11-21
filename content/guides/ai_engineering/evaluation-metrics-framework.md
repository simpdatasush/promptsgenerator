---
title: "Evaluation Metrics Framework: Deterministic vs. Semantic"
slug: "evaluation-metrics-framework"
topic: "ai_engineering"
topic_label: "AI Engineering"
difficulty: "Intermediate"
reading_time: "12 min"
order: 2
certification_track: "certified-ai-engineer"
milestone_id: "aie-m2"
sandbox_app:
  title: "Benchmark Response Consistency in LLM Benchmark"
  description: "Compare deterministic and semantic scoring behavior across models."
  url: "/llm-benchmark"
unlocked_roles:
  - "Production AI Engineer"
  - "LLMOps / AI Platform Engineer"
---

## Executive Summary
Relying on a single accuracy number hides critical failure modes in AI systems. A robust **Evaluation Metrics Framework** separates **Retrieval Metrics** (did we fetch the right facts?) from **Generation Metrics** (did the LLM answer faithfully without hallucinating?), combining fast deterministic assertions with semantic LLM-as-a-Judge checks.

## Two-Stage Evaluation Decomposition

```text
                        ┌──▶ [Retrieval Metrics: Recall@K, MRR, NDCG]
                        │
[Query + Golden Set] ───┤
                        │
                        └──▶ [Generation Metrics: Faithfulness, Answer Relevance, Schema Validity]
```
##Production Implementation (Hybrid Assertion + Semantic Runner)

```python
from dataclasses import dataclass

@dataclass
class EvalResult:
    case_id: str
    schema_valid: bool
    recall_at_k: float
    faithfulness_passed: bool

def compute_recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int = 5) -> float:
    if not expected_doc_ids:
        return 1.0
    top_k = set(retrieved_doc_ids[:k])
    hits = sum(1 for doc_id in expected_doc_ids if doc_id in top_k)
    return hits / len(expected_doc_ids)
```
