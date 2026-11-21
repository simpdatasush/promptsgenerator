---
title: "Golden Datasets: Curating Evaluation Benchmarks"
slug: "golden-datasets-curation"
topic: "ai_engineering"
topic_label: "AI Engineering"
difficulty: "Intermediate"
reading_time: "10 min"
order: 1
certification_track: "certified-ai-engineer"
milestone_id: "aie-m2"
sandbox_app:
  title: "Generate & Inspect Synthetic Test Cases in LLM Benchmark"
  description: "Test prompt behavior across edge-case inputs and compare outputs."
  url: "/llm-benchmark"
unlocked_roles:
  - "Production AI Engineer"
  - "LLMOps / AI Platform Engineer"
---

## Executive Summary
An AI evaluation pipeline is only as trustworthy as its test cases. A **Golden Dataset** is a version-controlled, human-verified collection of `(input, expected_context, reference_output, failure_tag)` records covering normal traffic, adversarial inputs, and historical production bugs—giving engineering teams an objective yardstick before shipping any change.

## Golden Dataset Lifecycle

```text
[Production Logs (Thumbs Down / Edge Cases)] ──┐
                                              ├──▶ [PII Scrubbing + SME Labeling] ──▶ [Versioned Golden Set (v1.4)]
[Synthetic Edge-Case Generator (LLM)] ────────┘                                                │
                                                                                               ▼
                                                                                    [Automated CI/CD Eval Gate]
```

## Production Implementation (Python + Structured JSON)

```python

import json
from pydantic import BaseModel, Field

class GoldenTestCase(BaseModel):
    case_id: str
    slice_tag: str = Field(description="happy_path | multi_hop | abstention | adversarial")
    user_query: str
    ground_truth_context: list[str]
    expected_answer: str
    must_not_contain: list[str] = []

def load_golden_dataset(filepath: str) -> list[GoldenTestCase]:
    with open(filepath, "r", encoding="utf-8") as f:
        raw_items = json.load(f)
    return [GoldenTestCase.model_validate(item) for item in raw_items]
```
