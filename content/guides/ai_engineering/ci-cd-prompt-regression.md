---
title: "CI/CD Regression Testing for Prompts & Models"
slug: "ci-cd-prompt-regression"
topic: "ai_engineering"
topic_label: "AI Engineering"
difficulty: "Advanced"
reading_time: "11 min"
order: 4
certification_track: "certified-ai-engineer"
milestone_id: "aie-m2"
sandbox_app:
  title: "Run Side-by-Side Prompt Sweeps in LLM Benchmark"
  description: "Test prompt variants before merging changes into production."
  url: "/llm-benchmark"
unlocked_roles:
  - "LLMOps / AI Platform Engineer"
  - "Production AI Engineer"
---

## Executive Summary
A tiny wording tweak in a system prompt or a minor model version bump can silently break edge-case reasoning. **CI/CD Regression Testing** treats prompts, schemas, and retrieval configs as versioned code artifacts—automatically executing the Golden Dataset suite on every pull request and blocking deployment if quality, latency, or token cost regresses beyond strict thresholds.

## Automated Pull-Request Evaluation Gate

```text
[Git Push: Prompt / Model / Chunker Change]
                     │
                     ▼
     [CI Runner: Execute Golden Dataset (N=100)]
                     │
                     ├──▶ [Compare vs. Main Branch Baseline]
                     │      • Faithfulness >= 95% (Delta >= -1%)
                     │      • Schema Pass == 100%
                     │      • P95 Latency <= 1.8s
                     │
              ┌──────┴──────┐
              ▼             ▼
        [PASS: Merge]  [FAIL: Block PR + Post Diff Report]
```

## Production Implementation (Python + Structured JSON)

```python
import pytest

BASELINE_FAITHFULNESS = 0.94
MIN_SCHEMA_PASS_RATE = 1.00

def test_production_eval_regression():
    # In CI, replace with actual batch runner over golden_dataset.json
    eval_summary = {
        "schema_pass_rate": 1.00,
        "faithfulness_score": 0.95,
        "p95_latency_ms": 1120,
    }

    assert eval_summary["schema_pass_rate"] >= MIN_SCHEMA_PASS_RATE, (
        f"Schema pass rate dropped to {eval_summary['schema_pass_rate']}"
    )
    assert eval_summary["faithfulness_score"] >= (BASELINE_FAITHFULNESS - 0.015), (
        f"Faithfulness regressed: {eval_summary['faithfulness_score']} < {BASELINE_FAITHFULNESS}"
    )
```
