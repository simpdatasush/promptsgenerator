---
title: "Fallback & Resilience Patterns: Circuit Breakers & Retries"
slug: "fallback-resilience-patterns"
topic: "architecture"
topic_label: "Architecture"
difficulty: "Advanced"
reading_time: "11 min"
order: 3
certification_track: "certified-ai-engineer"
milestone_id: "aie-m3"
sandbox_app:
  title: "Test Model Reliability in LLM Benchmark"
  description: "Verify backup model outputs for parity with your primary model."
  url: "/llm-benchmark"
unlocked_roles:
  - "Principal / Staff AI Architect"
  - "Production AI Engineer"
---

## Executive Summary
External LLM APIs experience transient rate limits (`429`), regional capacity spikes (`503`), and tail-latency stalls. **Fallback & Resilience Patterns** protect user-facing SLAs using bounded exponential backoff with jitter, strict per-call timeouts, circuit breakers that trip during provider outages, and cross-provider secondary models.

## Resilient Inference Execution Flow

```text
[Request] ──▶ [Circuit Breaker Check]
                    │
        ┌───────────┴───────────┐
   (CLOSED: Healthy)       (OPEN: Provider Down)
        ▼                       │
[Primary Model (Timeout: 8s)]   │
        │                       │
  ┌─────┴─────┐                 │
(200 OK)   (429/503/Timeout)    │
  │           ▼                 │
  │    [Retry x1 w/ Jitter]     │
  │           │ (Still Failing) │
  │           └────────┬────────┘
  │                    ▼
  │         [Secondary Fallback Model]
  └──────────┬─────────┘
             ▼
     [Validated Response]
```
## Production Implementation (Python + Structured JSON)

```python
import random
import time
from typing import Callable, Any

def call_with_resilience(
    primary_fn: Callable[[], Any],
    fallback_fn: Callable[[], Any],
    max_retries: int = 2,
    base_delay: float = 0.4,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return primary_fn()
        except Exception as err:
            # Do not retry non-transient validation errors
            if "400" in str(err) or "InvalidArgument" in str(err):
                raise
            if attempt < max_retries:
                jitter = random.uniform(0, 0.25)
                sleep_time = (base_delay * (2 ** attempt)) + jitter
                time.sleep(sleep_time)
            else:
                # Primary exhausted -> fail over to secondary model/provider
                return fallback_fn()
```
