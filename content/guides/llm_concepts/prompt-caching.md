---
title: "Prompt Caching: Cutting Latency & Token Costs"
slug: "prompt-caching"
topic: "llm_concepts"
topic_label: "LLM Concepts"
difficulty: "Intermediate"
reading_time: "10 min"
order: 3
certification_track: "certified-ai-engineer"
milestone_id: "aie-m1"
sandbox_app:
  title: "Measure Time-to-First-Token in LLM Benchmark"
  description: "Test repeated prefix prompts to observe latency improvements."
  url: "/llm-benchmark"
unlocked_roles:
  - "LLMOps / AI Platform Engineer"
  - "Production AI Engineer"
---

## Executive Summary
In production applications, 70–90% of prompt tokens are often identical across requests—such as long system instructions, few-shot examples, tool schemas, or a shared knowledge corpus. **Prompt Caching** stores the computed Key-Value (KV) attention tensors of static prompt prefixes so subsequent requests skip prefill computation, reducing Time-to-First-Token (TTFT) by up to 80% and cutting input token costs by 50–90%.

## Prefix Cache Hit vs. Miss Architecture

```text
[Static Prefix: System Rules + 30k Corpus + Tools] ──▶ [KV Cache Hit (90% Discount)]
                                                               │
[Dynamic Suffix: User Turn #14] ───────────────────────▶ [Prefill Only Suffix] ──▶ [Output]
```

## Production Implementation (Python + Structured JSON)

```python
# GOOD: Static instructions & large reference corpus are pinned FIRST (Cache Hit)
def build_cacheable_prompt(static_policy_manual: str, few_shot_examples: str, user_question: str) -> str:
    return f"""<SYSTEM_INSTRUCTIONS>
You are an enterprise compliance auditor. Answer strictly using the policy manual below.
</SYSTEM_INSTRUCTIONS>

<POLICY_MANUAL>
{static_policy_manual}
</POLICY_MANUAL>

<EXAMPLES>
{few_shot_examples}
</EXAMPLES>

<USER_QUERY>
{user_question}
</USER_QUERY>"""

# BAD: Putting a timestamp or user_id at line 1 invalidates the entire prefix cache on every call!
```
