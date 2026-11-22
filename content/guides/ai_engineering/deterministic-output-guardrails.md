---
title: "Deterministic Output Guardrails & Policy Verification"
slug: "deterministic-output-guardrails"
topic: "ai_engineering"
topic_label: "AI Engineering"
difficulty: "Intermediate"
reading_time: "10 min"
order: 9
certification_track: "certified-ai-engineer"
milestone_id: "aie-m4"
sandbox_app:
  title: "Validate Output Policies in Prompt Security Firewall"
  description: "Test output verification rules and policy compliance checks."
  url: "/prompt-firewall"
unlocked_roles:
  - "AI Safety & Guardrails Specialist"
  - "Production AI Engineer"
---

## Executive Summary
Input guardrails stop bad prompts from entering, while **Output Guardrails** intercept model responses *before* they reach the user or trigger downstream actions. Production output guardrails combine fast deterministic checks (regex, banned competitor/topic lists, citation ID verification, SQL read-only AST parsing) with lightweight semantic policy classifiers to guarantee compliance.

## Post-Generation Verification Pipeline

```text
[LLM Raw Output] ──▶ [1. Structural & Schema Check (1ms)]
                           │
                           ▼
                     [2. Deterministic Policy Rules: Citations, Banned Claims, SQL AST (2ms)]
                           │
                           ▼
                     [3. Groundedness / Toxicity Classifier (Optional, 40ms)]
                           │
                    ┌──────┴──────┐
                 (PASS)        (FAIL)
                    ▼             ▼
          [Deliver to User]  [Safe Fallback Message / Auto-Repair Retry]
```
## Production Implementation (Python + Structured JSON)

```python
import re

def verify_citations(answer_text: str, valid_chunk_ids: set[str]) -> tuple[bool, list[str]]:
    cited_ids = set(re.findall(r"\[(doc_\w+)\]", answer_text))
    hallucinated = sorted(cited_ids - valid_chunk_ids)
    return len(hallucinated) == 0, hallucinated

def is_safe_readonly_sql(sql_query: str) -> bool:
    forbidden = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|GRANT|EXEC)\b", re.IGNORECASE)
    cleaned = sql_query.strip().rstrip(";")
    if ";" in cleaned:  # Block stacked multi-statement injection
        return False
    if not cleaned.upper().startswith(("SELECT", "WITH")):
        return False
    return not bool(forbidden.search(cleaned))
```
