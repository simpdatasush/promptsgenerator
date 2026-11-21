---
title: "Context Engineering & Window Budgeting"
slug: "context-engineering"
topic: "llm_concepts"
topic_label: "LLM Concepts"
difficulty: "Intermediate"
reading_time: "12 min"
order: 2
certification_track: "certified-ai-engineer"
milestone_id: "aie-m1"
sandbox_app:
  title: "Analyze Context Token Impact in LLM Benchmark"
  description: "Benchmark response accuracy and latency as context length scales."
  url: "/llm-benchmark"
unlocked_roles:
  - "Generative AI Product Engineer"
  - "Production AI Engineer"
---

## Executive Summary
**Context Engineering** is the discipline of assembling, ordering, compressing, and pruning every token sent to an LLM so the model has exact signal without noise. Even with 1M+ token context windows, dumping raw unstructured history degrades recall ("Lost in the Middle"), inflates latency, and multiplies inference costs.

## Context Assembly Pipeline

```text
[System Instructions (Pinned Top)]
        +
[Compressed Conversation Summary + Last N Turns]
        +
[Re-Ranked Retrieval Chunks (Highest Relevance at Top & Bottom)]
        +
[Current User Query + Output Schema (Pinned Bottom)] ──▶ [Token Budget Gate] ──▶ [LLM]
```
## Production Implementation (Python + Structured JSON)
```python
from typing import Literal
from pydantic import BaseModel, Field
from google import genai

class SupportTicketTriage(BaseModel):
    reasoning: str = Field(description="Brief analysis of urgency and root cause.")
    category: Literal["billing", "technical_bug", "account_access", "feature_request"]
    priority: Literal["P1", "P2", "P3"]
    requires_human_escalation: bool

def triage_ticket(ticket_text: str) -> SupportTicketTriage:
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"Classify this customer support ticket:\n\n{ticket_text}",
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": SupportTicketTriage,
        },
    )
    return SupportTicketTriage.model_validate_json(response.text)
```
