---
title: "Structured Outputs & JSON Schema Enforcement"
slug: "structured-outputs-json-schema"
topic: "llm_concepts"
topic_label: "LLM Concepts"
difficulty: "Beginner"
reading_time: "9 min"
order: 1
certification_track: "certified-ai-engineer"
milestone_id: "aie-m1"
sandbox_app:
  title: "Inspect & Validate Code Schemas in Code Reviewer"
  description: "Test Pydantic models and JSON schema edge cases in your Code Reviewer micro-app."
  url: "/code-reviewer"
unlocked_roles:
  - "Generative AI Product Engineer"
  - "Production AI Engineer"
---

## Executive Summary
Downstream software cannot reliably parse free-form paragraphs. **Structured Outputs** constrain an LLM's token sampling at the decoding layer (using Context-Free Grammars) so every response is mathematically guaranteed to match a strict JSON Schema or Pydantic model.

## Constrained Decoding Architecture

```text
[User Input + Prompt] ──▶ [LLM Forward Pass (Logits)]
                                    │
[Pydantic JSON Schema] ──▶ [FSM / Grammar Token Mask] ──▶ [Zero Invalid Tokens]
                                                                 │
                                                                 ▼
                                                [Validated Python Object]
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
