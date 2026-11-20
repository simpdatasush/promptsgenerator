---
title: "LLM-as-a-Judge: Automated Evaluation at Scale"
slug: "llm-as-a-judge"
topic: "ai_engineering"
topic_label: "AI Engineering"
difficulty: "Intermediate"
reading_time: "11 min"
order: 3
certification_track: "certified-ai-engineer"
milestone_id: "aie-m2"
sandbox_app:
  title: "Test Evaluation Prompts in LLM Benchmark"
  description: "Compare judge consistency across models and temperature settings."
  url: "/llm-benchmark"
unlocked_roles:
  - "Production AI Engineer"
  - "LLMOps / AI Platform Engineer"
---

## Executive Summary
Human annotation does not scale when shipping daily prompt, retrieval, or model updates. **LLM-as-a-Judge** uses a high-reasoning evaluator model constrained by a strict scoring rubric and structured JSON schema to grade production outputs for faithfulness, relevance, and policy compliance.

## Evaluation Architecture Flow

```text
[Golden Dataset Input] ──▶ [Production AI Pipeline] ──▶ [Candidate Output]
                                                               │
[Strict 1-5 Rubric + Schema] ──────────────────────▶ [Judge LLM (Temp=0)]
                                                               │
                                                               ▼
                                              [Structured JSON Verdict + Reason]
```
## Calibration & Trade-Offs

| Judge Pattern | Best Use Case | Primary Bias Risk | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Pointwise Scoring (1–5)** | Faithfulness & groundedness checks against retrieved context | Central tendency bias (clustering around 3 or 4) | Use binary (`PASS`/`FAIL`) or anchored 1–3 integer rubrics |
| **Pairwise Comparison** | Comparing Prompt A vs. Prompt B before deployment | Position bias (favoring the first response shown) | Run twice with swapped A/B order and average verdicts |
| **Reference-Guided** | Technical Q&A with a known ground-truth answer | Over-penalizing valid phrasing variations | Instruct judge to grade factual equivalence, not exact wording |

## Production Implementation (Python + Structured JSON)

```python
from pydantic import BaseModel, Field
from google import genai

class FaithfulnessVerdict(BaseModel):
    reasoning: str = Field(description="Step-by-step verification against context.")
    unsupported_claims: list[str] = Field(description="Claims not found in source text.")
    passed: bool = Field(description="True only if zero unsupported claims exist.")

def evaluate_faithfulness(context: str, answer: str) -> FaithfulnessVerdict:
    client = genai.Client()
    prompt = f"""
    Verify whether the ANSWER is strictly supported by the CONTEXT.
    CONTEXT: {context}
    ANSWER: {answer}
    """
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "response_schema": FaithfulnessVerdict,
        },
    )
    return FaithfulnessVerdict.model_validate_json(response.text)
```
## Failure Modes & Guardrails
* **Self-Enhancement Bias:** Models tend to rate their own outputs higher. Use a distinct model family or stronger reasoning tier as the judge.
* **Non-Deterministic Scores:** Always set `temperature=0.0` and require a `reasoning` field *before* the boolean/numeric verdict in your JSON schema so the model reasons prior to scoring.

## Interview & Certification Check
1. Why should `reasoning` precede `score` in a structured output schema for an LLM judge?
2. How do you measure whether your LLM judge aligns with human domain experts? (Answer: *Cohen's Kappa* or *Spearman rank correlation* on a 100-item human-labeled calibration set.)
