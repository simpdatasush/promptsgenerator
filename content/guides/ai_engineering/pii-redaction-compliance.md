---
title: "PII Redaction & Data Privacy Compliance (GDPR / HIPAA)"
slug: "pii-redaction-compliance"
topic: "ai_engineering"
topic_label: "AI Engineering"
difficulty: "Intermediate"
reading_time: "10 min"
order: 8
certification_track: "certified-ai-engineer"
milestone_id: "aie-m4"
sandbox_app:
  title: "Test Sensitive Data Filtering in Prompt Security Firewall"
  description: "Verify how personally identifiable information is detected before LLM processing."
  url: "/prompt-firewall"
unlocked_roles:
  - "AI Safety & Guardrails Specialist"
  - "Production AI Engineer"
---

## Executive Summary
Sending raw customer names, emails, tax IDs, or medical records to third-party LLM APIs or storing them in unencrypted trace logs violates GDPR, HIPAA, and enterprise data-processing agreements. **PII Redaction & Anonymization** intercepts sensitive entities *before* inference using deterministic regex + Named Entity Recognition (NER), replacing them with reversible placeholders (`[PERSON_1]`, `[EMAIL_1]`) that can be re-hydrated locally after the LLM responds.

## Reversible Tokenization (Vaulted Anonymization) Flow

```text
[Raw Input: "Email Rahul at rahul@corp.in about INV-902"]
                         │
                         ▼
          [Pre-Inference PII Scrubber]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[Ephemeral Request Vault]      [Sanitized Prompt to LLM]
 [PERSON_1] -> "Rahul"          "Email [PERSON_1] at [EMAIL_1]
 [EMAIL_1]  -> "rahul@corp.in"   about INV-902"
        │                                 │
        │                      [LLM Response with Tokens]
        └────────────────┬────────────────┘
                         ▼
          [Post-Inference Re-Hydrator] ──▶ [Final User Output]
```
## Production Implementation (Python + Structured JSON)

```python
import re

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

def redact_pii(text: str) -> tuple[str, dict[str, str]]:
    vault: dict[str, str] = {}
    counter = 1

    def _replace(match: re.Match, label: str) -> str:
        nonlocal counter
        token = f"[{label}_{counter}]"
        vault[token] = match.group(0)
        counter += 1
        return token

    sanitized = EMAIL_REGEX.sub(lambda m: _replace(m, "EMAIL"), text)
    sanitized = PHONE_REGEX.sub(lambda m: _replace(m, "PHONE"), sanitized)
    return sanitized, vault

def restore_pii(llm_output: str, vault: dict[str, str]) -> str:
    restored = llm_output
    for token, original_value in vault.items():
        restored = restored.replace(token, original_value)
    return restored
```
