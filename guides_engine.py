import os
import json
import yaml
import markdown
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTENT_DIR = os.path.join(BASE_DIR, "content", "guides")

# Map hyphenated guide frontmatter URLs to your exact underscore routes in app.py
#SANDBOX_URL_MAP = {
#    "/llm-benchmark": "/llm_benchmark",
#    "/prompt-firewall": "/prompt_firewall",
#    "/code-reviewer": "/code_reviewer",
#    "/doc-extractor": "/doc_extractor",
#}


class ArchitectureRubricScore(BaseModel):
    architecture_clarity_score: int = Field(ge=0, le=5, description="0-5 score on system topology and component boundaries")
    failure_mode_handling_score: int = Field(ge=0, le=5, description="0-5 score on retries, fallbacks, and edge cases")
    latency_cost_optimization_score: int = Field(ge=0, le=5, description="0-5 score on caching, routing, and token budgets")
    security_isolation_score: int = Field(ge=0, le=5, description="0-5 score on prompt injection, PII, and tenant isolation")
    evaluation_observability_score: int = Field(ge=0, le=5, description="0-5 score on evals, tracing, and regression gates")
    strengths: list[str] = Field(description="2-3 specific architectural strengths in the candidate's response")
    gaps_to_improve: list[str] = Field(description="2-3 concrete missing guardrails or trade-offs")
    examiner_summary: str = Field(description="Concise 3-sentence principal architect feedback")


def _parse_markdown_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    meta = {}
    body = raw

    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()

    # Normalize sandbox_app.url to match app.py underscore routes
    sandbox = meta.get("sandbox_app")
    if isinstance(sandbox, dict) and sandbox.get("url") in SANDBOX_URL_MAP:
        sandbox["url"] = SANDBOX_URL_MAP[sandbox["url"]]

    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "codehilite", "toc"],
        extension_configs={
            "codehilite": {"css_class": "highlight", "guess_lang": False},
            "toc": {"permalink": False, "toc_depth": "2-3"},
        },
    )
    return {
        "meta": meta,
        "html": md.convert(body),
        "toc": md.toc,
    }


def load_all_topics():
    topics_path = os.path.join(CONTENT_DIR, "_topics.yaml")
    if not os.path.exists(topics_path):
        return []

    with open(topics_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    topics_list = []
    for slug, info in data.get("topics", {}).items():
        topic_dir = os.path.join(CONTENT_DIR, slug)
        actual_files = [fn for fn in os.listdir(topic_dir) if fn.endswith(".md")] if os.path.isdir(topic_dir) else []
        item = dict(info)
        item["slug"] = slug
        item["published_count"] = len(actual_files)
        topics_list.append(item)

    topics_list.sort(key=lambda x: x.get("order", 99))
    return topics_list


def load_certifications():
    cert_path = os.path.join(CONTENT_DIR, "_certifications.yaml")
    if not os.path.exists(cert_path):
        return []

    with open(cert_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return list(data.get("certifications", {}).values())


def get_certification_by_slug(track_slug):
    certs = load_certifications()
    return next((c for c in certs if c.get("slug") == track_slug), None)


def get_topic_with_guides(topic_slug):
    topics = load_all_topics()
    topic = next((t for t in topics if t["slug"] == topic_slug), None)
    if not topic:
        return None, []

    topic_dir = os.path.join(CONTENT_DIR, topic_slug)
    guides = []
    if os.path.isdir(topic_dir):
        for fn in os.listdir(topic_dir):
            if fn.endswith(".md"):
                parsed = _parse_markdown_file(os.path.join(topic_dir, fn))
                meta = parsed["meta"]
                meta["slug"] = meta.get("slug", fn[:-3])
                guides.append(meta)

    guides.sort(key=lambda g: g.get("order", 99))
    return topic, guides


def get_single_guide(topic_slug, guide_slug):
    topic, sibling_guides = get_topic_with_guides(topic_slug)
    if not topic:
        return None, None, []

    filepath = os.path.join(CONTENT_DIR, topic_slug, f"{guide_slug}.md")
    if not os.path.exists(filepath):
        return topic, None, sibling_guides

    return topic, _parse_markdown_file(filepath), sibling_guides


def get_milestone_questions(track_slug, milestone_id):
    seed_path = os.path.join(CONTENT_DIR, "_seed_questions_aie.json")
    if not os.path.exists(seed_path):
        return []

    with open(seed_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    return [
        q for q in all_questions
        if q.get("track_slug") == track_slug and q.get("milestone_id") == milestone_id
    ]


def grade_and_save_milestone(db, MilestoneAttempt, user_id, candidate_email, track_slug, milestone_id, submitted_answers, pass_threshold=75):
    questions = get_milestone_questions(track_slug, milestone_id)
    if not questions:
        return None

    correct_count = 0
    details = []

    for q in questions:
        user_choice = submitted_answers.get(q["id"], "")
        is_correct = (user_choice == q["correct_option"])
        if is_correct:
            correct_count += 1
        details.append({
            "id": q["id"],
            "prompt_text": q["prompt_text"],
            "options": q["options"],
            "user_choice": user_choice,
            "correct_option": q["correct_option"],
            "is_correct": is_correct,
            "explanation": q["explanation"],
        })

    total_count = len(questions)
    score_pct = round((correct_count / total_count) * 100, 1)
    passed = score_pct >= pass_threshold

    attempt = MilestoneAttempt(
        user_id=user_id,
        candidate_email=candidate_email.strip().lower(),
        track_slug=track_slug,
        milestone_id=milestone_id,
        attempt_type="milestone",
        score_pct=score_pct,
        correct_count=correct_count,
        total_count=total_count,
        passed=passed,
        details_json=json.dumps(details),
    )
    db.session.add(attempt)
    db.session.commit()

    return {
        "score_pct": score_pct,
        "correct_count": correct_count,
        "total_count": total_count,
        "passed": passed,
        "pass_threshold": pass_threshold,
        "details": details,
    }


def get_candidate_progress(MilestoneAttempt, user_id, candidate_email, track_slug):
    query = MilestoneAttempt.query.filter_by(track_slug=track_slug)
    if user_id:
        attempts = query.filter_by(user_id=user_id).all()
    elif candidate_email:
        attempts = query.filter_by(candidate_email=candidate_email.strip().lower()).all()
    else:
        return {}

    progress = {}
    for a in attempts:
        existing = progress.get(a.milestone_id)
        if not existing:
            progress[a.milestone_id] = {"best_score": a.score_pct, "passed": a.passed}
        else:
            progress[a.milestone_id] = {
                "best_score": max(existing["best_score"], a.score_pct),
                "passed": existing["passed"] or a.passed,
            }
    return progress


def grade_architecture_case_with_gemini(client, gemma_types, case_prompt: str, candidate_solution: str, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Reuses app.py's existing `gemma_client` / `get_v1beta_client()` and `gemma_types`
    to grade a 25-point Architecture Design Case via strict Pydantic JSON schema.
    """
    if not client:
        raise RuntimeError("Gemini client is not configured in app.py.")

    system_instruction = (
        "You are a Principal AI Systems Architect grading a certification design exam. "
        "Score strictly on a 0-5 scale per rubric dimension (25 points total). "
        "Reward concrete engineering mechanisms (schemas, thresholds, delimiters, circuit breakers, eval metrics) "
        "and penalize vague buzzwords."
    )
    prompt = f"DESIGN SCENARIO:\n{case_prompt}\n\nCANDIDATE ARCHITECTURE SUBMISSION:\n{candidate_solution}"

    response = client.models.generate_content(
        model=model_name,
        config=gemma_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=ArchitectureRubricScore,
        ),
        contents=[prompt],
    )
    parsed = json.loads((response.text or "{}").strip())
    total_points = (
        parsed.get("architecture_clarity_score", 0)
        + parsed.get("failure_mode_handling_score", 0)
        + parsed.get("latency_cost_optimization_score", 0)
        + parsed.get("security_isolation_score", 0)
        + parsed.get("evaluation_observability_score", 0)
    )
    parsed["total_points"] = total_points
    parsed["max_points"] = 25
    return parsed

DESIGN_CASES = {
    "certified-ai-engineer": {
        "title": "Enterprise Multi-Tenant Compliance & Support Copilot",
        "scenario": (
            "You are architecting a B2B SaaS AI copilot used by 450 enterprise customers to query internal "
            "HR/security policies (50,000+ PDFs) and trigger automated support ticket triage. "
            "Strict SLAs: P95 TTFT < 600ms, 100% valid JSON payloads for downstream ticket routing, "
            "zero cross-tenant data leakage, and resistance to indirect prompt injection embedded inside uploaded PDFs."
        ),
        "requirements": [
            "1. Model Routing & Schema Enforcement: How you dispatch requests across model tiers and guarantee 100% valid structured JSON without hallucinating missing optional fields.",
            "2. Latency & Cost SLAs: How you use KV prompt caching, semantic caching (with tenant partitioning), and async I/O to meet P95 TTFT < 600ms.",
            "3. Security & Multi-Tenant Isolation: How you defend against indirect prompt injection inside PDFs, redact PII before external LLM calls, and isolate vector searches by org_id.",
            "4. Resilience & Circuit Breakers: How you handle HTTP 429/503 provider outages, timeouts, and retry storms.",
            "5. Evals & Observability: How you structure Golden Datasets, CI/CD PR regression gates, and OpenTelemetry tracing.",
        ],
    }
}


def get_final_exam_bundle(track_slug: str):
    """Selects 8 cross-domain MCQs (2 per milestone) + the track's Architecture Design Case."""
    seed_path = os.path.join(CONTENT_DIR, "_seed_questions_aie.json")
    all_questions = []
    if os.path.exists(seed_path):
        with open(seed_path, "r", encoding="utf-8") as f:
            all_questions = [q for q in json.load(f) if q.get("track_slug") == track_slug]

    # Pick up to 2 questions from each milestone for balanced 8-question coverage
    by_milestone = {}
    for q in all_questions:
        by_milestone.setdefault(q["milestone_id"], []).append(q)

    selected_mcqs = []
    for m_id in sorted(by_milestone.keys()):
        selected_mcqs.extend(by_milestone[m_id][:2])

    design_case = DESIGN_CASES.get(track_slug) or DESIGN_CASES["certified-ai-engineer"]
    return selected_mcqs, design_case
