from flask import Blueprint, render_template, abort, request, session
from flask_login import current_user
import json
from guides_engine import (
    load_all_topics,
    load_certifications,
    get_certification_by_slug,
    get_topic_with_guides,
    get_single_guide,
    get_milestone_questions,
    grade_and_save_milestone,
    get_candidate_progress,
    grade_architecture_case_with_gemini,
    get_final_exam_bundle,
)

guides_bp = Blueprint("guides", __name__, url_prefix="/guides")


def _get_candidate_identity():
    if getattr(current_user, "is_authenticated", False):
        email = current_user.email or f"{current_user.username}@promptsgenerator.ai"
        return current_user.id, email
    return None, session.get("candidate_email", "")


@guides_bp.route("/")
def all_guides():
    topics = load_all_topics()
    certifications = load_certifications()
    return render_template(
        "all_guides.html",
        topics=topics,
        certifications=certifications,
    )


@guides_bp.route("/<topic_slug>")
def guide_topic(topic_slug):
    topic, guides = get_topic_with_guides(topic_slug)
    if not topic:
        abort(404)
    return render_template(
        "guide_topic.html",
        topic=topic,
        guides=guides,
    )


@guides_bp.route("/<topic_slug>/<guide_slug>")
def single_guide(topic_slug, guide_slug):
    topic, guide_data, sibling_guides = get_single_guide(topic_slug, guide_slug)
    if not topic or not guide_data:
        abort(404)
    return render_template(
        "single_guide.html",
        topic=topic,
        meta=guide_data["meta"],
        content_html=guide_data["html"],
        toc_html=guide_data["toc"],
        sibling_guides=sibling_guides,
    )


@guides_bp.route("/certifications/<track_slug>", methods=["GET", "POST"])
def certification_track(track_slug):
    from app import MilestoneAttempt

    cert = get_certification_by_slug(track_slug)
    if not cert:
        abort(404)

    if request.method == "POST":
        email = request.form.get("candidate_email", "").strip().lower()
        if email:
            session["candidate_email"] = email

    user_id, candidate_email = _get_candidate_identity()
    progress = get_candidate_progress(MilestoneAttempt, user_id, candidate_email, track_slug)

    return render_template(
        "certification_track.html",
        cert=cert,
        candidate_email=candidate_email,
        progress=progress,
    )


@guides_bp.route("/certifications/<track_slug>/milestone/<milestone_id>", methods=["GET", "POST"])
def milestone_exam(track_slug, milestone_id):
    from app import db, MilestoneAttempt

    cert = get_certification_by_slug(track_slug)
    if not cert:
        abort(404)

    milestone = next((m for m in cert.get("milestones", []) if m["id"] == milestone_id), None)
    if not milestone:
        abort(404)

    questions = get_milestone_questions(track_slug, milestone_id)
    user_id, candidate_email = _get_candidate_identity()
    result = None

    if request.method == "POST":
        if not candidate_email:
            candidate_email = request.form.get("candidate_email", "").strip().lower() or "guest@promptsgenerator.ai"
            session["candidate_email"] = candidate_email

        submitted = {q["id"]: request.form.get(f"q_{q['id']}", "") for q in questions}
        threshold = milestone.get("pass_threshold", 75)
        result = grade_and_save_milestone(
            db,
            MilestoneAttempt,
            user_id,
            candidate_email,
            track_slug,
            milestone_id,
            submitted,
            pass_threshold=threshold,
        )

    return render_template(
        "guide_exam.html",
        cert=cert,
        milestone=milestone,
        questions=questions,
        candidate_email=candidate_email,
        result=result,
    )

@guides_bp.route("/certifications/<track_slug>/final-exam", methods=["GET", "POST"])
def final_exam(track_slug):
    from app import db, MilestoneAttempt, get_v1beta_client, gemma_types

    cert = get_certification_by_slug(track_slug)
    if not cert:
        abort(404)

    questions, design_case = get_final_exam_bundle(track_slug)
    user_id, candidate_email = _get_candidate_identity()
    pass_threshold = cert.get("pass_threshold") or (cert.get("final_exam") or {}).get("pass_threshold", 80)
    duration_minutes = (cert.get("final_exam") or {}).get("duration_minutes", 60)

    result = None
    error_message = None
    submitted_solution = ""

    if request.method == "POST":
        if not candidate_email:
            candidate_email = request.form.get("candidate_email", "").strip().lower() or "guest@promptsgenerator.ai"
            session["candidate_email"] = candidate_email

        submitted_solution = request.form.get("architecture_solution", "").strip()

        # 1. Grade Part 1: 8 Cross-Domain MCQs (50% weight)
        mcq_correct = 0
        mcq_details = []
        for q in questions:
            user_choice = request.form.get(f"q_{q['id']}", "")
            is_correct = (user_choice == q["correct_option"])
            if is_correct:
                mcq_correct += 1
            mcq_details.append({
                "id": q["id"],
                "prompt_text": q["prompt_text"],
                "user_choice": user_choice,
                "correct_option": q["correct_option"],
                "is_correct": is_correct,
                "explanation": q["explanation"],
            })

        mcq_total = len(questions) or 1
        mcq_pct = round((mcq_correct / mcq_total) * 100, 1)

        # 2. Grade Part 2: 25-Point Architecture Design Case via shared v1beta client (50% weight)
        try:
            case_prompt_text = f"{design_case['title']}\n{design_case['scenario']}\nRequirements:\n" + "\n".join(design_case["requirements"])
            rubric = grade_architecture_case_with_gemini(
                client=get_v1beta_client(),
                gemma_types=gemma_types,
                case_prompt=case_prompt_text,
                candidate_solution=submitted_solution,
                model_name="gemini-2.5-flash",
            )
            case_pct = round((rubric["total_points"] / rubric["max_points"]) * 100, 1)
            composite_pct = round((mcq_pct * 0.5) + (case_pct * 0.5), 1)
            passed = composite_pct >= pass_threshold

            attempt = MilestoneAttempt(
                user_id=user_id,
                candidate_email=candidate_email,
                track_slug=track_slug,
                milestone_id="final-exam",
                attempt_type="final_exam",
                score_pct=composite_pct,
                correct_count=mcq_correct,
                total_count=mcq_total,
                passed=passed,
                details_json=json.dumps({
                    "mcq_pct": mcq_pct,
                    "case_pct": case_pct,
                    "rubric": rubric,
                    "mcq_details": mcq_details,
                }),
            )
            db.session.add(attempt)
            db.session.commit()

            result = {
                "passed": passed,
                "pass_threshold": pass_threshold,
                "composite_pct": composite_pct,
                "mcq_correct": mcq_correct,
                "mcq_total": mcq_total,
                "mcq_pct": mcq_pct,
                "case_pct": case_pct,
                "rubric": rubric,
                "mcq_details": mcq_details,
            }
        except Exception as e:
            db.session.rollback()
            error_message = f"AI Examiner grading encountered a temporary issue: {str(e)}. Please try submitting again."

    return render_template(
        "final_exam.html",
        cert=cert,
        questions=questions,
        design_case=design_case,
        candidate_email=candidate_email,
        pass_threshold=pass_threshold,
        duration_minutes=duration_minutes,
        submitted_solution=submitted_solution,
        result=result,
        error_message=error_message,
    )
