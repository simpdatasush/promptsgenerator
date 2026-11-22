from flask import Blueprint, render_template, abort, request, session
from flask_login import current_user
from guides_engine import (
    load_all_topics,
    load_certifications,
    get_certification_by_slug,
    get_topic_with_guides,
    get_single_guide,
    get_milestone_questions,
    grade_and_save_milestone,
    get_candidate_progress,
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
