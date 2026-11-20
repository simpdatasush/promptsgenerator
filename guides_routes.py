from flask import Blueprint, render_template, abort, request
from guides_engine import (
    load_all_topics,
    load_certifications,
    get_topic_with_guides,
    get_single_guide,
)

guides_bp = Blueprint("guides", __name__, url_prefix="/guides")


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
