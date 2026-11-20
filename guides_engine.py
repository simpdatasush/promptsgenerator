import os
import yaml
import markdown

CONTENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "content", "guides")


def _parse_markdown_file(filepath):
    """Splits YAML frontmatter and Markdown body, returning metadata + rendered HTML + TOC."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    meta = {}
    body = raw

    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()

    md = markdown.Markdown(
        extensions=["fenced_code", "tables", "codehilite", "toc"],
        extension_configs={
            "codehilite": {"css_class": "highlight", "guess_lang": False},
            "toc": {"permalink": False, "toc_depth": "2-3"},
        },
    )
    html_content = md.convert(body)
    toc_html = md.toc

    return {
        "meta": meta,
        "html": html_content,
        "toc": toc_html,
    }


def load_all_topics():
    """Loads _topics.yaml and counts available Markdown guides per topic."""
    topics_path = os.path.join(CONTENT_DIR, "_topics.yaml")
    if not os.path.exists(topics_path):
        return []

    with open(topics_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    topics_dict = data.get("topics", {})
    topics_list = []

    for slug, info in topics_dict.items():
        topic_dir = os.path.join(CONTENT_DIR, slug)
        actual_files = []
        if os.path.isdir(topic_dir):
            actual_files = [fn for fn in os.listdir(topic_dir) if fn.endswith(".md")]

        item = dict(info)
        item["slug"] = slug
        item["published_count"] = len(actual_files)
        topics_list.append(item)

    topics_list.sort(key=lambda x: x.get("order", 99))
    return topics_list


def load_certifications():
    """Loads V1 certification tracks from _certifications.yaml."""
    cert_path = os.path.join(CONTENT_DIR, "_certifications.yaml")
    if not os.path.exists(cert_path):
        return []

    with open(cert_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    certs = data.get("certifications", {})
    return list(certs.values())


def get_topic_with_guides(topic_slug):
    """Returns topic metadata + ordered list of published guides inside that topic."""
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
    """Fetches a specific rendered guide and sibling guides in the same topic."""
    topic, sibling_guides = get_topic_with_guides(topic_slug)
    if not topic:
        return None, None, []

    filepath = os.path.join(CONTENT_DIR, topic_slug, f"{guide_slug}.md")
    if not os.path.exists(filepath):
        return topic, None, sibling_guides

    parsed = _parse_markdown_file(filepath)
    return topic, parsed, sibling_guides
