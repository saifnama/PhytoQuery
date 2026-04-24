import re

try:
    import nh3  # type: ignore

    _NH3_AVAILABLE = True
except Exception:
    nh3 = None  # type: ignore
    _NH3_AVAILABLE = False


def sanitize(html_content: str) -> str:
    """Sanitize HTML content using nh3 with a strict allowlist.

    Allowed tags:
      p, h3, table, tr, td, th, thead, tbody, figure, img, a, span, cite, em, strong, sub, sup, br, div

    Allowed attributes per tag:
      cite: data-rid
      a: href
      img: src, alt, title, width, height
      td: colspan, rowspan
      th: colspan, rowspan, scope
    """
    if html_content is None:
        return ""
    text = html_content.strip()
    if not text:
        return ""

    allowed_tags = {
        "p",
        "h2",
        "h3",
        "table",
        "tr",
        "td",
        "th",
        "thead",
        "tbody",
        "caption",
        "figure",
        "figcaption",
        "img",
        "a",
        "span",
        "cite",
        "em",
        "strong",
        "sub",
        "sup",
        "br",
        "div",
        "code",
        "blockquote",
        "ul",
        "ol",
        "li",
        "section",
    }
    allowed_attributes = {
        "cite": {"data-rid"},
        "a": {"href", "target", "rel", "class"},
        "img": {"src", "alt", "title", "width", "height", "loading"},
        "td": {"colspan", "rowspan"},
        "th": {"colspan", "rowspan", "scope"},
        "span": {"class"},
        "h3": {"id", "class"},
        "h2": {"id", "class"},
        "div": {"class"},
        "figure": {"class"},
        "ol": {"type", "style"},
        "li": {"style"},
    }

    if _NH3_AVAILABLE and nh3 is not None:
        cleaned = nh3.clean(
            text,
            tags=allowed_tags,
            attributes=allowed_attributes,
            link_rel=None,
        )
    else:
        # Lightweight fallback sanitization (only used if nh3 is unavailable).
        # Removes script tags and strips javascript hrefs.
        s = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S | re.I)
        s = re.sub(r'href=["\']javascript:[^"\']+["\']', 'href=""', s, flags=re.I)
        cleaned = s

    def _normalize_anchor_rel(match: re.Match[str]) -> str:
        tag = match.group(0)
        target_match = re.search(r'target=("|\')([^"\']*)(\1)', tag, flags=re.I)
        if not target_match or target_match.group(2).lower() != "_blank":
            return tag

        rel_tokens = ["noopener", "noreferrer"]
        rel_match = re.search(r'rel=("|\')([^"\']*)(\1)', tag, flags=re.I)
        if rel_match:
            existing_tokens = rel_match.group(2).split()
            merged_tokens = []
            for token in [*existing_tokens, *rel_tokens]:
                normalized = token.strip().lower()
                if normalized and normalized not in merged_tokens:
                    merged_tokens.append(normalized)
            replacement = f'rel="{" ".join(merged_tokens)}"'
            return re.sub(r'rel=("|\')[^"\']*(\1)', replacement, tag, flags=re.I)

        insert_at = target_match.end()
        return f"{tag[:insert_at]} rel=\"noopener noreferrer\"{tag[insert_at:]}"

    cleaned = re.sub(r"<a\b[^>]*>", _normalize_anchor_rel, cleaned, flags=re.I)
    # Ensure we return a string (nh3 returns str). If nothing remains, return empty string.
    return cleaned or ""
