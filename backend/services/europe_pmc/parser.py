"""Europe PMC XML parser and JATS-to-HTML converter.

Handles all XML parsing and HTML conversion:
- JATS inline tag conversion to HTML
- XML namespace normalization
- TOC extraction from HTML
- Section/reference extraction from JATS XML
- Title extraction from XML
"""

import re
import logging
from typing import Optional, Tuple, Dict, Any, List
from lxml import etree as ET
from bs4 import BeautifulSoup

from backend.core.sanitizer import sanitize

logger = logging.getLogger(__name__)


class JATSConverter:
    """Converts JATS XML tags to semantic HTML."""

    @staticmethod
    def inline_to_html(text: str) -> str:
        """Convert JATS inline formatting tags in raw text to HTML equivalents.

        Handles common JATS tags found in abstracts and metadata from Europe PMC:
        <italic> -> <em>, <bold> -> <strong>, <sup>/<sub> preserved, etc.

        Uses BeautifulSoup for robust parsing of potentially malformed markup.
        """
        if not text:
            return ""

        soup = BeautifulSoup(text, "html.parser")

        # Simple tag renames: JATS -> HTML
        TAG_MAP = {
            "italic": "em",
            "bold": "strong",
            "sup": "sup",
            "sub": "sub",
            "monospace": "code",
        }
        for jats_tag, html_tag in TAG_MAP.items():
            for tag in soup.find_all(jats_tag):
                tag.name = html_tag

        # <sc> (small caps) -> <span class="small-caps">
        for tag in soup.find_all("sc"):
            tag.name = "span"
            tag.attrs = {"class": "small-caps"}

        # <ext-link xlink:href="..."> -> <a href="..." target="_blank">
        for tag in soup.find_all("ext-link"):
            tag.name = "a"
            href = tag.get("xlink:href", "") or tag.get(
                "{http://www.w3.org/1999/xlink}href", ""
            )
            tag.attrs = (
                {"href": href, "target": "_blank", "rel": "noopener"} if href else {}
            )

        # <email> -> <a href="mailto:...">
        for tag in soup.find_all("email"):
            addr = tag.get_text(strip=True)
            tag.name = "a"
            tag.attrs = {"href": f"mailto:{addr}"}
            tag.string = addr

        # Clean <p> attributes (nh3 allows <p> but strip extra attrs)
        for tag in soup.find_all("p"):
            tag.attrs = {}

        return str(soup)

    @staticmethod
    def ensure_xmlns(xml_content: str) -> str:
        """Ensure xmlns:xlink is declared at root level for lxml parsing.

        Europe PMC XML often declares xmlns:xlink on child elements (e.g. <self-uri>)
        but lxml requires it at the root <article> level.
        """
        if (
            "xlink:" in xml_content
            and "xmlns:xlink" not in xml_content.split(">", 1)[0]
        ):
            # Inject xmlns:xlink into the root element's opening tag
            return xml_content.replace(
                "<article ",
                '<article xmlns:xlink="http://www.w3.org/1999/xlink" ',
                1,
            )
        return xml_content

    @staticmethod
    def clean_xml(xml_content: str) -> str:
        """Strip XML tags and return clean text."""
        try:
            root = ET.fromstring(
                JATSConverter.ensure_xmlns(xml_content).encode("utf-8")
            )
            body = root.find(".//body")
            if body is not None:
                return "".join(body.itertext()).strip()
            return "".join(root.itertext()).strip()
        except Exception:
            clean = re.sub(r"<[^>]+>", " ", xml_content)
            return re.sub(r"\s+", " ", clean).strip()


class XMLParser:
    """Parses Europe PMC JATS XML into structured sections and references."""

    @staticmethod
    def extract_title_from_xml(xml_content: str) -> str:
        """Extract paper title from PMC fullTextXML."""
        try:
            root = ET.fromstring(
                JATSConverter.ensure_xmlns(xml_content).encode("utf-8")
            )
            title_node = root.find(".//article-title")
            if title_node is not None:
                return "".join(title_node.itertext()).strip()
        except Exception:
            pass
        return ""

    @staticmethod
    def extract_metadata_from_xml(xml_content: str) -> dict:
        """Extract title, authors, journal, and date from PMC fullTextXML in a SINGLE parse pass."""
        result = {"title": "", "authors": [], "journal": "", "date": ""}
        try:
            root = ET.fromstring(
                JATSConverter.ensure_xmlns(xml_content).encode("utf-8")
            )

            # Title
            title_node = root.find(".//article-title")
            if title_node is not None:
                result["title"] = "".join(title_node.itertext()).strip()

            # Authors
            authors = []
            for contrib in root.findall(
                ".//contrib-group/contrib[@contrib-type='author']"
            ):
                surname = contrib.findtext("name/surname", "").strip()
                given = contrib.findtext("name/given-names", "").strip()
                if surname:
                    authors.append(f"{given} {surname}".strip())
            if not authors:
                for contrib in root.findall(".//contrib-group/contrib"):
                    surname = contrib.findtext("name/surname", "").strip()
                    given = contrib.findtext("name/given-names", "").strip()
                    if surname:
                        authors.append(f"{given} {surname}".strip())
            result["authors"] = authors

            # Journal
            journal_title = root.findtext(".//journal-title", "").strip()
            if not journal_title:
                journal_title = root.findtext(".//abbrev-journal-title", "").strip()
            result["journal"] = journal_title

            # Date
            for date_path in [
                ".//pub-date[@date-type='pub']/year",
                ".//pub-date/year",
                ".//article-meta/pub-date/year",
                ".//pub-date[@date-type='epub']/year",
                ".//pub-date[@date-type='collection']/year",
            ]:
                year_node = root.find(date_path)
                if year_node is not None and year_node.text:
                    year = year_node.text.strip()
                    parent = year_node.getparent()
                    if parent is not None:
                        month = parent.findtext("month", "").strip()
                        day = parent.findtext("day", "").strip()
                        if month and day:
                            from calendar import month_name

                            try:
                                month_str = month_name[int(month)]
                            except (ValueError, IndexError):
                                month_str = month
                            result["date"] = f"{day} {month_str} {year}"
                            break
                    result["date"] = year
                    break
        except Exception:
            pass
        return result

    @staticmethod
    def extract_authors_from_xml(xml_content: str) -> list:
        """Extract author names from PMC fullTextXML."""
        try:
            root = ET.fromstring(
                JATSConverter.ensure_xmlns(xml_content).encode("utf-8")
            )
            authors = []
            for contrib in root.findall(
                ".//contrib-group/contrib[@contrib-type='author']"
            ):
                surname = contrib.findtext("name/surname", "").strip()
                given = contrib.findtext("name/given-names", "").strip()
                if surname:
                    authors.append(f"{given} {surname}".strip())
            # Fallback: try any contrib without contrib-type
            if not authors:
                for contrib in root.findall(".//contrib-group/contrib"):
                    surname = contrib.findtext("name/surname", "").strip()
                    given = contrib.findtext("name/given-names", "").strip()
                    if surname:
                        authors.append(f"{given} {surname}".strip())
            return authors
        except Exception:
            return []

    @staticmethod
    def extract_journal_from_xml(xml_content: str) -> str:
        """Extract journal title from PMC fullTextXML."""
        try:
            root = ET.fromstring(
                JATSConverter.ensure_xmlns(xml_content).encode("utf-8")
            )
            journal_title = root.findtext(".//journal-title", "").strip()
            if journal_title:
                return journal_title
            # Fallback: abbreviated journal title
            return root.findtext(".//abbrev-journal-title", "").strip()
        except Exception:
            return ""

    @staticmethod
    def extract_date_from_xml(xml_content: str) -> str:
        """Extract publication date from PMC fullTextXML."""
        try:
            root = ET.fromstring(
                JATSConverter.ensure_xmlns(xml_content).encode("utf-8")
            )
            # Try various pub-date locations
            for date_path in [
                ".//pub-date[@date-type='pub']/year",
                ".//pub-date/year",
                ".//article-meta/pub-date/year",
                ".//pub-date[@date-type='epub']/year",
                ".//pub-date[@date-type='collection']/year",
            ]:
                year_node = root.find(date_path)
                if year_node is not None and year_node.text:
                    year = year_node.text.strip()
                    # Also get month and day if available
                    parent = year_node.getparent()
                    if parent is not None:
                        month = parent.findtext("month", "").strip()
                        day = parent.findtext("day", "").strip()
                        if month and day:
                            from calendar import month_name

                            try:
                                month_str = month_name[int(month)]
                            except (ValueError, IndexError):
                                month_str = month
                            return f"{day} {month_str} {year}"
                    return year
            return ""
        except Exception:
            return ""

    @staticmethod
    def extract_toc_from_html(html_content: str) -> List[Dict[str, Any]]:
        """Extract a two-level Table of Contents from semantic HTML.

        Strategy:
        - Find all h2 headings and record their id, text, and level (2).
        - For each h2, collect any immediately following h3 headings (until the next h2)
          as children with their own id, text, and level (3).
        - Do not include headings without IDs (per requirement: IDs must match actual HTML IDs).
        - Return a nested list: [{ id, title, level, children: [...] }, ...]
        """
        if not html_content:
            return []

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            toc: List[Dict[str, Any]] = []

            for h2 in soup.find_all("h2"):
                h2_id = h2.get("id", "")
                h2_text = h2.get_text(strip=True)
                if not h2_id and not h2_text:
                    continue
                item: Dict[str, Any] = {
                    "id": h2_id,
                    "title": h2_text,
                    "level": 2,
                    "children": [],
                }

                # Collect following h3 siblings until the next h2
                sibling = h2.find_next_sibling()
                while sibling is not None and sibling.name != "h2":
                    if sibling.name == "h3":
                        sid = sibling.get("id", "")
                        stx = sibling.get_text(strip=True)
                        if sid or stx:
                            item["children"].append(
                                {"id": sid, "title": stx, "level": 3}
                            )
                    sibling = sibling.find_next_sibling()

                toc.append(item)

            # Fallback: if there are headings but no h2, attach any h3s as top-level items
            if not toc:
                for h3 in soup.find_all("h3"):
                    sid = h3.get("id", "")
                    stx = h3.get_text(strip=True)
                    if sid or stx:
                        toc.append(
                            {"id": sid, "title": stx, "level": 3, "children": []}
                        )

            return toc
        except Exception:
            return []

    @staticmethod
    def parse_sections_from_xml(
        xml_content: str, pmcid: str = ""
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Extract sections and references from PMC XML using a high-fidelity approach."""
        try:
            root = ET.fromstring(
                JATSConverter.ensure_xmlns(xml_content).encode("utf-8")
            )
            sections = []
            references = {}

            # 0. Extract Bibliography (References) - Structured parsing
            for ref in root.findall(".//ref-list/ref"):
                ref_id = ref.get("id")
                if not ref_id:
                    continue

                ref_data = {"id": ref_id}

                # Extract citation element (contains structured reference data)
                citation = (
                    ref.find(".//citation")
                    or ref.find(".//element-citation")
                    or ref.find(".//mixed-citation")
                )
                if citation is not None:
                    # Authors
                    authors = []
                    person_group = citation.find(".//person-group")
                    if person_group is not None:
                        for name in person_group.findall(".//name"):
                            surname = name.findtext("surname", "")
                            given_names = name.findtext("given-names", "")
                            if surname:
                                authors.append(f"{surname} {given_names}".strip())
                    ref_data["authors"] = authors

                    # Article title
                    article_title = citation.findtext(".//article-title", "")
                    ref_data["title"] = article_title.strip() if article_title else ""

                    # Journal/Source
                    source = citation.findtext(".//source", "")
                    ref_data["journal"] = source.strip() if source else ""

                    # Year
                    year = citation.findtext(".//year", "")
                    ref_data["year"] = year.strip() if year else ""

                    # Volume, Issue, Pages
                    ref_data["volume"] = citation.findtext(".//volume", "").strip()
                    ref_data["issue"] = citation.findtext(".//issue", "").strip()
                    ref_data["fpage"] = citation.findtext(".//fpage", "").strip()
                    ref_data["lpage"] = citation.findtext(".//lpage", "").strip()

                    # DOI
                    doi_elem = citation.find(".//pub-id[@pub-id-type='doi']")
                    ref_data["doi"] = (
                        doi_elem.text.strip()
                        if doi_elem is not None and doi_elem.text
                        else ""
                    )

                    # PMID
                    pmid_elem = citation.find(".//pub-id[@pub-id-type='pmid']")
                    ref_data["pmid"] = (
                        pmid_elem.text.strip()
                        if pmid_elem is not None and pmid_elem.text
                        else ""
                    )
                else:
                    # Fallback: extract as plain text
                    ref_text = "".join(ref.itertext()).strip()
                    ref_text = re.sub(r"^\[?\d+\]?[\s.]+", "", ref_text)
                    ref_data["authors"] = []
                    ref_data["title"] = ref_text
                    ref_data["journal"] = ""
                    ref_data["year"] = ""
                    ref_data["volume"] = ""
                    ref_data["issue"] = ""
                    ref_data["fpage"] = ""
                    ref_data["lpage"] = ""
                    ref_data["doi"] = ""
                    ref_data["pmid"] = ""

                references[ref_id] = ref_data

            # Counter for generating unique heading IDs
            heading_counter = [0]

            def make_heading_id(text):
                """Generate a unique, URL-safe ID from heading text."""
                heading_counter[0] += 1
                slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50]
                return f"h-{heading_counter[0]}-{slug}"

            # Helper: recursively convert JATS XML to clean semantic HTML
            def get_html_recursive(node, is_root=False):
                parts = []

                # --- Block-level elements ---

                # CASE: Nested section title -> <h3>
                # (label is prepended by the parent <sec> handler)
                if not is_root and node.tag == "title":
                    title_text = "".join(node.itertext()).strip()
                    hid = make_heading_id(title_text)
                    return f'<h3 id="{hid}" class="article-h3">{title_text}</h3>'

                # CASE: Paragraph -> <p>
                if node.tag == "p":
                    inner = _collect_children_html(node)
                    return f"<p>{inner}</p>"

                # CASE: Citations (xref to bibliography)
                if node.tag == "xref" and node.get("ref-type") == "bibr":
                    rid = node.get("rid")
                    return f'<cite class="citation" data-rid="{rid}">{"".join(node.itertext())}</cite>'

                # CASE: Cross-references (figures, tables, etc.)
                if node.tag == "xref":
                    rid = node.get("rid", "")
                    return f'<a href="#{rid}" class="xref-link">{"".join(node.itertext())}</a>'

                # CASE: Table
                if node.tag == "table-wrap":
                    label = node.findtext("label") or "Table"
                    cap_node = node.find("caption")
                    caption = (
                        "".join(cap_node.itertext()) if cap_node is not None else ""
                    )
                    table_node = node.find(".//table")
                    if table_node is not None:
                        table_html = ET.tostring(
                            table_node, encoding="unicode", method="html"
                        )
                        cap_html = (
                            f'<p class="table-caption">{caption}</p>' if caption else ""
                        )
                        return (
                            f'<div class="article-table-wrap">'
                            f'<span class="table-label">{label}</span>'
                            f'<div class="table-scroll-container jats-table">{table_html}</div>'
                            f"{cap_html}"
                            f"</div>"
                        )
                    return ""

                # CASE: Figure
                if node.tag == "fig":
                    label = node.findtext("label") or "Figure"
                    cap_node = node.find("caption")
                    caption = (
                        "".join(cap_node.itertext()) if cap_node is not None else ""
                    )
                    graphic = node.find(".//graphic")
                    if graphic is not None:
                        href = graphic.get("{http://www.w3.org/1999/xlink}href")
                        img_url = (
                            f"https://europepmc.org/articles/{pmcid}/bin/{href}"
                            if pmcid
                            else ""
                        )
                        if img_url:
                            cap_html = (
                                f'<figcaption><span class="fig-label">{label}</span> {caption}</figcaption>'
                                if caption
                                else f'<figcaption><span class="fig-label">{label}</span></figcaption>'
                            )
                            return (
                                f'<figure class="article-figure">'
                                f'<img src="{img_url}" alt="{label}" loading="lazy" />'
                                f"{cap_html}"
                                f"</figure>"
                            )
                    return ""

                # CASE: Lists - preserve EXACTLY what Europe PMC provides
                if node.tag == "list":
                    list_type = node.get("list-type", "")
                    # Check if list-items carry their own <label> elements
                    first_li = node.find("list-item")
                    has_labels = (
                        first_li is not None and first_li.find("label") is not None
                    )

                    if has_labels:
                        # Items have explicit labels (verbatim)
                        items = []
                        for li_node in node.findall("list-item"):
                            label_el = li_node.find("label")
                            lbl = (
                                "".join(label_el.itertext()).strip()
                                if label_el is not None
                                else ""
                            )
                            inner_parts = []
                            for child in li_node:
                                if child.tag == "label":
                                    if child.tail:
                                        inner_parts.append(child.tail)
                                    continue
                                inner_parts.append(
                                    get_html_recursive(child, is_root=False)
                                )
                                if child.tail:
                                    inner_parts.append(child.tail)
                            li_inner = "".join(inner_parts)
                            items.append(
                                f'<li style="list-style:none !important"><span class="list-label">{lbl}</span> {li_inner}</li>'
                            )
                        return (
                            '<ol style="list-style:none !important;padding-left:1.5em">'
                            + "".join(items)
                            + "</ol>"
                        )
                    else:
                        # No explicit labels - use semantic HTML ol type
                        OL_TYPE_MAP = {
                            "order": ("1", "decimal"),
                            "ordered": ("1", "decimal"),
                            "alpha-lower": ("a", "lower-alpha"),
                            "alpha-upper": ("A", "upper-alpha"),
                            "roman-lower": ("i", "lower-roman"),
                            "roman-upper": ("I", "upper-roman"),
                            "bullet": ("", "disc"),
                        }
                        mapped_info = OL_TYPE_MAP.get(list_type, ("", "disc"))
                        mapped_type, css_style = mapped_info

                        if mapped_type:
                            tag_open = f'<ol type="{mapped_type}" style="list-style-type: {css_style} !important;">'
                            tag_close = "</ol>"
                        else:
                            tag_open = (
                                f'<ul style="list-style-type: {css_style} !important;">'
                            )
                            tag_close = "</ul>"

                        items = []
                        for li_node in node.findall("list-item"):
                            li_inner = "".join(
                                get_html_recursive(child, is_root=False)
                                for child in li_node
                            )
                            items.append(f"<li>{li_inner}</li>")
                        return tag_open + "".join(items) + tag_close

                if node.tag == "list-item":
                    inner = "".join(
                        get_html_recursive(child, is_root=False) for child in node
                    )
                    return f"<li>{inner}</li>"

                # CASE: Display quotes
                if node.tag == "disp-quote":
                    inner = _collect_children_html(node)
                    return f"<blockquote>{inner}</blockquote>"

                # --- Inline-level elements ---

                INLINE_FORMATTING = {
                    "italic": ("em", ""),
                    "bold": ("strong", ""),
                    "sub": ("sub", ""),
                    "sup": ("sup", ""),
                    "sc": ("span", ' class="small-caps"'),
                    "monospace": ("code", ""),
                    "underline": ("span", ' class="underline"'),
                }
                if node.tag in INLINE_FORMATTING:
                    html_tag, extra_attrs = INLINE_FORMATTING[node.tag]
                    inner = _collect_children_html(node)
                    return f"<{html_tag}{extra_attrs}>{inner}</{html_tag}>"

                # CASE: External links
                if node.tag == "ext-link":
                    href = node.get("{http://www.w3.org/1999/xlink}href") or ""
                    inner = _collect_children_html(node)
                    if href:
                        return f'<a href="{href}" target="_blank" rel="noopener">{inner}</a>'
                    return inner

                # CASE: Email
                if node.tag == "email":
                    addr = (node.text or "").strip()
                    return f'<a href="mailto:{addr}">{addr}</a>'

                # CASE: Named content (e.g. genus/species)
                if node.tag == "named-content":
                    return _collect_children_html(node)

                # --- Nested section ---
                if node.tag == "sec":
                    inner_parts = []
                    # Grab label for sub-section numbering (preserves original from Europe PMC)
                    label_node = node.find("label")
                    label_text = (
                        "".join(label_node.itertext()).strip()
                        if label_node is not None
                        else ""
                    )
                    title_node = node.find("title")
                    title_text = (
                        "".join(title_node.itertext()).strip()
                        if title_node is not None
                        else ""
                    )

                    # Detect duplicate: skip this h3 if parent <sec> has
                    # the exact same label+title (normalized)
                    parent = node.getparent()
                    is_duplicate = False
                    if parent is not None and parent.tag == "sec":
                        p_label = parent.find("label")
                        p_title = parent.find("title")

                        def normalize(t):
                            if not t:
                                return ""
                            return re.sub(r"[^a-z0-9]+", "", t.lower())

                        p_label_text = (
                            "".join(p_label.itertext()).strip()
                            if p_label is not None
                            else ""
                        )
                        p_title_text = (
                            "".join(p_title.itertext()).strip()
                            if p_title is not None
                            else ""
                        )

                        if normalize(label_text) == normalize(
                            p_label_text
                        ) and normalize(title_text) == normalize(p_title_text):
                            is_duplicate = True

                    for child in node:
                        if child.tag == "label" or child.tag == "title":
                            if is_root:
                                # Skip the top-level title and label (already rendered as H2)
                                if child.tail:
                                    inner_parts.append(child.tail)
                                continue
                            if child.tag == "label":
                                continue  # Already captured in label_node
                            if child.tag == "title":
                                if is_duplicate:
                                    continue  # Redundant structural header
                                if label_text:
                                    full_title = f"{label_text} {title_text}"
                                else:
                                    full_title = title_text
                                hid = make_heading_id(full_title)
                                inner_parts.append(
                                    f'<h3 id="{hid}" class="article-h3">{full_title}</h3>'
                                )
                                if child.tail:
                                    inner_parts.append(child.tail)
                                continue

                        inner_parts.append(get_html_recursive(child, is_root=False))
                        if child.tail:
                            inner_parts.append(child.tail)
                    return "".join(inner_parts)

                # --- Default/fallback: collect children ---
                return _collect_children_html(node, is_root=is_root)

            def _collect_children_html(node, is_root=False):
                """Collect node.text + children HTML + tails into a single string."""
                parts = []
                if node.text:
                    parts.append(node.text)
                for child in node:
                    if is_root and child.tag in ("title", "label"):
                        # Skip the top-level title and label (rendered separately as the h2 heading)
                        if child.tail:
                            parts.append(child.tail)
                        continue
                    parts.append(get_html_recursive(child, is_root=False))
                    if child.tail:
                        parts.append(child.tail)
                return "".join(parts)

            def _extract_headings_from_html(html_content):
                """Find all <h3 id="..."> tags AND [H3]...[/H3] markers in the HTML and return heading metadata."""
                headings = []
                # Find real <h3> tags generated from JATS <title>
                for m in re.finditer(
                    r'<h3[^>]*id="([^"]*)"[^>]*class="[^"]*article-h3[^"]*"[^>]*>(.*?)</h3>',
                    html_content,
                    re.S,
                ):
                    headings.append({"id": m.group(1), "text": m.group(2).strip()})
                # Find [H3]...[/H3] markers preserved in content
                counter = [0]
                for m in re.finditer(
                    r"\[H3\](.*?)\[/H3\]",
                    html_content,
                    re.S,
                ):
                    counter[0] += 1
                    text = m.group(1).strip()
                    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())[:50]
                    headings.append({"id": slug, "text": text})
                return headings

            # 1. Extract Abstract
            abstract_node = root.find(".//abstract")
            if abstract_node is not None:
                abs_html = get_html_recursive(abstract_node, is_root=True).strip()
                if abs_html:
                    headings = _extract_headings_from_html(abs_html)
                    sections.append(
                        {
                            "title": "Abstract",
                            "content": abs_html,
                            "headings": headings,
                        }
                    )

            # 2. Extract Body sections
            body = root.find(".//body")
            if body is not None:
                for sec in body.findall("./sec"):
                    # Read both <label> and <title> to preserve original numbering
                    label_node = sec.find("label")
                    title_node = sec.find("title")
                    label_text = (
                        label_node.text.strip()
                        if label_node is not None and label_node.text
                        else ""
                    )
                    title = "Section"
                    if title_node is not None:
                        title = "".join(title_node.itertext()).strip()
                    if label_text:
                        title = f"{label_text} {title}"
                    content = get_html_recursive(sec, is_root=True).strip()
                    if content:
                        headings = _extract_headings_from_html(content)
                        sections.append(
                            {"title": title, "content": content, "headings": headings}
                        )

            # 3. Extract Back sections (excluding ref-list)
            back = root.find(".//back")
            if back is not None:
                for sec in back.findall("./sec"):
                    if sec.find(".//ref-list") is not None:
                        continue
                    label_node = sec.find("label")
                    title_node = sec.find("title")
                    label_text = (
                        label_node.text.strip()
                        if label_node is not None and label_node.text
                        else ""
                    )
                    title = "Section"
                    if title_node is not None:
                        title = "".join(title_node.itertext()).strip()
                    if label_text:
                        title = f"{label_text} {title}"
                    content = get_html_recursive(sec, is_root=True).strip()
                    if content:
                        headings = _extract_headings_from_html(content)
                        sections.append(
                            {"title": title, "content": content, "headings": headings}
                        )

            return sections, references
        except Exception as e:
            logger.error(f"Error parsing XML sections: {e}")
            return [], {}
