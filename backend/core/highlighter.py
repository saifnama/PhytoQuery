import re
from typing import List, Dict, Any


class Highlighter:
    # Entity type to CSS class mapping
    COLOR_MAP = {
        "CHEMICAL": "ent-chemical",
        "BIOACTIVITY": "ent-bioactivity",
        "LOCATION": "ent-location",
        "SPECIES": "ent-species",
        "PLANT PART": "ent-plant-part",
        "EXTRACTION METHOD": "ent-extraction-method",
        "DEVELOPMENT STAGE": "ent-development-stage",
        "SEASON": "ent-season",
        "ANALYTICAL TECHNIQUE": "ent-analytical-technique",
        "ISOLATION METHOD": "ent-isolation-method",
        "DRUG": "ent-drug",
        "DISEASE": "ent-disease",
    }

    @classmethod
    def highlight(cls, html_content: str, entities: List[Dict[str, Any]]) -> str:
        """
        Highlight entities in HTML text safely by only touching text nodes.
        Uses BeautifulSoup to parse the HTML and avoid breaking tags.
        """
        if not html_content or not entities:
            return html_content

        try:
            from bs4 import BeautifulSoup, NavigableString

            soup = BeautifulSoup(html_content, "html.parser")

            # 1. Create a sorted list of unique entities to highlight (longest first to avoid nested mismatch)
            unique_entities = sorted(
                {(e["text"].lower(), e["label"]) for e in entities if e.get("text")},
                key=lambda x: len(x[0]),
                reverse=True,
            )

            # 2. Iterate through all text nodes in the soup
            for text_node in soup.find_all(string=True):
                if not isinstance(text_node, NavigableString) or not text_node.strip():
                    continue

                # Check if parent is a protected tag (like <code>, <script>, or already a highlight)
                if text_node.parent.name in ("code", "script", "style", "span"):
                    # If it's a span, check if it's already one of our highlights
                    p_classes = text_node.parent.get("class", [])
                    if any(c.startswith("ent-") for c in p_classes):
                        continue

                original_node_text = str(text_node)
                current_node_text = original_node_text

                has_changes = False
                for ent_text, label in unique_entities:
                    if ent_text not in current_node_text.lower():
                        continue

                    css_class = cls.COLOR_MAP.get(label, "bg-gray-200")
                    # Replace occurrences while preserving case and escaping for regex
                    # Note: Using a wrapper function for replacement to avoid re-highlighting
                    pattern = re.compile(rf"\b({re.escape(ent_text)})\b", re.IGNORECASE)

                    # We need to replace in a way that doesn't mess with HTML we might insert
                    # Since we are inside a NavigableString, we can't just insert tags as strings easily
                    # without BeautifulSoup escaping them.
                    # Instead, we will perform a string replacement with a unique placeholder,
                    # then later turn those into soup objects.

                # A simpler approach: For each text node, perform regex highlighting
                # but use a marker that won't be escaped.
                # Actually, the safest way is to rebuild the node content as a sequence of strings and tags.

                new_content = []
                last_end = 0

                # We'll use a single complex regex for all entities to avoid overlapping issues
                if not unique_entities:
                    continue

                full_pattern_str = "|".join(
                    [rf"\b{re.escape(e[0])}\b" for e in unique_entities]
                )
                full_pattern = re.compile(f"({full_pattern_str})", re.IGNORECASE)

                # Use finditer to locate all matches in this specific text node
                matches = list(full_pattern.finditer(original_node_text))
                if not matches:
                    continue

                # Build a list of elements (strings and new tags)
                for m in matches:
                    start, end = m.start(), m.end()
                    found_text = m.group(0)

                    # Look up the label for this specific text
                    match_label = "ENTITY"
                    match_entity = None
                    for t, l in unique_entities:
                        if t == found_text.lower():
                            match_label = l
                            break

                    # Find the full entity with metadata
                    for e in entities:
                        if e.get("text", "").lower() == found_text.lower() and e.get("label") == match_label:
                            match_entity = e
                            break

                    css_class = cls.COLOR_MAP.get(match_label, "bg-gray-200")

                    # Add the leading un-highlighted text
                    if start > last_end:
                        new_content.append(
                            NavigableString(original_node_text[last_end:start])
                        )

                    # Create the span tag
                    span = soup.new_tag("span")
                    span["class"] = (
                        f"{css_class} rounded-sm cursor-help transition-all hover:brightness-95"
                    )
                    span["title"] = match_label
                    
                    # Add species metadata as data attributes if available
                    if match_label == "SPECIES" and match_entity:
                        if match_entity.get("accepted_scientific_name"):
                            span["data-accepted-scientific-name"] = match_entity["accepted_scientific_name"]
                        if match_entity.get("scientific_name_verified"):
                            span["data-scientific-name-verified"] = match_entity["scientific_name_verified"]
                        if match_entity.get("common_name"):
                            span["data-common-name"] = match_entity["common_name"]
                        if match_entity.get("name_type"):
                            span["data-name-type"] = match_entity["name_type"]
                        if match_entity.get("taxon_id"):
                            span["data-taxon-id"] = str(match_entity["taxon_id"])
                        if match_entity.get("source_db"):
                            span["data-source-db"] = match_entity["source_db"]
                        if match_entity.get("source_url"):
                            span["data-source-url"] = match_entity["source_url"]
                    
                    span.string = found_text
                    new_content.append(span)

                    last_end = end

                # Add the trailing un-highlighted text
                if last_end < len(original_node_text):
                    new_content.append(NavigableString(original_node_text[last_end:]))

                # Replace the original text node with the new sequence of nodes
                text_node.replace_with(*new_content)

            return str(soup)
        except Exception as e:
            # Fallback to original text if something goes wrong with soup parsing
            print(f"Highlighter error: {e}")
            return html_content
