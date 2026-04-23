import DOMPurify from 'dompurify';

// Client-side HTML sanitizer using a strict allowlist mirroring server-side NH3 rules.
// This function returns sanitized HTML string ready to inject via dangerouslySetInnerHTML.
export function sanitizeHtml(html: string): string {
  if (typeof window === 'undefined' || !html) return html ?? '';
  
  // First decode HTML entities (e.g. &lt; becomes <)
  let decoded = html;
  if (typeof document !== 'undefined') {
    const textarea = document.createElement('textarea');
    textarea.innerHTML = html;
    decoded = textarea.value;
  }
  
  // Define allowlist matching backend's sanitizer.py
  const ALLOWED_TAGS = [
    'p','h2','h3','table','tr','td','th','thead','tbody','caption',
    'figure','figcaption','img','a','span','cite','em','strong','sub','sup',
    'br','div','code','blockquote','ul','ol','li','section'
  ];
  // Allow server-side attributes; allow data-rid and data-entity across all tags
  const PURIFY_CONFIG: any = {
    ALLOWED_TAGS,
    ADD_ATTR: [
      'data-rid',
      'data-entity',
      // Species metadata attributes from highlighter
      'data-accepted-scientific-name',
      'data-scientific-name-verified',
      'data-common-name',
      'data-name-type',
      'data-taxon-id',
      'data-source-db',
      'data-source-url',
    ],
  };
  try {
    // dompurify expects to be called as DOMPurify.sanitize(html, config)
    // Cast to any to avoid strict TS issues with the library types in this project.
    return (DOMPurify as any).sanitize(decoded, PURIFY_CONFIG);
  } catch {
    // In case of any sanitization failure, return empty string to avoid injecting unsafe content
    return '';
  }
}

// Format text preserving italic/bold for display in titles and summaries.
// Universal: handles HTML entities (e.g. &lt;i&gt;) and actual tags.
// Uses browser parser to handle any edge case.
export function formatTextWithFormatting(text: string | null | undefined): string {
  if (!text) return '';
  if (typeof document === 'undefined') return text; // SSR guard

  try {
    // Step 1: Decode HTML entities to actual tags
    // e.g. "&lt;i&gt;" becomes "<i>"
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    const decoded = textarea.value;

    // Step 2: Parse as HTML
    const parser = new DOMParser();
    const doc = parser.parseFromString(`<div>${decoded}</div>`, 'text/html');
    const div = doc.querySelector('div');
    if (!div) return text;

    // Step 3: Remove dangerous elements
    const dangerous = div.querySelectorAll('script, iframe, object, embed, link, style, form, input, button, textarea, select');
    dangerous.forEach(el => el.remove());

    // Step 4: Serialize back - browser handles all edge cases
    let result = div.innerHTML;

    // Clean up empty formatting tags
    result = result.replace(/<(i|b|strong|em|sub|sup)>\s*<\/\1>/gi, '');

    // Clean up orphan whitespace
    result = result.replace(/\s+/g, ' ').trim();

    return result;
  } catch {
    // Fallback: decode entities then strip dangerous tags
    const textarea = document.createElement('textarea');
    textarea.innerHTML = text;
    return textarea.value
      .replace(/<script[^>]*>.*?<\/script>/gi, '')
      .replace(/<[^>]+>/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  }
}