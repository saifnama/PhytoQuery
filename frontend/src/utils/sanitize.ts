import DOMPurify from 'dompurify';

// Client-side HTML sanitizer using a strict allowlist mirroring server-side NH3 rules.
// This function returns sanitized HTML string ready to inject via dangerouslySetInnerHTML.
export function sanitizeHtml(html: string): string {
  if (typeof window === 'undefined' || !html) return html ?? '';
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
    return (DOMPurify as any).sanitize(html, PURIFY_CONFIG);
  } catch {
    // In case of any sanitization failure, return empty string to avoid injecting unsafe content
    return '';
  }
}
