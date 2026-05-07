/**
 * MarkdownPreviewPanel — side drawer that opens when the user clicks
 * a citation superscript in an assistant answer.
 *
 * Behavior:
 *   - Fetches ``/api/chat/files/<source>/markdown`` (utf-8 markdown
 *     extracted at ingest time; the backend lazy-regenerates it for
 *     papers uploaded before this feature shipped).
 *   - Locates the cited chunk's text inside the markdown and wraps
 *     it in ``<mark class="chunk-highlight">``. If the verbatim quote
 *     from Pass-2 citation extraction is also present inside that
 *     chunk, wraps just the quote in ``<mark class="quote-highlight">``
 *     (bright yellow, bold).
 *   - Scrolls the chunk-highlight into view on mount.
 *
 * No new dependencies — uses ``react-markdown`` already in the bundle.
 */

import { type FC, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import { X } from '@phosphor-icons/react';
import type { Citation, RagSource } from './assistant/runtime';

// react-markdown@10 ignores inline HTML in the source by default. We
// splice our highlights in as ``<mark>`` tags, so we need rehype-raw
// to walk the AST and turn those raw HTML nodes back into real
// elements that React renders. Without this, the literal text
// ``<mark class="...">`` appears in the rendered output (the bug
// the user reported).
const rehypePlugins = [rehypeRaw];

interface MarkdownPreviewPanelProps {
  source: RagSource;
  citation?: Citation;
  /** Bumped by the parent on every citation click. Used as React
   * ``key`` on the inner markdown container so the same click
   * (or repeat-click on the same source) restarts the CSS flash
   * animation by force-remounting the highlight elements. */
  triggerKey: number;
  onClose: () => void;
}

/** Find the best in-text occurrence of ``needle`` inside ``haystack``
 * after whitespace+case normalization. Returns the [start, end)
 * indices into the original haystack, or null if no good match
 * exists. Cascades through three strategies of increasing
 * looseness so reassembled / lightly-paraphrased chunks still
 * anchor on the right region: exact match → multi-slice match →
 * n-gram density cluster (the strongest fallback, picks the spot
 * where the most distinctive substrings of the needle co-occur). */
function findFlexibleSpan(
  haystack: string,
  needle: string,
): [number, number] | null {
  if (!needle || !haystack) return null;
  const normalizedNeedle = needle.replace(/\s+/g, ' ').trim();
  if (normalizedNeedle.length < 8) return null;

  // Build a parallel haystack where consecutive whitespace becomes a
  // single space, but track original indices.
  const indexMap: number[] = [];
  let normalized = '';
  let prevWasSpace = false;
  for (let i = 0; i < haystack.length; i += 1) {
    const ch = haystack[i];
    if (/\s/.test(ch)) {
      if (!prevWasSpace) {
        normalized += ' ';
        indexMap.push(i);
        prevWasSpace = true;
      }
    } else {
      normalized += ch;
      indexMap.push(i);
      prevWasSpace = false;
    }
  }

  const lowerNorm = normalized.toLowerCase();
  const lowerNeedle = normalizedNeedle.toLowerCase();

  // Strategy 1: exact substring match after whitespace collapse.
  const exact = lowerNorm.indexOf(lowerNeedle);
  if (exact !== -1) {
    const start = indexMap[exact];
    const endNormIdx = exact + lowerNeedle.length - 1;
    const end = indexMap[Math.min(endNormIdx, indexMap.length - 1)] + 1;
    return [start, end];
  }

  // Strategy 2: multi-offset slice match — pick the longest
  // matching slice and use its position. Catches contiguous-tail
  // matches when the leading text was paraphrased.
  const sliceLength = 60;
  const candidateOffsets = [
    0,
    Math.floor(lowerNeedle.length * 0.25),
    Math.floor(lowerNeedle.length * 0.5),
    Math.floor(lowerNeedle.length * 0.75),
    Math.max(0, lowerNeedle.length - sliceLength),
  ];
  let bestSliceStart = -1;
  let bestSliceOffset = -1;
  let bestSliceLen = 0;
  for (const offset of candidateOffsets) {
    const slice = lowerNeedle.slice(offset, offset + sliceLength);
    if (slice.length < 24) continue;
    const found = lowerNorm.indexOf(slice);
    if (found !== -1 && slice.length > bestSliceLen) {
      bestSliceStart = found;
      bestSliceOffset = offset;
      bestSliceLen = slice.length;
    }
  }
  if (bestSliceStart !== -1) {
    const startInOrig = indexMap[bestSliceStart];
    const expandedStart = Math.max(0, startInOrig - bestSliceOffset);
    const expandedEnd = Math.min(haystack.length, expandedStart + needle.length);
    return [expandedStart, expandedEnd];
  }

  // Strategy 3: n-gram density cluster.
  //
  // Robust fallback for non-contiguous or lightly-paraphrased
  // chunks — Docling HybridChunker reassembles content across
  // structural boundaries, so the chunk's bytes exist in the
  // markdown but not as a single substring. Build a fingerprint of
  // the chunk as 8-char windows, locate each window in the
  // markdown, then find the densest cluster of those positions
  // (the spot where the most distinctive parts of the chunk
  // co-occur). This is what diff and content-fingerprinting tools
  // use; it gracefully degrades from "exact match" toward "best
  // approximate region" without ever returning a confidently-wrong
  // position the way leading-N-chars matching does.
  const N = 8;
  const STEP = 4; // sparser sampling — we don't need every n-gram
  const grams = new Set<string>();
  for (let i = 0; i + N <= lowerNeedle.length; i += STEP) {
    const g = lowerNeedle.slice(i, i + N);
    // Only keep grams with at least one alphanumeric char so we
    // don't anchor on whitespace or punctuation runs.
    if (/[a-z0-9]/.test(g)) grams.add(g);
  }
  if (grams.size < 3) return null;

  // Locate each n-gram. Cap total positions to keep the worst
  // case bounded for large papers.
  const POSITION_CAP = 20000;
  const positionsNorm: number[] = [];
  for (const g of grams) {
    let from = 0;
    let cycles = 0;
    while (cycles < 200) {
      const idx = lowerNorm.indexOf(g, from);
      if (idx === -1) break;
      positionsNorm.push(idx);
      from = idx + 1;
      cycles += 1;
      if (positionsNorm.length >= POSITION_CAP) break;
    }
    if (positionsNorm.length >= POSITION_CAP) break;
  }
  if (positionsNorm.length < 2) return null;

  positionsNorm.sort((a, b) => a - b);
  const windowSize = lowerNeedle.length;
  let bestCount = 0;
  let bestStartNorm = positionsNorm[0];
  let left = 0;
  for (let right = 0; right < positionsNorm.length; right += 1) {
    while (positionsNorm[right] - positionsNorm[left] > windowSize) {
      left += 1;
    }
    const count = right - left + 1;
    if (count > bestCount) {
      bestCount = count;
      bestStartNorm = positionsNorm[left];
    }
  }

  // Require multiple distinctive matches to claim a hit — a single
  // generic n-gram match isn't enough to anchor on.
  if (bestCount < 3) return null;

  const startInOrig = indexMap[bestStartNorm];
  return [startInOrig, Math.min(haystack.length, startInOrig + needle.length)];
}

/** Splice a single ``mark`` tag into the markdown anchored on the
 * most-specific match we can find. Strategy:
 *   1. If Pass 2 gave us a verbatim ``quote`` and it appears in the
 *      paper markdown, highlight ONLY the quote — bright yellow,
 *      scroll into view. Each citation has its own quote so
 *      different clicks land on different precise spans. This is
 *      the primary mechanism now.
 *   2. Otherwise (Pass 2 failed for this marker, or quote not
 *      locatable), fall back to highlighting the chunk_text region
 *      as a softer chunk-level match. Less precise, still useful.
 *   3. If neither anchors, render the markdown unhighlighted. */
function highlightInMarkdown(
  markdown: string,
  chunkText: string,
  quote?: string,
): { highlighted: string; chunkAnchorId: string | null } {
  const anchorId = 'pq-citation-anchor';

  // Primary: anchor on the verbatim quote.
  if (quote && quote.trim().length >= 8) {
    const quoteSpan = findFlexibleSpan(markdown, quote);
    if (quoteSpan) {
      const [qStart, qEnd] = quoteSpan;
      const before = markdown.slice(0, qStart);
      const matched = markdown.slice(qStart, qEnd);
      const after = markdown.slice(qEnd);
      const wrapped =
        `<mark class="quote-highlight" id="${anchorId}">${matched}</mark>`;
      return { highlighted: before + wrapped + after, chunkAnchorId: anchorId };
    }
  }

  // Fallback: anchor on the full chunk text (less precise — chunks
  // may not appear contiguously in the source markdown).
  const chunkSpan = findFlexibleSpan(markdown, chunkText);
  if (chunkSpan) {
    const [cStart, cEnd] = chunkSpan;
    const before = markdown.slice(0, cStart);
    const matched = markdown.slice(cStart, cEnd);
    const after = markdown.slice(cEnd);
    const wrapped =
      `<mark class="chunk-highlight" id="${anchorId}">${matched}</mark>`;
    return { highlighted: before + wrapped + after, chunkAnchorId: anchorId };
  }

  // Last resort: render unhighlighted. The user still sees the
  // paper's markdown opened to the cited file/section.
  return { highlighted: markdown, chunkAnchorId: null };
}

export const MarkdownPreviewPanel: FC<MarkdownPreviewPanelProps> = ({
  source,
  citation,
  triggerKey,
  onClose,
}) => {
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Re-fetch every time the source filename changes.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setMarkdown(null);

    const url = `/api/chat/files/${encodeURIComponent(source.source)}/markdown`;
    fetch(url, { credentials: 'include' })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to load markdown (${res.status})`);
        }
        return res.json() as Promise<{ markdown: string }>;
      })
      .then((data) => {
        if (cancelled) return;
        setMarkdown(data.markdown);
        setLoading(false);
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setError(err.message ?? 'Failed to load markdown.');
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [source.source]);

  const { highlighted, chunkAnchorId } = useMemo(() => {
    if (!markdown) return { highlighted: '', chunkAnchorId: null };
    return highlightInMarkdown(markdown, source.chunk_text, citation?.quote);
  }, [markdown, source.chunk_text, citation?.quote]);

  // Scroll the highlighted chunk into view once the markdown renders.
  useEffect(() => {
    if (!chunkAnchorId) return;
    const t = window.setTimeout(() => {
      const el = containerRef.current?.querySelector(`#${chunkAnchorId}`);
      if (el) {
        (el as HTMLElement).scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        });
      }
    }, 50);
    return () => window.clearTimeout(t);
  }, [highlighted, chunkAnchorId]);

  return (
    <aside className="w-[min(36rem,46vw)] min-w-[24rem] border-l border-base-200 bg-base-100 flex flex-col">
      <div className="flex items-center justify-between gap-2 px-4 py-3 border-b border-base-200">
        <div className="min-w-0">
          <h3
            className="text-sm font-semibold text-base-content truncate"
            title={source.source}
          >
            {source.source}
          </h3>
          {source.section && (
            <p className="text-xs text-base-content/60 truncate">
              {source.section}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onClose}
          className="btn btn-ghost btn-sm btn-square"
          title="Close preview"
          aria-label="Close preview"
        >
          <X size={16} weight="bold" />
        </button>
      </div>

      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto px-5 py-4 text-sm leading-relaxed"
      >
        {loading && (
          <div className="flex items-center gap-2 text-base-content/60">
            <span className="loading loading-spinner loading-sm" />
            <span>Loading paper…</span>
          </div>
        )}
        {error && (
          <div className="alert alert-error text-sm">
            <span>{error}</span>
          </div>
        )}
        {!loading && !error && markdown && (
          // ``key`` includes triggerKey so each citation click force-
          // remounts the rendered markdown — the new <mark> elements
          // re-trigger their CSS keyframe animation, producing the
          // brief yellow flash the user sees on click.
          <div
            key={triggerKey}
            className="prose prose-sm max-w-none prose-headings:mt-4 prose-headings:mb-2"
          >
            <ReactMarkdown
              skipHtml={false}
              rehypePlugins={rehypePlugins}
              components={{
                mark: ({ children, className, id }) => (
                  <mark className={className} id={id}>
                    {children}
                  </mark>
                ),
              }}
            >
              {highlighted}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </aside>
  );
};

export default MarkdownPreviewPanel;
