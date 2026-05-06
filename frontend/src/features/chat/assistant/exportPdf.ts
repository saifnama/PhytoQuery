/**
 * PDF export of a single assistant answer + its sources.
 *
 * Triggered from a button below each assistant message. Generates a
 * paginated PDF with: question (the user turn that prompted this
 * answer), answer body (markdown markers stripped to plain text), and
 * the list of source citations.
 *
 * Uses jspdf only — no html2canvas — so the bundle stays small and the
 * export is text-selectable (better for scientific quoting than a
 * raster screenshot would be).
 */

import { jsPDF } from 'jspdf';
import type { RagSource } from './runtime';

export interface AnswerExportPayload {
  /** The user question that produced this answer. May be empty. */
  question: string;
  /** Assistant answer text (markdown). Markers are stripped at render time. */
  answer: string;
  /** Sources cited in the answer (may be empty). */
  sources: readonly RagSource[];
  /** ISO timestamp for the file name and document footer. Defaults to now. */
  timestamp?: string;
}

const PAGE_MARGIN_X = 14;
const PAGE_MARGIN_Y = 16;
const LINE_HEIGHT = 5.5;
const HEADING_GAP = 4;
const SECTION_GAP = 6;

/** Strip just enough markdown so the PDF reads as clean prose. We don't
 * try to faithfully render bold/headings — text-only PDF is acceptable
 * for quoting. */
function stripMarkdown(input: string): string {
  return input
    .replace(/```[\s\S]*?```/g, (block) => {
      // Keep code blocks as-is but unwrap the fences.
      return block.replace(/^```[a-zA-Z0-9_-]*\n?/, '').replace(/```\s*$/, '');
    })
    .replace(/`([^`]+)`/g, '$1')          // inline code
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')   // headings
    .replace(/\*\*(.+?)\*\*/g, '$1')      // bold
    .replace(/\*(.+?)\*/g, '$1')          // italic
    .replace(/__(.+?)__/g, '$1')          // bold (alt)
    .replace(/_(.+?)_/g, '$1')            // italic (alt)
    .replace(/^\s*[-*+]\s+/gm, '• ')      // bullets
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // markdown links
    .trim();
}

function safeFilenameTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) {
    return new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
  }
  return d.toISOString().slice(0, 16).replace(/[:T]/g, '-');
}

interface PdfRenderState {
  doc: jsPDF;
  y: number;
  pageHeight: number;
  contentWidth: number;
}

function ensureSpace(state: PdfRenderState, neededLines: number): void {
  const neededHeight = neededLines * LINE_HEIGHT;
  if (state.y + neededHeight > state.pageHeight - PAGE_MARGIN_Y) {
    state.doc.addPage();
    state.y = PAGE_MARGIN_Y;
  }
}

function writeWrapped(state: PdfRenderState, text: string): void {
  if (!text) return;
  const lines = state.doc.splitTextToSize(text, state.contentWidth) as string[];
  for (const line of lines) {
    ensureSpace(state, 1);
    state.doc.text(line, PAGE_MARGIN_X, state.y);
    state.y += LINE_HEIGHT;
  }
}

function writeHeading(state: PdfRenderState, label: string): void {
  ensureSpace(state, 2);
  state.y += HEADING_GAP;
  state.doc.setFont('helvetica', 'bold');
  state.doc.setFontSize(11);
  state.doc.text(label, PAGE_MARGIN_X, state.y);
  state.y += LINE_HEIGHT;
  state.doc.setFont('helvetica', 'normal');
  state.doc.setFontSize(10);
}

export function exportAnswerAsPdf(payload: AnswerExportPayload): void {
  const timestamp = payload.timestamp ?? new Date().toISOString();
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const state: PdfRenderState = {
    doc,
    y: PAGE_MARGIN_Y,
    pageHeight,
    contentWidth: pageWidth - PAGE_MARGIN_X * 2,
  };

  // Document title
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.text('PhytoQuery Answer', PAGE_MARGIN_X, state.y);
  state.y += LINE_HEIGHT + 1;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  doc.setTextColor(120);
  doc.text(new Date(timestamp).toLocaleString(), PAGE_MARGIN_X, state.y);
  state.y += LINE_HEIGHT;
  doc.setTextColor(0);
  doc.setFontSize(10);
  state.y += SECTION_GAP;

  if (payload.question.trim()) {
    writeHeading(state, 'Question');
    writeWrapped(state, stripMarkdown(payload.question));
  }

  writeHeading(state, 'Answer');
  writeWrapped(state, stripMarkdown(payload.answer || '(empty)'));

  if (payload.sources.length > 0) {
    writeHeading(state, 'Sources');
    payload.sources.forEach((source, idx) => {
      const header = source.section
        ? `${idx + 1}. ${source.source} — ${source.section} (${source.score}%)`
        : `${idx + 1}. ${source.source} (${source.score}%)`;
      doc.setFont('helvetica', 'bold');
      writeWrapped(state, header);
      doc.setFont('helvetica', 'normal');
      const excerpt = source.chunk_text.slice(0, 600);
      if (excerpt) writeWrapped(state, excerpt);
      state.y += 2;
    });
  }

  doc.save(`phytoquery-answer-${safeFilenameTimestamp(timestamp)}.pdf`);
}
