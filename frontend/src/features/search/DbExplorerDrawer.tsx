/**
 * Database panel — retractable right-side drawer.
 *
 * Slide-in chrome, vertical trigger tab, drag-resize (440px → 35% of
 * viewport width) persisted to localStorage, overlay/Escape/close
 * dismissal. Three tabs (Papers / Entities / Journals). Auto-opens via
 * chart click hooks driven by the parent (clicking a stat card or
 * chart segment in Dashboard.tsx).
 *
 * Resize bounds match the design spec: hard 440px lower bound (the
 * panel needs that much breathing room for the paper card layout) and
 * a dynamic 35% upper bound computed against window.innerWidth so the
 * main content always retains 65% of the viewport.
 */

import React, { useEffect, useRef, useState } from 'react';
import { X as CloseIcon, Database } from '@phosphor-icons/react';
import { dbApi } from '../../lib/api';
import { sanitizeHtml } from '../../utils/sanitize';
import { Skeleton } from '@/components/ui/skeleton';

const XP_WIDTH_KEY = 'phytoquery-xp-width';
const MIN_W = 440;
const MAX_W_PCT = 0.35;
const DEFAULT_W = 440;
/** Runtime maximum width — recomputed at every drag against the live
 * viewport. Falls back to a generous 720px for SSR / pre-mount. */
const computeMaxW = (): number => {
  if (typeof window === 'undefined') return 720;
  return Math.max(MIN_W, Math.floor(window.innerWidth * MAX_W_PCT));
};

export type DrawerTab = 'papers' | 'entities' | 'journals';

export interface DrawerFilter {
  kind: 'country' | 'entity' | 'journal' | 'papers' | 'entities' | 'journals';
  label: string;
  value?: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
  tab: DrawerTab;
  onTabChange: (t: DrawerTab) => void;
  filter: DrawerFilter | null;
  onClearFilter: () => void;
  entities: { name: string; value: number }[];
  journals: { name: string; value: number }[];
}

interface PaperRow {
  title?: string;
  journal?: string;
  year?: number | string;
  doi?: string;
}

const DbExplorerDrawer: React.FC<Props> = ({
  open, onClose, tab, onTabChange, filter, onClearFilter, entities, journals,
}) => {
  const [width, setWidth] = useState<number>(() => {
    if (typeof window === 'undefined') return DEFAULT_W;
    const saved = parseInt(localStorage.getItem(XP_WIDTH_KEY) || '', 10);
    const maxW = computeMaxW();
    return saved >= MIN_W && saved <= maxW ? saved : DEFAULT_W;
  });

  const [papers, setPapers] = useState<PaperRow[] | null>(null);
  const [papersError, setPapersError] = useState(false);

  const draggingRef = useRef(false);
  const dragStartX = useRef(0);
  const dragStartW = useRef(0);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open || tab !== 'papers' || papers !== null || papersError) return;
    dbApi.getPapers(50, 0)
      .then((data: any) => {
        const list: PaperRow[] = Array.isArray(data) ? data : data?.papers ?? data?.items ?? [];
        setPapers(list);
      })
      .catch(() => { setPapersError(true); setPapers([]); });
  }, [open, tab, papers, papersError]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current) return;
      const delta = dragStartX.current - e.clientX;
      const w = Math.min(computeMaxW(), Math.max(MIN_W, dragStartW.current + delta));
      setWidth(w);
      try { localStorage.setItem(XP_WIDTH_KEY, String(w)); } catch { /* private mode */ }
    };
    const onUp = () => {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, []);

  const startDrag = (e: React.MouseEvent) => {
    draggingRef.current = true;
    dragStartX.current = e.clientX;
    dragStartW.current = width;
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    e.preventDefault();
  };

  return (
    <>
      {/* Dim overlay */}
      <div
        onClick={onClose}
        className={`fixed inset-0 z-40 transition-colors duration-200 ${
          open ? 'bg-black/[0.06] pointer-events-auto' : 'bg-transparent pointer-events-none'
        }`}
        aria-hidden
      />

      {/* Drawer */}
      <aside
        className={`fixed top-0 right-0 bottom-0 z-50 bg-white border-l border-slate-200 flex flex-col overflow-hidden ${
          open ? 'translate-x-0 shadow-[-4px_0_24px_rgba(0,0,0,0.08)]' : 'translate-x-full shadow-none'
        }`}
        style={{
          width: `${width}px`,
          transition: 'transform 220ms cubic-bezier(0.4,0,0.2,1), box-shadow 220ms ease',
        }}
        aria-hidden={!open}
      >
        {/* Resize handle */}
        <div
          onMouseDown={startDrag}
          className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500 z-10 transition-colors"
        />

        {/* Header */}
        <div className="px-4 pt-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <Database size={14} weight="bold" />
              Database
            </div>
            <button
              onClick={onClose}
              className="w-7 h-7 rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700 flex items-center justify-center"
              aria-label="Close"
            >
              <CloseIcon size={14} weight="bold" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-slate-200">
            {(['papers', 'entities', 'journals'] as const).map((t) => (
              <button
                key={t}
                onClick={() => onTabChange(t)}
                className={`flex-1 text-center py-2 text-xs font-medium capitalize transition-colors border-b-2 -mb-px ${
                  tab === t
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-slate-500 hover:text-slate-800'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Filter chip */}
        {filter && (
          <div className="flex items-center gap-2 mx-3 mt-2 px-3 py-1.5 bg-blue-50 border border-blue-200 rounded-md text-[11px] text-blue-700 font-medium">
            <span className="flex-1 leading-snug">{filter.label}</span>
            <button
              onClick={onClearFilter}
              className="text-slate-400 hover:text-slate-700 text-base leading-none px-1"
              aria-label="Clear filter"
            >
              ×
            </button>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {tab === 'papers' && <PapersList papers={papers} error={papersError} />}
          {tab === 'entities' && <EntitiesList entities={entities} filter={filter} />}
          {tab === 'journals' && <JournalsList journals={journals} filter={filter} />}
        </div>

        {/* Footer */}
        <div className="px-3 py-2 bg-slate-50 border-t border-slate-200 text-[11px] text-slate-500 flex-shrink-0">
          {tab === 'papers' && (papers === null ? 'Loading papers…' : `Showing ${papers.length} papers`)}
          {tab === 'entities' && `${entities.length} entity types`}
          {tab === 'journals' && `${journals.length} journals`}
        </div>
      </aside>

    </>
  );
};

// ─── Subcomponents ────────────────────────────────────────────────────────────

function PapersList({ papers, error }: { papers: PaperRow[] | null; error: boolean }) {
  if (error) {
    return <div className="p-7 text-center text-slate-400 text-sm">Failed to load papers</div>;
  }
  if (papers === null) {
    return (
      <div className="px-4 py-2 space-y-3" aria-label="Loading papers">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="space-y-1.5 py-1.5">
            <Skeleton className="h-3 w-full" />
            <Skeleton className="h-3 w-4/5" />
            <div className="flex gap-2 pt-0.5">
              <Skeleton className="h-2 w-24" />
              <Skeleton className="h-2 w-10" />
            </div>
          </div>
        ))}
      </div>
    );
  }
  if (papers.length === 0) {
    return <div className="p-7 text-center text-slate-400 text-sm">No papers</div>;
  }
  return (
    <>
      {papers.map((p, i) => (
        <div key={i} className="px-4 py-2.5 border-b border-slate-100 hover:bg-slate-50 cursor-pointer transition-colors">
          <div
            className="text-xs text-slate-800 leading-snug line-clamp-2"
            dangerouslySetInnerHTML={{ __html: sanitizeHtml(p.title ?? '(untitled)') }}
          />
          <div className="flex gap-2 mt-1 text-[10px]">
            {p.journal && <span className="text-blue-600 truncate max-w-[60%]">{p.journal}</span>}
            {p.year != null && <span className="font-mono text-slate-400">{p.year}</span>}
          </div>
        </div>
      ))}
    </>
  );
}

function EntitiesList({
  entities,
  filter,
}: {
  entities: { name: string; value: number }[];
  filter: DrawerFilter | null;
}) {
  const filterValue = filter?.kind === 'entity' ? filter.value?.toLowerCase() : null;
  if (entities.length === 0) {
    return <div className="p-7 text-center text-slate-400 text-sm">No entities</div>;
  }
  return (
    <>
      {entities.map((e, i) => {
        const isMatch = filterValue && e.name.toLowerCase().includes(filterValue);
        return (
          <div
            key={i}
            className={`px-4 py-2.5 border-b border-slate-100 flex items-center gap-2 transition-colors ${
              isMatch ? 'bg-blue-50/50' : 'hover:bg-slate-50'
            }`}
          >
            <span className="text-xs font-medium text-slate-800 flex-1 capitalize">
              {e.name.replace(/_/g, ' ')}
            </span>
            <span className="text-[10px] font-mono text-purple-600 font-medium">
              {e.value.toLocaleString()}
            </span>
          </div>
        );
      })}
    </>
  );
}

function JournalsList({
  journals,
  filter,
}: {
  journals: { name: string; value: number }[];
  filter: DrawerFilter | null;
}) {
  const filterValue = filter?.kind === 'journal' ? filter.value : null;
  if (journals.length === 0) {
    return <div className="p-7 text-center text-slate-400 text-sm">No journals</div>;
  }
  const max = journals[0]?.value ?? 1;
  return (
    <>
      {journals.map((j, i) => {
        const pct = Math.max(2, Math.round((j.value / max) * 100));
        const isMatch = filterValue && j.name === filterValue;
        return (
          <div
            key={i}
            className={`px-4 py-2.5 border-b border-slate-100 transition-colors ${
              isMatch ? 'bg-blue-50/50' : 'hover:bg-slate-50'
            }`}
          >
            <div className="text-xs font-medium text-slate-800 mb-1.5 truncate">{j.name}</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1 bg-slate-200 rounded overflow-hidden">
                <div className="h-full bg-blue-500 rounded transition-all" style={{ width: `${pct}%` }} />
              </div>
              <span className="text-[10px] font-mono text-blue-600 min-w-[2rem] text-right">
                {j.value}
              </span>
            </div>
          </div>
        );
      })}
    </>
  );
}

export default DbExplorerDrawer;
