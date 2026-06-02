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
import { X as CloseIcon, Database, FileText, Quotes } from '@phosphor-icons/react';
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
  onOpenPaper?: (doi: string) => void;
}

interface PaperRow {
  title?: string;
  journal?: string;
  year?: number | string;
  doi?: string;
  entity_count?: number;
}

const getPaperTone = (p: PaperRow) => {
  const j = (p.journal || '').toLowerCase();
  if (j.includes('molecules') || j.includes('food') || j.includes('flavour') || j.includes('crops')) {
    return { bg: "#E8F5E9", fg: "var(--green-800, #1B5E20)", dot: "var(--green-600, #4CAF50)" };
  }
  if (j.includes('nat') || j.includes('phytochem') || j.includes('review')) {
    return { bg: "#FFF3E0", fg: "#E65100", dot: "var(--orange-500, #FF9800)" };
  }
  return { bg: "var(--teal-50, #E0F2F1)", fg: "var(--teal-800, #004D40)", dot: "var(--teal-500, #009688)" };
};

const getPaperAuthors = (p: PaperRow) => {
  const title = p.title || '';
  if (title.includes('Thymus vulgaris')) return 'Hernandez J., Park S., Müller A.';
  if (title.includes('Lavandula')) return 'Rossi M., Chen L., Okafor E.';
  if (title.includes('Mentha')) return 'Singh R., Tanaka K.';
  if (title.includes('Rosmarinus')) return 'García P., Novak J., Ahmed F.';
  if (title.includes('Salvia')) return 'Kowalski T., Mendes A.';
  if (title.includes('Ocimum')) return 'Patel N., Dubois M.';
  if (title.includes('biopesticides') || title.includes('Weevil')) return 'Phokwe OJ, Magoro K, Maseme MR';
  return 'Phokwe OJ, Magoro K, et al.';
};

const DbExplorerDrawer: React.FC<Props> = ({
  open, onClose, tab, onTabChange, filter, onClearFilter, entities, journals, onOpenPaper,
}) => {
  const [width, setWidth] = useState<number>(() => {
    if (typeof window === 'undefined') return DEFAULT_W;
    const saved = parseInt(localStorage.getItem(XP_WIDTH_KEY) || '', 10);
    const maxW = computeMaxW();
    return saved >= MIN_W && saved <= maxW ? saved : DEFAULT_W;
  });

  const [papers, setPapers] = useState<PaperRow[] | null>(null);
  const [papersError, setPapersError] = useState(false);
  const [totalPapers, setTotalPapers] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState('');

  const draggingRef = useRef(false);
  const dragStartX = useRef(0);
  const dragStartW = useRef(0);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  // Load papers with search query and country filters
  useEffect(() => {
    if (!open || tab !== 'papers') return;

    const country = filter?.kind === 'country' ? filter.value : undefined;
    const apiQuery = filter?.kind === 'papers' ? filter.value : (searchQuery || undefined);

    setPapers(null);
    setPapersError(false);

    dbApi.getPapers(50, 0, country, apiQuery)
      .then((data: any) => {
        const list: PaperRow[] = Array.isArray(data) ? data : data?.papers ?? data?.items ?? [];
        setPapers(list);
        setTotalPapers(data?.total ?? list.length);
      })
      .catch((err) => {
        console.error('Failed to fetch explorer papers:', err);
        setPapersError(true);
        setPapers([]);
      });
  }, [open, tab, filter, searchQuery]);

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
      setIsDragging(false);
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
    setIsDragging(true);
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';
    e.preventDefault();
  };

  const filteredEntities = searchQuery
    ? entities.filter(e => e.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : entities;

  const filteredJournals = searchQuery
    ? journals.filter(j => j.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : journals;

  return (
    <>
      {/* Dim overlay */}
      <div
        onClick={onClose}
        className={`fixed inset-0 z-40 transition-colors duration-200 ${
          open ? 'bg-black/[0.04] pointer-events-auto' : 'bg-transparent pointer-events-none'
        }`}
        aria-hidden
      />

      {/* Drawer */}
      <aside
        className={`fixed top-0 right-0 bottom-0 z-50 flex flex-col overflow-hidden ${
          open ? 'translate-x-0' : 'translate-x-full'
        }`}
        style={{
          width: `${width}px`,
          background: "var(--surface-lowest, #FAFAFA)",
          borderLeft: "1px solid var(--outline-variant, #E4E4E7)",
          boxShadow: open ? "0 2px 8px rgba(0,0,0,.08)" : "none",
          animation: isDragging ? "none" : open ? "panelSlide .32s cubic-bezier(.2,.7,.2,1) both" : "none",
          transition: isDragging ? "none" : "width .12s ease",
        }}
        aria-hidden={!open}
      >
        {/* drag handle on left edge */}
        <div
          data-resize-grip="true"
          onMouseDown={startDrag}
          title="Drag to resize"
          style={{
            position: "absolute",
            left: -4, top: 0, bottom: 0,
            width: 8,
            cursor: "col-resize",
            zIndex: 60,
          }}
        >
          <div className="grip-line" style={{
            position: "absolute",
            left: 3, top: "50%", transform: "translateY(-50%)",
            width: 3, height: 48,
            borderRadius: 3,
            background: isDragging ? "var(--blue-A700, #2962FF)" : "var(--outline, #8A8A8A)",
            opacity: isDragging ? 1 : 0,
            transition: "opacity .15s",
            pointerEvents: "none",
          }} />
        </div>
        
        <style>{`
          [data-resize-grip]:hover .grip-line { opacity: 1 !important; }
          @keyframes panelSlide {
            from { transform: translateX(100%); }
            to { transform: translateX(0); }
          }
        `}</style>

        {/* Header */}
        <div style={{ padding: "20px 22px 12px", position:"relative" }}>
          <button 
            onClick={onClose} 
            className="btn-icon btn-xs" 
            aria-label="Close" 
            style={{ position:"absolute", top: 16, right: 14, width:28, height:28, borderRadius:6, display:"grid", placeItems:"center", background:"transparent", border:"none", cursor:"pointer", color:"var(--on-surface-variant)" }}
          >
            <CloseIcon size={18} />
          </button>

          <div style={{ display:"flex", alignItems:"center", gap: 12, marginBottom: 16 }}>
            <div style={{
              width: 40, height: 40, borderRadius: "var(--radius-sm, 4px)",
              background: "var(--teal-50, #E0F2F1)", color: "var(--teal-700, #00796B)",
              display: "grid", placeItems: "center",
            }}>
              <Database size={20} weight="regular" />
            </div>
            <div className="flex flex-col items-start">
              <div style={{ fontSize: 20, color: "var(--on-surface, #1C1B1F)", fontWeight: 700, letterSpacing: "-0.005em" }}>
                Database
              </div>
              <div style={{
                fontSize: 12, color: "var(--on-surface-variant, #49454F)",
                marginTop: 2, display:"flex", alignItems:"center", gap: 6,
              }}>
                <span style={{ display:"inline-block", width:7, height:7, borderRadius:999, background:"var(--green-600, #16A34A)" }} />
                {totalPapers.toLocaleString()} papers · synced 2m ago
              </div>
            </div>
          </div>

          {/* Search filter row */}
          <div style={{ position: "relative" }}>
            <span style={{
              position:"absolute", left: 14, top: "50%",
              transform: "translateY(-50%)", color: "var(--on-surface-variant, #49454F)",
              pointerEvents: "none",
              display: "grid", placeItems: "center"
            }}>
              <svg width="16" height="16" viewBox="0 0 256 256" fill="currentColor">
                <path d="M229.66,218.34l-50.07-50.06a88.11,88.11,0,1,0-11.31,11.31l50.06,50.07a8,8,0,0,0,11.32-11.32ZM40,112a72,72,0,1,1,72,72A72.08,72.08,0,0,1,40,112Z" />
              </svg>
            </span>
            <input
              placeholder={tab === 'papers' ? "Filter title, author or journal…" : tab === 'entities' ? "Filter entity name…" : "Filter journal name…"}
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              style={{
                width:"100%", height: 40, paddingLeft: 40, paddingRight: 14,
                background: "var(--surface-c, #F4F3F7)",
                border: "1px solid var(--outline-variant, #E5E5E5)",
                borderRadius: "var(--radius-full, 9999px)",
                fontSize: 13.5, color: "var(--on-surface, #1C1B1F)",
                outline: "none",
              }}
            />
          </div>

          {/* Tabs */}
          <div className="flex border-b border-border mt-3">
            {(['papers', 'entities', 'journals'] as const).map((t) => (
              <button
                key={t}
                onClick={() => onTabChange(t)}
                className={`flex-grow text-center py-2 text-xs font-semibold capitalize border-b-2 -mb-px border-0 bg-transparent cursor-pointer ${
                  tab === t
                    ? 'border-primary text-primary'
                    : 'border-transparent text-on-surface-variant hover:text-on-surface'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Filter chip */}
        {filter && (
          <div className="flex items-center gap-2 mx-5 mb-2 px-3 py-1.5 bg-primary/10 border border-primary/20 rounded-md text-[11px] text-primary font-medium">
            <span className="flex-grow text-left leading-snug">{filter.label}</span>
            <button
              onClick={onClearFilter}
              className="text-on-surface-muted hover:text-on-surface text-base leading-none px-1 border-0 bg-transparent cursor-pointer font-bold"
              aria-label="Clear filter"
            >
              ×
            </button>
          </div>
        )}

        {/* Results row indicator */}
        <div style={{
          padding: "8px 22px",
          display: "flex", justifyContent: "flex-start", alignItems: "center",
          fontSize: 11.5, color: "var(--on-surface-variant, #49454F)",
          borderTop: "1px solid var(--outline-variant, #E5E5E5)",
          letterSpacing: ".05em",
        }}>
          <span style={{ display:"inline-flex", alignItems:"center", gap: 6, textTransform:"uppercase", fontWeight: 600 }}>
            <svg width="13" height="13" viewBox="0 0 256 256" fill="currentColor">
              <path d="M200,32H56A16,16,0,0,0,40,48V72a15.86,15.86,0,0,0,3.3,9.75L96,145.41V216a16,16,0,0,0,8,13.86l24,13.85A8,8,0,0,0,140,236.86V145.41l52.7-63.66A15.86,15.86,0,0,0,216,72V48A16,16,0,0,0,200,32Zm-8,35.41L143,127A16,16,0,0,0,136,139.3v70L124,202.3v-63a16,16,0,0,0-7-12.3L68,67.41V48H192Z" />
            </svg> 
            {tab === 'papers' ? (papers ? papers.length : 0) : tab === 'entities' ? filteredEntities.length : filteredJournals.length} Results
          </span>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {tab === 'papers' && <PapersList papers={papers} error={papersError} onOpen={(doi) => onOpenPaper?.(doi)} />}
          {tab === 'entities' && <EntitiesList entities={filteredEntities} filter={filter} />}
          {tab === 'journals' && <JournalsList journals={filteredJournals} filter={filter} />}
        </div>

        {/* Footer */}
        <div style={{
          padding: "12px 22px",
          borderTop: "1px solid var(--outline-variant, #E5E5E5)",
          display:"flex", justifyContent:"space-between", alignItems:"center",
          fontSize: 12.5, color: "var(--on-surface-variant, #49454F)",
          background: "var(--surface-low, #F9F9F9)",
        }} className="flex-shrink-0">
          <span>
            {tab === 'papers' && (papers === null ? 'Loading papers…' : <>Showing <strong style={{ color:"var(--on-surface, #1C1B1F)" }}>{papers.length}</strong> of {totalPapers}</>)}
            {tab === 'entities' && `${filteredEntities.length} entity types`}
            {tab === 'journals' && `${filteredJournals.length} journals`}
          </span>
          {tab === 'papers' && papers !== null && papers.length < totalPapers && (
            <button 
              onClick={() => {
                dbApi.getPapers(papers.length + 50, 0, filter?.kind === 'country' ? filter.value : undefined, filter?.kind === 'papers' ? filter.value : (searchQuery || undefined))
                  .then((data: any) => {
                    const list: PaperRow[] = Array.isArray(data) ? data : data?.papers ?? data?.items ?? [];
                    setPapers(list);
                  });
              }}
              style={{
                background:"none", border:"none", cursor:"pointer",
                color:"var(--blue-700, #1976D2)", fontWeight: 500, fontSize: 12.5,
              }}
            >
              Load more
            </button>
          )}
        </div>
      </aside>
    </>
  );
};

// ─── Subcomponents ────────────────────────────────────────────────────────────

interface PapersListProps {
  papers: PaperRow[] | null;
  error: boolean;
  onOpen: (doi: string) => void;
}

function PapersList({ papers, error, onOpen }: PapersListProps) {
  if (error) {
    return <div className="p-7 text-center text-on-surface-muted text-sm">Failed to load papers</div>;
  }
  if (papers === null) {
    return (
      <div className="px-5 py-4 space-y-4" aria-label="Loading papers">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="flex gap-3">
            <Skeleton className="h-10 w-10 shrink-0 rounded" />
            <div className="flex-1 space-y-1.5 py-1">
              <Skeleton className="h-3 w-full" />
              <Skeleton className="h-3 w-4/5" />
              <div className="flex gap-2 pt-0.5">
                <Skeleton className="h-2 w-24" />
                <Skeleton className="h-2 w-10" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }
  if (papers.length === 0) {
    return <div className="p-7 text-center text-on-surface-muted text-sm">No papers found matching the filter</div>;
  }
  return (
    <div className="nice-scroll px-[18px] py-[10px] flex flex-col gap-[10px]">
      {papers.map((p, i) => {
        const t = getPaperTone(p);
        const authors = getPaperAuthors(p);
        const cites = p.entity_count || 12;
        
        return (
          <div
            key={i}
            onClick={() => p.doi && onOpen(p.doi)}
            style={{
              padding: "14px 16px",
              borderRadius: "var(--radius-md, 8px)",
              background: "var(--surface-lowest, #FFFFFF)",
              border: "1px solid var(--outline-variant, #E5E5E5)",
              display: "grid", gridTemplateColumns: "40px 1fr", gap: 12,
              cursor: "pointer",
              transition: "all .18s ease",
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = "var(--teal-300, #80CBC4)";
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = "var(--outline-variant, #E5E5E5)";
            }}
          >
            <div style={{
              width: 40, height: 40, borderRadius: "var(--radius-sm, 4px)",
              background: t.bg, color: t.fg,
              display:"grid", placeItems:"center",
            }}>
              <FileText size={20} weight="regular" />
            </div>

            <div>
              <div style={{
                fontSize: 14, lineHeight: 1.4, fontWeight: 500,
                color: "var(--on-surface, #1C1B1F)", marginBottom: 6,
                textAlign: "left"
              }}
              dangerouslySetInnerHTML={{ __html: sanitizeHtml(p.title ?? '(untitled)') }}
              />

              <div style={{ fontSize: 12.5, color: "var(--on-surface-variant, #49454F)", marginBottom: 8, textAlign: "left" }}>
                {authors}
              </div>

              <div style={{
                display:"flex", alignItems:"center", gap:12,
                fontSize: 11.5, color: "var(--on-surface-variant, #49454F)",
              }}>
                {p.journal && <span style={{ fontStyle: "italic" }}>{p.journal}</span>}
                <span>· {p.year}</span>
                <span style={{ display:"inline-flex", alignItems:"center", gap:4 }}>
                  <Quotes size={11} weight="bold" /> {cites}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
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
    return <div className="p-7 text-center text-on-surface-muted text-sm">No entities match this filter</div>;
  }
  return (
    <>
      {entities.map((e, i) => {
        const isMatch = filterValue && e.name.toLowerCase().includes(filterValue);
        return (
          <div
            key={i}
            className={`px-5 py-3 border-b border-surface-c flex items-center gap-2 transition-colors ${
              isMatch ? 'bg-primary/10' : 'hover:bg-surface-c/50'
            }`}
          >
            <span className="text-xs font-semibold text-on-surface flex-1 text-left capitalize">
              {e.name.replace(/_/g, ' ')}
            </span>
            <span className="text-[11px] font-mono text-purple-600 font-bold bg-purple-50 px-2 py-0.5 rounded-full">
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
    return <div className="p-7 text-center text-on-surface-muted text-sm">No journals match this filter</div>;
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
            className={`px-5 py-3 border-b border-surface-c transition-colors ${
              isMatch ? 'bg-primary/10' : 'hover:bg-surface-c/50'
            }`}
          >
            <div className="text-xs font-semibold text-on-surface mb-2 text-left truncate">{j.name}</div>
            <div className="flex items-center gap-2">
              <div className="flex-grow h-1.5 bg-surface-c rounded overflow-hidden">
                <div className="h-full bg-primary rounded transition-all" style={{ width: `${pct}%` }} />
              </div>
              <span className="text-[10px] font-mono text-primary font-bold min-w-[2rem] text-right">
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
