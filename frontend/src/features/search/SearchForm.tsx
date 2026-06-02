import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  MagnifyingGlass,
  X,
  CaretDown,
  LockSimple,
  LockSimpleOpen,
  Article,
  ArrowUp,
  ArrowDown,
} from '@phosphor-icons/react';
import type { SearchFilters } from '../../types';
import { searchTypesApi } from '../../lib/api';

/**
 * SearchForm — pill-shaped search bar with expandable filters.
 *
 * Design:
 *   - Idle: just the input + magnifying glass + ⌘K Kbd hint (right side).
 *   - Focused / expanded: two rows below the divider:
 *       Row 1 — SOURCES (Europe PMC → OpenAlex → Database) + AVAILABILITY
 *               (Open Access + Full Text — Europe PMC only)
 *       Row 2 — SORT BY (Type dropdown + Relevance/Citations/Date with
 *               inline ↑/↓ for Date)
 *   - The X clear icon (no background) appears when there's text.
 *   - On Enter: a pink trail-light streaks along the bottom inner edge
 *     before submission completes.
 *
 * Source semantics:
 *   - europepmc / openalex → submit to ``/search/json``; current PhytoQuery
 *     search flow (no backend change).
 *   - database → call ``onOpenDatabasePanel(query)`` instead of submitting;
 *     parent opens the existing DbExplorerDrawer with the query pre-filled.
 *
 * Conditional filters:
 *   - database → no Availability, no Sort, no Type — just the source pill.
 *   - europepmc → Availability + Sort + Type (Research Articles / Review).
 *   - openalex → Sort + Type (full 21-item OpenAlex vocabulary, fetched).
 */

type SourceKey = 'europepmc' | 'openalex' | 'database';
type SortType = 'Relevance' | 'Citations' | 'Date';
type SortDir = 'asc' | 'desc';

interface SearchFormProps {
  onSearch: (query: string, filters: SearchFilters) => void;
  /** Called instead of onSearch when the user submits with source=database.
   * Parent (NerPage) should open the DbExplorerDrawer with the query
   * pre-filled. Optional — if not provided, falls back to onSearch. */
  onOpenDatabasePanel?: (query: string) => void;
  isLoading?: boolean;
  defaultQuery?: string;
  defaultFilters?: SearchFilters;
}

interface ArticleTypeOption {
  key: string;
  display_name: string;
  count: number | null;
}

const EUROPEPMC_TYPES: ArticleTypeOption[] = [
  { key: '',                 display_name: 'Type',             count: null },
  { key: 'Research-article', display_name: 'Research Articles', count: null },
  { key: 'Review',           display_name: 'Review Articles',   count: null },
];

const OPENALEX_FALLBACK: ArticleTypeOption[] = [
  { key: '',                       display_name: 'Type',                  count: null },
  { key: 'article',                display_name: 'Article',               count: null },
  { key: 'preprint',               display_name: 'Preprint',              count: null },
  { key: 'review',                 display_name: 'Review',                count: null },
  { key: 'dissertation',           display_name: 'Dissertation',          count: null },
  { key: 'letter',                 display_name: 'Letter',                count: null },
  { key: 'book',                   display_name: 'Book',                  count: null },
  { key: 'book-chapter',           display_name: 'Book Chapter',          count: null },
  { key: 'erratum',                display_name: 'Erratum',               count: null },
  { key: 'editorial',              display_name: 'Editorial',             count: null },
  { key: 'paratext',               display_name: 'Paratext',              count: null },
  { key: 'reference-entry',        display_name: 'Reference Entry',       count: null },
  { key: 'report',                 display_name: 'Report',                count: null },
  { key: 'dataset',                display_name: 'Dataset',               count: null },
  { key: 'peer-review',            display_name: 'Peer Review',           count: null },
  { key: 'other',                  display_name: 'Other',                 count: null },
  { key: 'retraction',             display_name: 'Retraction',            count: null },
  { key: 'supplementary-materials',display_name: 'Supplementary Materials',count: null },
  { key: 'report-component',       display_name: 'Report Component',      count: null },
  { key: 'database',               display_name: 'Database',              count: null },
  { key: 'standard',               display_name: 'Standard',              count: null },
  { key: 'grant',                  display_name: 'Grant',                 count: null },
];

const titleCase = (s: string): string =>
  s
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');

// ─── small UI atoms ─────────────────────────────────────────────────────────

const Kbd: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <kbd
    className="
      pointer-events-none select-none inline-flex items-center justify-center
      h-6 min-w-6 px-1.5 rounded
      bg-surface-c border border-border text-on-surface-muted
      font-mono text-[14px] font-medium leading-none
    "
  >
    {children}
  </kbd>
);

interface SourcePillProps {
  role: SourceKey;
  label: string;
  count?: number;
  active: boolean;
  onClick: () => void;
}

const SourcePill: React.FC<SourcePillProps> = ({ role, label, count, active, onClick }) => {
  const tone = {
    europepmc: 'pill-europepmc',
    openalex:  'pill-openalex',
    database:  'pill-database',
  }[role];
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`
        pill ${tone}
        ${active ? 'data-[active=true]' : ''}
      `}
      data-active={active ? 'true' : 'false'}
    >
      <span>{label}</span>
      {count != null && (
        <span className="grid place-items-center min-w-[28px] h-5 px-1.5 rounded-full bg-background/70 text-[11px] font-bold">
          {count}
        </span>
      )}
    </button>
  );
};

interface FilterPillProps {
  icon: React.ReactNode;
  label: string;
  tone: 'orange' | 'blue';
  active: boolean;
  onClick: () => void;
}

const FilterPill: React.FC<FilterPillProps> = ({ icon, label, tone, active, onClick }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className="pill"
      data-active={active ? 'true' : 'false'}
      data-tone={tone}
    >
      {icon}
      <span>{label}</span>
    </button>
  );
};

interface DropdownProps {
  label: string;
  value: string;
  options: ArticleTypeOption[];
  onChange: (key: string) => void;
}

const Dropdown: React.FC<DropdownProps> = ({ label, value, options, onChange }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, [open]);
  const display = options.find((o) => o.key === value)?.display_name ?? label;
  return (
    <div ref={ref} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`
          inline-flex items-center justify-between gap-2 h-[34px] px-3 rounded-full
          ${open ? 'bg-surface-c' : 'bg-transparent'}
          border border-border text-on-surface text-[13.5px] font-medium
          transition-colors
        `}
      >
        <span>{display}</span>
        <CaretDown size={11} weight="bold" className="text-on-surface-variant" />
      </button>
      {open && (
        <div
          className="
            absolute top-[calc(100%+6px)] left-0 z-[100]
            min-w-[220px] max-h-[320px] overflow-y-auto
            bg-background border border-border rounded-xl shadow-lg p-1.5
            animate-in fade-in slide-in-from-top-1 duration-150
          "
        >
          {options.map((opt) => {
            const active = opt.key === value;
            return (
              <button
                key={opt.key || '__type__'}
                type="button"
                onClick={() => { onChange(opt.key); setOpen(false); }}
                className={`
                  w-full text-left px-3 py-2 rounded-md text-[13.5px]
                  ${active ? 'bg-surface-c font-semibold' : 'hover:bg-surface-c font-normal'}
                  text-on-surface transition-colors
                `}
              >
                {opt.display_name}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

interface SortControlProps {
  value: { type: SortType; dir: SortDir };
  onChange: (v: { type: SortType; dir: SortDir }) => void;
}

const SortControl: React.FC<SortControlProps> = ({ value, onChange }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onMouseDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onMouseDown);
    return () => document.removeEventListener('mousedown', onMouseDown);
  }, [open]);
  const isDate = value.type === 'Date';
  const Arrow = value.dir === 'asc' ? ArrowUp : ArrowDown;
  const sortOptions: SortType[] = ['Relevance', 'Citations', 'Date'];
  return (
    <div ref={ref} className="relative inline-flex">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`
          inline-flex items-center gap-1.5 h-[34px] px-3 rounded-full
          ${open ? 'bg-surface-c' : 'bg-transparent'}
          border border-border text-on-surface text-[13.5px] font-medium
          transition-colors
        `}
      >
        <span>{value.type}</span>
        {isDate && <Arrow size={12} weight="bold" />}
        <CaretDown size={11} weight="bold" className="ml-0.5 text-on-surface-muted" />
      </button>
      {open && (
        <div
          className="
            absolute top-[calc(100%+6px)] right-0 z-[100]
            min-w-[200px] max-h-[280px] overflow-y-auto
            bg-background border border-border rounded-xl shadow-lg p-1.5
            animate-in fade-in slide-in-from-top-1 duration-150
          "
        >
          {sortOptions.map((opt) => {
            const active = opt === value.type;
            return (
              <button
                key={opt}
                type="button"
                onClick={() => {
                  if (opt === 'Date' && active) { setOpen(false); return; }
                  onChange({ type: opt, dir: opt === 'Date' ? (value.dir || 'desc') : 'desc' });
                  if (opt !== 'Date') setOpen(false);
                }}
                className={`
                  flex w-full items-center justify-between
                  px-3 py-2 rounded-md text-[13.5px]
                  ${active ? 'bg-surface-c font-semibold' : 'hover:bg-surface-c font-normal'}
                  text-on-surface transition-colors
                `}
              >
                <span>{opt}</span>
                {opt === 'Date' && (
                  <span
                    role="button"
                    aria-label={value.dir === 'asc' ? 'Ascending — click for descending' : 'Descending — click for ascending'}
                    onClick={(e) => {
                      e.stopPropagation();
                      onChange({ ...value, dir: value.dir === 'asc' ? 'desc' : 'asc' });
                    }}
                    className="grid place-items-center p-1 rounded text-on-surface cursor-pointer hover:bg-surface-high"
                  >
                    <Arrow size={13} weight="bold" />
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ─── main component ─────────────────────────────────────────────────────────

const SearchForm: React.FC<SearchFormProps> = ({
  onSearch,
  onOpenDatabasePanel,
  isLoading = false,
  defaultQuery = '',
  defaultFilters,
}) => {
  const [query, setQuery] = useState(defaultQuery);
  const [filters, setFilters] = useState<SearchFilters>({
    open_access:   defaultFilters?.open_access   ?? false,
    has_full_text: defaultFilters?.has_full_text ?? false,
    article_type:  defaultFilters?.article_type  ?? '',
    sort:          defaultFilters?.sort          ?? '',
    source:        defaultFilters?.source        ?? 'europepmc',
  });

  // SortControl uses a richer { type, dir } shape — derive it from
  // filters.sort and write back through a single setter so the underlying
  // string-based contract with NerPage stays the same.
  const sortValue: { type: SortType; dir: SortDir } =
    filters.sort === 'cited'    ? { type: 'Citations', dir: 'desc' } :
    filters.sort === 'date'     ? { type: 'Date',      dir: 'desc' } :
    filters.sort === 'date_asc' ? { type: 'Date',      dir: 'asc'  } :
                                  { type: 'Relevance', dir: 'desc' };
  const setSortValue = (v: { type: SortType; dir: SortDir }) => {
    const raw =
      v.type === 'Citations' ? 'cited' :
      v.type === 'Date'      ? (v.dir === 'asc' ? 'date_asc' : 'date') :
                               '';
    setFilters((prev) => ({ ...prev, sort: raw }));
  };

  const [typeOptions, setTypeOptions] = useState<ArticleTypeOption[]>(EUROPEPMC_TYPES);
  const [typesLoading, setTypesLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [searching, setSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // ⌘K / Ctrl+K → focus & expand
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setExpanded(true);
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // Click-outside / Esc → collapse
  useEffect(() => {
    if (!expanded) return;
    const onMouseDown = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setExpanded(false);
        inputRef.current?.blur();
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setExpanded(false);
        inputRef.current?.blur();
      }
    };
    document.addEventListener('mousedown', onMouseDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [expanded]);

  // Sync from URL when navigating back
  useEffect(() => {
    setQuery(defaultQuery);
  }, [defaultQuery]);

  useEffect(() => {
    if (!defaultFilters) return;
    setFilters((prev) => {
      const same =
        prev.open_access   === (defaultFilters.open_access   ?? false) &&
        prev.has_full_text === (defaultFilters.has_full_text ?? false) &&
        prev.article_type  === (defaultFilters.article_type  ?? '')    &&
        prev.sort          === (defaultFilters.sort          ?? '')    &&
        prev.source        === (defaultFilters.source        ?? 'europepmc');
      if (same) return prev;
      return {
        open_access:   defaultFilters.open_access   ?? false,
        has_full_text: defaultFilters.has_full_text ?? false,
        article_type:  defaultFilters.article_type  ?? '',
        sort:          defaultFilters.sort          ?? '',
        source:        defaultFilters.source        ?? 'europepmc',
      };
    });
  }, [defaultFilters]);

  // Fetch types when source changes (Europe PMC = static; OpenAlex = live)
  const fetchTypeOptions = useCallback(async (source: string) => {
    if (source === 'europepmc') {
      setTypeOptions(EUROPEPMC_TYPES);
      return;
    }
    if (source === 'openalex') {
      setTypesLoading(true);
      try {
        const data = await searchTypesApi.getTypes('openalex');
        const mapped: ArticleTypeOption[] = (data.types || []).map((t) => ({
          key: t.key,
          display_name: titleCase(t.display_name || t.key),
          count: t.count ?? null,
        }));
        setTypeOptions([{ key: '', display_name: 'Type', count: null }, ...mapped]);
      } catch (err) {
        console.error('Failed to fetch OpenAlex types:', err);
        setTypeOptions(OPENALEX_FALLBACK);
      } finally {
        setTypesLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    fetchTypeOptions(filters.source);
  }, [filters.source, fetchTypeOptions]);

  // Switching source resets article_type to "" (different option set)
  const handleSourceChange = (newSource: SourceKey) => {
    setFilters((prev) => ({ ...prev, source: newSource, article_type: '' }));
  };

  const clear = () => {
    setQuery('');
    inputRef.current?.focus();
  };

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault();
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    window.setTimeout(() => {
      setExpanded(false);
      inputRef.current?.blur();
      if (filters.source === 'database' && onOpenDatabasePanel) {
        onOpenDatabasePanel(q);
      } else {
        onSearch(q, filters);
      }
      window.setTimeout(() => setSearching(false), 120);
    }, 650);
  };

  const showAvailability = expanded && filters.source === 'europepmc';
  const showSort         = expanded && filters.source !== 'database';

  return (
    <div ref={containerRef} className="relative z-0 mb-8 w-full">
      {/* Ambient halo glow — static, only active on expand (no continuous GPU hit) */}
      {expanded && (
        <div
          aria-hidden
          className="pointer-events-none absolute -inset-x-8 -top-10 -z-10 h-56 opacity-60 blur-3xl"
          style={{
            background: `
              radial-gradient(50% 60% at 25% 40%, color-mix(in oklab, var(--primary) 28%, transparent) 0%, transparent 70%),
              radial-gradient(50% 60% at 75% 30%, color-mix(in oklab, var(--cyan-500) 22%, transparent) 0%, transparent 70%)
            `,
          }}
        />
      )}
      <div
        className={`
          search-shell
          ${expanded ? 'is-focused' : ''}
          transition-[padding] duration-200
        `}
        style={{ padding: '10px 8px' }}
      >
        {/* Pink trail-light on submit — keyed by submission time so animation restarts */}
        {searching && <span className="pq-search-trail" key={Date.now()} />}

        <form onSubmit={submit}>
          {/* Row 1 — icon (left corner) | input | clear / ⌘K (right corner) */}
          <div
            role="search"
            onClick={() => {
              setExpanded(true);
              inputRef.current?.focus();
            }}
            className="flex items-center gap-2.5 transition-[padding] duration-200"
          >
            {/* Search icon — clickable, at left corner of pill */}
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                e.preventDefault();
                submit();
              }}
              aria-label="Search"
              className="
                grid place-items-center shrink-0 w-10 h-10
                text-on-surface-variant hover:text-primary
                transition-colors rounded-full
              "
            >
              <MagnifyingGlass size={22} weight="regular" />
            </button>
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => setExpanded(true)}
              placeholder="Search scholarly publications (DOI, PMID, PMCID, or keywords)…"
              autoComplete="off"
              spellCheck={false}
              className="
                flex-1 min-w-0 h-11
                bg-transparent border-none outline-none
                text-[15px] text-on-surface placeholder:text-on-surface-muted caret-teal-500
                tracking-[-0.003em]
              "
            />
            {query && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); clear(); }}
                aria-label="Clear search"
                title="Clear"
                className="grid place-items-center p-1.5 rounded-full text-on-surface-variant hover:text-on-surface bg-transparent transition-colors shrink-0"
              >
                <X size={16} weight="bold" />
              </button>
            )}
            {!expanded && !query && (
              <span className="inline-flex items-center gap-1 mr-1 shrink-0">
                <Kbd>⌘</Kbd><Kbd>K</Kbd>
              </span>
            )}
          </div>

          {/* Filter rows — animated expand/collapse via grid-rows */}
          <div
            className="grid transition-[grid-template-rows,opacity] duration-300 ease-out"
            style={{
              gridTemplateRows: expanded ? '1fr' : '0fr',
              opacity: expanded ? 1 : 0,
            }}
          >
            <div className="min-h-0 overflow-hidden">
              {/* Row 2 — SOURCES | AVAILABILITY */}
              <div
                className="
                  flex items-center gap-6 flex-wrap
                  border-t border-border mt-2.5
                  px-4 pt-3.5 pb-1.5
                  animate-spring-in
                "
              >
                <div className="flex items-center gap-3">
                  <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-on-surface-variant min-w-[78px]">
                    Sources
                  </span>
                  <div className="flex items-center gap-2 flex-wrap">
                    <SourcePill role="europepmc" label="Europe PMC"
                      active={filters.source === 'europepmc'}
                      onClick={() => handleSourceChange('europepmc')} />
                    <SourcePill role="openalex"  label="OpenAlex"
                      active={filters.source === 'openalex'}
                      onClick={() => handleSourceChange('openalex')} />
                    <SourcePill role="database"  label="Database"
                      active={filters.source === 'database'}
                      onClick={() => handleSourceChange('database')} />
                  </div>
                </div>

                {showAvailability && (
                  <>
                    <span aria-hidden className="w-px h-6 bg-surface-c" />
                    <div className="flex items-center gap-3">
                      <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-on-surface-muted">
                        Availability
                      </span>
                      <div className="flex items-center gap-2 flex-wrap">
                        <FilterPill
                          icon={filters.open_access
                            ? <LockSimpleOpen size={14} weight="fill" />
                            : <LockSimple size={14} weight="regular" />}
                          label="Open Access"
                          tone="orange"
                          active={filters.open_access}
                          onClick={() => setFilters((p) => ({ ...p, open_access: !p.open_access }))}
                        />
                        <FilterPill
                          icon={<Article size={14} weight={filters.has_full_text ? 'fill' : 'regular'} />}
                          label="Full Text"
                          tone="blue"
                          active={filters.has_full_text}
                          onClick={() => setFilters((p) => ({ ...p, has_full_text: !p.has_full_text }))}
                        />
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Row 3 — SORT BY */}
              {showSort && (
                <div
                  className="flex items-center gap-3 px-4 pb-2 animate-spring-in"
                  style={{ animationDelay: '0.05s' }}
                >
                  <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-on-surface-muted min-w-[78px]">
                    Sort by
                  </span>
                  <div className="flex items-center gap-2 flex-wrap">
                    <Dropdown
                      label="Type"
                      value={filters.article_type}
                      options={typesLoading
                        ? [{ key: '', display_name: 'Loading…', count: null }]
                        : typeOptions}
                      onChange={(k) => setFilters((p) => ({ ...p, article_type: k }))}
                    />
                    <SortControl value={sortValue} onChange={setSortValue} />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Loading shimmer when an outer fetch is in flight */}
          {isLoading && (
            <span
              aria-hidden
              className="
                absolute left-4 right-4 bottom-0 h-px rounded-full
                bg-gradient-to-r from-transparent via-primary/50 to-transparent
                animate-pulse
              "
            />
          )}
        </form>
      </div>
    </div>
  );
};

export default SearchForm;
