import React, {
  useState,
  useEffect,
  useRef,
  useMemo,
  lazy,
  Suspense,
} from 'react';
import { useNavigate, getRouteApi } from '@tanstack/react-router';
import SearchForm from './SearchForm';
import { nerApi } from '../../lib/api';
import { useSearchStore } from '../../stores/searchStore';
import { useDrawerStore } from '../../stores/drawerStore';
import { formatTextWithFormatting } from '../../utils/sanitize';
import type { SearchFilters, SearchResult } from '../../types';
import {
  CaretDown,
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  DownloadSimple,
  ChartLine,
  ChatCircle,
  Circle,
  LockSimple,
  LockSimpleOpen,
  Article,
} from '@phosphor-icons/react';

const route = getRouteApi('/');

const Dashboard = lazy(() => import('./Dashboard').then(m => ({ default: m.default })));

type SortType = 'Relevance' | 'Citations' | 'Date';
type SortDir = 'asc' | 'desc';
type SourceKey = 'europepmc' | 'openalex' | 'database';

/** Compare two SearchFilters objects for value equality. Cheap fixed-
 * shape compare — avoids pulling lodash for one call. */
function filtersEqual(a: SearchFilters | null, b: SearchFilters): boolean {
  if (!a) return false;
  return (
    a.open_access === b.open_access &&
    a.has_full_text === b.has_full_text &&
    (a.article_type ?? '') === (b.article_type ?? '') &&
    (a.sort ?? '') === (b.sort ?? '') &&
    (a.source ?? 'europepmc') === (b.source ?? 'europepmc')
  );
}

const sortFromRaw = (raw: string | undefined | null): { type: SortType; dir: SortDir } =>
  raw === 'cited'    ? { type: 'Citations', dir: 'desc' } :
  raw === 'date'     ? { type: 'Date',      dir: 'desc' } :
  raw === 'date_asc' ? { type: 'Date',      dir: 'asc'  } :
                       { type: 'Relevance', dir: 'desc' };

const sortToRaw = (v: { type: SortType; dir: SortDir }): string =>
  v.type === 'Citations' ? 'cited' :
  v.type === 'Date'      ? (v.dir === 'asc' ? 'date_asc' : 'date') :
                           '';

// ─── small UI atoms — co-located so NerPage stays a single file ─────────────

interface FilterSectionProps {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const FilterSection: React.FC<FilterSectionProps> = ({ title, defaultOpen = false, children }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`
          flex w-full items-center justify-between bg-transparent border-0 cursor-pointer
          py-1 text-[12px] font-bold uppercase tracking-[0.14em] text-on-surface
          ${open ? 'mb-3.5' : 'mb-0'}
          transition-[margin-bottom] duration-200
        `}
      >
        <span>{title}</span>
        <CaretDown
          size={12}
          weight="bold"
          className="text-on-surface-variant transition-transform duration-200"
          style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}
        />
      </button>
      {open && (
        <div className="flex flex-col gap-2.5 animate-fade-up">
          {children}
        </div>
      )}
    </div>
  );
};

interface SourcePillProps {
  role: SourceKey;
  label: string;
  active: boolean;
  onClick: () => void;
}

const SourcePill: React.FC<SourcePillProps> = ({ role, label, active, onClick }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`pill pill-${role}`}
      data-active={active ? 'true' : 'false'}
    >
      <span>{label}</span>
    </button>
  );
};

interface FilterTogglePillProps {
  icon: React.ReactNode;
  label: string;
  tone: 'orange' | 'blue';
  active: boolean;
  onClick: () => void;
}

const FilterTogglePill: React.FC<FilterTogglePillProps> = ({ icon, label, tone, active, onClick }) => {
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

interface SortSegmentedProps {
  value: { type: SortType; dir: SortDir };
  onChange: (v: { type: SortType; dir: SortDir }) => void;
}

/** Relevance | Citations | Date — plain text with thin vertical dividers.
 * Active = bold + full-contrast color. Inactive = dim. Click Date when
 * active = flip ↑/↓ direction. */
const SortSegmented: React.FC<SortSegmentedProps> = ({ value, onChange }) => {
  const options: SortType[] = ['Relevance', 'Citations', 'Date'];
  return (
    <div className="inline-flex items-center">
      {options.map((opt, i) => {
        const active = opt === value.type;
        const isDate = opt === 'Date';
        const Arrow = value.dir === 'asc' ? ArrowUp : ArrowDown;
        return (
          <React.Fragment key={opt}>
            {i > 0 && <span aria-hidden className="w-px h-3.5 mx-3.5 bg-border" />}
            <button
              type="button"
              onClick={() => {
                if (active && isDate) {
                  onChange({ ...value, dir: value.dir === 'asc' ? 'desc' : 'asc' });
                } else if (!active) {
                  onChange({ type: opt, dir: isDate ? (value.dir || 'desc') : 'desc' });
                }
              }}
              className={`
                inline-flex items-center gap-1.5 bg-transparent border-0 p-0 cursor-pointer
                text-sm transition-colors duration-150
                ${active
                  ? 'font-bold text-on-surface'
                  : 'font-medium text-on-surface-variant hover:text-on-surface'}
              `}
            >
              <span>{opt}</span>
              {isDate && active && <Arrow size={13} weight="bold" />}
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );
};

interface CircleCheckProps {
  checked: boolean;
  onToggle: () => void;
  label: string;
  count?: number | null;
}

const CircleCheck: React.FC<CircleCheckProps> = ({ checked, onToggle, label, count }) => (
  <label
    onClick={onToggle}
    className="flex items-center gap-2.5 cursor-pointer text-sm text-on-surface"
  >
    <Circle
      size={18}
      weight={checked ? 'fill' : 'regular'}
      className={checked ? 'text-primary' : 'text-on-surface-muted'}
    />
    <span className="flex-1">{label}</span>
    {count != null && (
      <span className="mono text-xs text-on-surface-muted tabular-nums">{count}</span>
    )}
  </label>
);

// ─── result card per design spec ────────────────────────────────────────────

interface ResultCardProps {
  result: SearchResult;
  onOpen: () => void;
  /** Animation delay so consecutive cards stagger in (40 + idx*60 ms). */
  delayMs: number;
}

const ResultCard: React.FC<ResultCardProps> = ({ result, onOpen, delayMs }) => {
  const year = result.year || '';
  const authors = result.authors || '';
  const journal = result.journal || '';
  const isOA = !!result.isOpenAccess;
  const hasFT = !!result.hasFullText;
  const citationCount = typeof result.citationCount === 'number' ? result.citationCount : null;

  return (
    <article
      onClick={onOpen}
      style={{ animationDelay: `${delayMs}ms` }}
      className="
        pq-result-card card is-hoverable relative px-6 py-5
        cursor-pointer
      "
    >
      {/* Top row — DOI left, year right */}
      <div className="flex items-center justify-between mb-2.5">
        {result.doi ? (
          <a
            href={`https://doi.org/${result.doi}`}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="mono text-[13px] font-medium text-primary no-underline hover:underline"
          >
            {result.doi}
          </a>
        ) : <span />}
        <span className="text-[13px] font-medium text-on-surface-variant">{year}</span>
      </div>

      {/* Title — serif, bold */}
      <h3
        className="
          mb-2.5
          font-bold text-[19px] leading-snug text-on-surface
        "
        style={{ fontFamily: 'var(--font-serif)' }}
        dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(result.title || '') }}
      />

      {/* Meta — authors • journal (italic) • Open Access (orange) • citations */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[13.5px] text-on-surface-variant mb-3.5">
        {authors && (
          <span
            dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(authors) }}
          />
        )}
        {journal && (
          <>
            <span className="text-on-surface-muted">•</span>
            <span className="italic">{journal}</span>
          </>
        )}
        {citationCount != null && citationCount > 0 && (
          <>
            <span className="text-on-surface-muted">•</span>
            <span>{citationCount} citations</span>
          </>
        )}
      </div>

      {/* Excerpt (abstract) */}
      {result.abstract && (
        <div
          className="text-sm text-on-surface-variant leading-relaxed mb-4 line-clamp-3"
          dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(result.abstract) }}
        />
      )}

      {/* Borderless action row — Download · Analyse · Chat | right: OA + FT indicators */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          <button
            type="button"
            onClick={(e) => e.stopPropagation()}
            title="Download"
            className="result-action"
          >
            <DownloadSimple size={17} weight="regular" />
            <span>Download</span>
          </button>
          <button
            type="button"
            onClick={(e) => e.stopPropagation()}
            title="Analyse"
            className="result-action"
          >
            <ChartLine size={17} weight="regular" />
            <span>Analyse</span>
          </button>
          <button
            type="button"
            onClick={(e) => e.stopPropagation()}
            title="Chat"
            className="result-action"
          >
            <ChatCircle size={17} weight="regular" />
            <span>Chat</span>
          </button>
        </div>
        {/* Access / full-text indicators — right side, matching mockup */}
        <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
          {isOA && (
            <span
              title="Open Access"
              className="grid place-items-center w-7 h-7 rounded-lg text-orange-500"
            >
              <LockSimpleOpen size={16} weight="fill" />
            </span>
          )}
          {hasFT && (
            <span
              title="Full Text Available"
              className="grid place-items-center w-7 h-7 rounded-lg text-emerald-600"
            >
              <Article size={16} weight="fill" />
            </span>
          )}
        </div>
      </div>
    </article>
  );
};

// ─── filter sidebar ─────────────────────────────────────────────────────────

const EUROPEPMC_TYPES = ['Research Articles', 'Review Articles'] as const;
// Single fallback list — full OpenAlex types (loaded statically; the live
// list is fetched into the SearchForm dropdown but for the sidebar's
// checkbox list a static set is fine and keeps the markup deterministic).
const OPENALEX_TYPES = [
  'Article', 'Preprint', 'Review', 'Dissertation', 'Letter',
  'Book', 'Book Chapter', 'Erratum', 'Editorial', 'Paratext',
  'Reference Entry', 'Report', 'Dataset', 'Peer Review', 'Other',
  'Retraction', 'Supplementary Materials', 'Report Component',
  'Database', 'Standard', 'Grant',
] as const;

interface FilterSidebarProps {
  filters: SearchFilters;
  onChange: (next: SearchFilters) => void;
}

const FilterSidebar: React.FC<FilterSidebarProps> = ({ filters, onChange }) => {
  const source = (filters.source ?? 'europepmc') as SourceKey;
  return (
    <aside
      className="
        sticky top-[100px] mt-[92px] h-fit
        flex flex-col gap-5 pr-6
        border-r border-border
      "
    >
      <div className="text-[15px] font-bold tracking-[-0.005em] text-on-surface">
        Filter by
      </div>

      <FilterSection title="Sources">
        <div className="flex flex-col gap-2 items-start">
          <SourcePill role="europepmc" label="Europe PMC"
            active={source === 'europepmc'}
            onClick={() => onChange({ ...filters, source: 'europepmc', article_type: '' })} />
          <SourcePill role="openalex" label="OpenAlex"
            active={source === 'openalex'}
            onClick={() => onChange({ ...filters, source: 'openalex', article_type: '' })} />
          <SourcePill role="database" label="Database"
            active={source === 'database'}
            onClick={() => onChange({ ...filters, source: 'database', article_type: '' })} />
        </div>
      </FilterSection>

      {source === 'europepmc' && (
        <FilterSection title="Availability">
          <div className="flex flex-col gap-2 items-start">
            <FilterTogglePill
              icon={filters.open_access
                ? <LockSimpleOpen size={14} weight="fill" />
                : <LockSimple     size={14} weight="regular" />}
              label="Open Access"
              tone="orange"
              active={!!filters.open_access}
              onClick={() => onChange({ ...filters, open_access: !filters.open_access })} />
            <FilterTogglePill
              icon={<Article size={14} weight={filters.has_full_text ? 'fill' : 'regular'} />}
              label="Full Text"
              tone="blue"
              active={!!filters.has_full_text}
              onClick={() => onChange({ ...filters, has_full_text: !filters.has_full_text })} />
          </div>
        </FilterSection>
      )}

      {source !== 'database' && (
        <FilterSection title="Type">
          {(source === 'openalex' ? OPENALEX_TYPES : EUROPEPMC_TYPES).map((t) => {
            // Map UI label back to backend key.
            const key = source === 'openalex'
              ? t.toLowerCase().replace(/\s+/g, '-')
              : t === 'Research Articles' ? 'Research-article'
              : t === 'Review Articles'   ? 'Review'
              : '';
            const checked = filters.article_type === key;
            return (
              <CircleCheck
                key={t}
                label={t}
                checked={checked}
                onToggle={() =>
                  onChange({ ...filters, article_type: checked ? '' : key })
                }
              />
            );
          })}
        </FilterSection>
      )}
    </aside>
  );
};

// ─── main page ──────────────────────────────────────────────────────────────

const NerPage: React.FC = () => {
  // ── Persisted search state (sessionStorage-backed, per-tab) ─────────────
  const results       = useSearchStore((s) => s.results);
  const pagination    = useSearchStore((s) => s.pagination);
  const currentPage   = useSearchStore((s) => s.currentPage);
  const lastQuery     = useSearchStore((s) => s.lastQuery);
  const lastFilters   = useSearchStore((s) => s.lastFilters);
  const scrollY       = useSearchStore((s) => s.scrollY);
  const setSearchResult = useSearchStore((s) => s.setSearchResult);
  const setScrollY    = useSearchStore((s) => s.setScrollY);

  // Transient (intentionally NOT persisted)
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const search = route.useSearch();
  const didInitFromUrl = useRef(false);

  // Restore on mount: if URL query matches cache, paint cached results.
  useEffect(() => {
    if (didInitFromUrl.current) return;
    didInitFromUrl.current = true;

    const q = search.q;
    if (!q) return;
    const filters: SearchFilters = {
      open_access:   search.oa === '1',
      has_full_text: search.ft === '1',
      article_type:  search.type ?? '',
      sort:          search.sort ?? '',
      source:        search.src ?? 'europepmc',
    };
    const cacheMatches =
      results.length > 0 && lastQuery === q && filtersEqual(lastFilters, filters);
    if (cacheMatches) {
      requestAnimationFrame(() => window.scrollTo({ top: scrollY }));
      return;
    }
    doSearch(q, filters, 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Track scroll position so navigating back lands where we left off.
  useEffect(() => {
    let frame: number | null = null;
    const onScroll = () => {
      if (frame !== null) return;
      frame = requestAnimationFrame(() => {
        setScrollY(window.scrollY);
        frame = null;
      });
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      if (frame !== null) cancelAnimationFrame(frame);
    };
  }, [setScrollY]);

  const doSearch = async (
    query: string,
    filters: SearchFilters,
    page: number = 1,
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await nerApi.search(query, filters, page, filters.source || 'europepmc');
      if ('error' in data && data.error) {
        throw new Error(data.error);
      }
      setSearchResult({
        results:      data.results || [],
        pagination:   data.pagination || null,
        currentPage:  page,
        lastQuery:    query,
        lastFilters:  filters,
      });
      navigate({
        to: '/',
        search: {
          q:    query,
          oa:   filters.open_access ? '1' : undefined,
          ft:   filters.has_full_text ? '1' : undefined,
          type: filters.article_type || undefined,
          sort: filters.sort || undefined,
          src:  filters.source && filters.source !== 'europepmc' ? filters.source : undefined,
        },
        replace: true,
      });
    } catch (err: any) {
      console.error('Search failed:', err);
      setError(err?.response?.data?.error || err?.response?.data?.detail || err?.message || 'Search failed. Please try again.');
      setSearchResult({
        results: [], pagination: null, currentPage: 1,
        lastQuery: query, lastFilters: filters,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async (query: string, filters: SearchFilters) => {
    // Detect identifier patterns and short-circuit to the paper viewer.
    const isIdentifier = (q: string) => {
      const trimmed = q.trim();
      return (
        /^10\.\d{4,}/.test(trimmed) ||
        /^https?:\/\/(dx\.)?doi\.org\//.test(trimmed) ||
        /^doi:/i.test(trimmed) ||
        /^PMC\d+/i.test(trimmed) ||
        /^https?:\/\/(www\.)?pubmed\.ncbi\.nlm\.nih\.gov\//.test(trimmed) ||
        /^https?:\/\/www\.ncbi\.nlm\.nih\.gov\/pmc\//.test(trimmed) ||
        /^https?:\/\/pmc\.ncbi\.nlm\.nih\.gov\/articles\//.test(trimmed) ||
        /^https?:\/\/europepmc\.org\/article\//.test(trimmed) ||
        /^\d+$/.test(trimmed)
      );
    };
    if (isIdentifier(query.trim())) {
      let cleanID = query.trim();
      if (cleanID.startsWith('http') && /(dx\.)?doi\.org\//.test(cleanID)) {
        cleanID = cleanID.replace(/^https?:\/\/(dx\.)?doi\.org\//, '');
      } else if (cleanID.toLowerCase().startsWith('doi:')) {
        cleanID = cleanID.substring(4).trim();
      } else if (/(www\.)?pubmed\.ncbi\.nlm\.nih\.gov/.test(cleanID)) {
        const m = cleanID.match(/\/(\d+)/);
        if (m) cleanID = m[1];
      } else if (/ncbi\.nlm\.nih\.gov\/pmc/.test(cleanID) || /pmc\.ncbi\.nlm\.nih\.gov/.test(cleanID)) {
        const m = cleanID.match(/PMC\d+/i);
        if (m) cleanID = m[0].toUpperCase();
      } else if (/europepmc\.org\/article\/PMC\//.test(cleanID)) {
        const m = cleanID.match(/PMC\d+/i);
        if (m) cleanID = m[0].toUpperCase();
      } else if (/europepmc\.org\/article\/MED\//.test(cleanID)) {
        const m = cleanID.match(/MED\/(\d+)/i);
        if (m) cleanID = m[1];
      }
      navigate({ to: '/paper/$doi', params: { doi: cleanID } });
      return;
    }
    doSearch(query, filters, 1);
  };

  const handleNextPage = () => {
    if (lastQuery && lastFilters && pagination?.hasMore) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      doSearch(lastQuery, lastFilters, currentPage + 1);
    }
  };

  const handlePrevPage = () => {
    if (lastQuery && lastFilters && currentPage > 1) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      doSearch(lastQuery, lastFilters, currentPage - 1);
    }
  };

  // Filter sidebar wiring — when user toggles a filter, re-run search
  // with the same query + updated filter set.
  const handleFilterChange = (next: SearchFilters) => {
    if (!lastQuery) return;
    doSearch(lastQuery, next, 1);
  };

  // Sort segmented control — also re-runs the search.
  const sortValue = useMemo(
    () => sortFromRaw(lastFilters?.sort),
    [lastFilters?.sort],
  );
  const handleSortChange = (v: { type: SortType; dir: SortDir }) => {
    if (!lastQuery || !lastFilters) return;
    const nextFilters: SearchFilters = { ...lastFilters, sort: sortToRaw(v) };
    doSearch(lastQuery, nextFilters, 1);
  };

  // ── render ────────────────────────────────────────────────────────────
  const hasResults = results.length > 0;
  const noResults  = !isLoading && !error && lastQuery && results.length === 0;
  const totalCount = pagination?.total ?? (hasResults ? results.length : 0);

  return (
    <div className="w-full px-8 pt-7 pb-10 results-page">
      <div className="mx-auto max-w-[1440px]">
        <SearchForm
          onSearch={handleSearch}
          onOpenDatabasePanel={(q) => {
            // Signal Dashboard to open its drawer with the query
            // pre-applied. The drawer lives inside Dashboard; this
            // store is the cross-page bridge (see stores/drawerStore.ts).
            useDrawerStore.getState().requestOpenWithQuery(q);
          }}
          isLoading={isLoading}
          defaultQuery={search.q ?? ''}
          defaultFilters={{
            open_access:   search.oa === '1',
            has_full_text: search.ft === '1',
            article_type:  search.type ?? '',
            sort:          search.sort ?? '',
            source:        search.src ?? 'europepmc',
          }}
        />

        {error && (
          <div className="mx-auto mb-6 max-w-4xl rounded-lg bg-red-50 p-4 text-red-700">
            {error}
          </div>
        )}

        {isLoading && !hasResults && (
          <div className="mx-auto max-w-4xl p-12 text-center">
            <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-4 border-teal-200 border-t-teal-500" />
            <p className="text-sm font-medium text-on-surface-variant">Searching publications...</p>
          </div>
        )}

        {/* No-results screen — cat-in-a-bag illustration + plain text.
            Asset lives at frontend/public/404.png (copied from the design
            bundle). Layout matches the design's results.jsx 404 branch:
            stacked center, ~70px top padding, image capped at 460px. */}
        {noResults && (
          <div className="flex flex-col items-center gap-6 px-5 pt-[70px] pb-10">
            <img
              src="/404.png"
              alt="No results found"
              className="w-full max-w-[460px] rounded-2xl"
            />
            <div className="text-center">
              <div className="mb-1.5 text-[20px] font-bold text-on-surface">
                No publications found
              </div>
              <div className="text-[14.5px] text-on-surface-variant">
                We couldn't find anything for &ldquo;{lastQuery}&rdquo;. Try different keywords.
              </div>
            </div>
          </div>
        )}

        {/* Dashboard (when nothing has been searched yet) */}
        {!lastQuery && !isLoading && !error && (
          <Suspense fallback={
            <div className="flex items-center justify-center py-12">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-teal-200 border-t-teal-500" />
            </div>
          }>
            <Dashboard />
          </Suspense>
        )}

        {/* Results view */}
        {hasResults && lastFilters && (
          <div
            className="grid gap-10 mt-1"
            style={{ gridTemplateColumns: '240px 1fr' }}
          >
            <FilterSidebar filters={lastFilters} onChange={handleFilterChange} />

            <section>
              {/* Header row — "1,247 publications found" + SortSegmented */}
              <div className="flex items-center justify-between mb-5">
                <div className="text-[15px] text-on-surface whitespace-nowrap">
                  <strong className="font-bold">{totalCount.toLocaleString()}</strong>{' '}
                  <span className="text-on-surface-variant">publications found</span>
                </div>
                <SortSegmented value={sortValue} onChange={handleSortChange} />
              </div>

              {/* Result cards */}
              <div className="flex flex-col gap-4">
                {results.map((r, i) => (
                  <ResultCard
                    key={r.id || i}
                    result={r}
                    delayMs={40 + Math.min(i, 7) * 60}
                    onOpen={() => {
                      const paperId = r.doi || r.pmcid || r.pmid;
                      if (paperId) {
                        const src = (r.source || 'Europe PMC').toLowerCase();
                        navigate({
                          to: '/paper/$doi',
                          params: { doi: paperId },
                          search: { src },
                        });
                      }
                    }}
                  />
                ))}
              </div>

              {/* Pagination — borderless arrows. Left arrow hidden on page 1. */}
              {pagination && (
                <div className="mt-8 mb-4 flex items-center justify-center gap-3">
                  {currentPage > 1 && (
                    <button
                      type="button"
                      onClick={handlePrevPage}
                      disabled={isLoading}
                      aria-label="Previous page"
                      title="Previous page"
                      className="
                        grid h-10 w-10 place-items-center rounded-full
                        bg-transparent text-on-surface border-0 cursor-pointer
                        hover:bg-surface-c transition-colors duration-150
                        disabled:opacity-40 disabled:cursor-default disabled:hover:bg-transparent
                      "
                    >
                      <ArrowLeft size={18} weight="bold" />
                    </button>
                  )}
                  {pagination.hasMore && (
                    <button
                      type="button"
                      onClick={handleNextPage}
                      disabled={isLoading}
                      aria-label="Next page"
                      title="Next page"
                      className="
                        grid h-10 w-10 place-items-center rounded-full
                        bg-transparent text-on-surface border-0 cursor-pointer
                        hover:bg-surface-c transition-colors duration-150
                        disabled:opacity-40 disabled:cursor-default disabled:hover:bg-transparent
                      "
                    >
                      <ArrowRight size={18} weight="bold" />
                    </button>
                  )}
                </div>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
};

export default NerPage;
