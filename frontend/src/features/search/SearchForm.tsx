import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MagnifyingGlass, X, Check, CaretDown } from '@phosphor-icons/react';
import type { SearchFilters } from '../../types';
import { searchTypesApi } from '../../lib/api';

// ── Local UI atoms ─────────────────────────────────────────────────────────

interface SourceChipProps {
  checked: boolean;
  onClick: () => void;
  dotClass: string;   // tailwind bg-* for the colored dot
  tintBg: string;     // tailwind bg-* for the active chip pill
  tintBorder: string; // tailwind border-* for the active chip pill
  children: React.ReactNode;
}

const SourceChip: React.FC<SourceChipProps> = ({ checked, onClick, dotClass, tintBg, tintBorder, children }) => (
  <button
    type="button"
    onClick={onClick}
    aria-pressed={checked}
    className={`
      group inline-flex items-center gap-2.5
      pl-1.5 pr-4 py-1.5 rounded-full border
      transition-all duration-200
      ${checked
        ? `${tintBg} ${tintBorder} shadow-[0_1px_2px_rgba(15,23,42,0.04)]`
        : 'bg-white border-slate-200/70 hover:border-slate-300'}
    `}
  >
    <span
      aria-hidden
      className={`
        flex items-center justify-center w-5 h-5 rounded-md
        transition-all duration-150
        ${checked
          ? 'bg-blue-600 text-white shadow-[0_1px_2px_rgba(37,99,235,0.35)]'
          : 'bg-white border border-slate-300'}
      `}
    >
      {checked && <Check size={12} weight="bold" />}
    </span>
    <span aria-hidden className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
    <span className={`text-[13.5px] font-medium ${checked ? 'text-slate-800' : 'text-slate-500'}`}>
      {children}
    </span>
  </button>
);

interface RoundCheckProps {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
}

const RoundCheck: React.FC<RoundCheckProps> = ({ checked, onChange, label }) => (
  <label className="inline-flex items-center gap-2.5 cursor-pointer group select-none">
    <input
      type="checkbox"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      className="sr-only peer"
    />
    <span
      aria-hidden
      className={`
        relative w-[18px] h-[18px] rounded-full border-2
        transition-all duration-150
        ${checked
          ? 'border-blue-600 bg-white'
          : 'border-slate-300 group-hover:border-slate-400 bg-white'}
        peer-focus-visible:ring-2 peer-focus-visible:ring-blue-200
      `}
    >
      {checked && (
        <span className="absolute inset-[3px] rounded-full bg-blue-600" />
      )}
    </span>
    <span className="text-[14px] text-slate-600 group-hover:text-slate-800 transition-colors">
      {label}
    </span>
  </label>
);

interface PillSelectProps {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  disabled?: boolean;
  minWidth?: number;
}

const PillSelect: React.FC<PillSelectProps> = ({ value, onChange, options, disabled, minWidth }) => (
  <div className="relative inline-block">
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      style={minWidth ? { minWidth } : undefined}
      className="
        appearance-none cursor-pointer
        bg-white border border-slate-200/80
        rounded-full pl-4 pr-9 py-2
        text-[13.5px] text-slate-600
        hover:border-slate-300 hover:text-slate-800
        focus:outline-none focus:border-blue-300 focus:ring-2 focus:ring-blue-100
        disabled:opacity-50 disabled:cursor-not-allowed
        transition-colors
      "
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
    <CaretDown
      size={11}
      weight="bold"
      className="absolute right-3.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
    />
  </div>
);

interface SearchFormProps {
  onSearch: (query: string, filters: SearchFilters) => void;
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
  { key: '', display_name: 'Any Type', count: null },
  { key: 'Research-article', display_name: 'Research Article', count: null },
  { key: 'Review', display_name: 'Review', count: null },
];

const SearchForm: React.FC<SearchFormProps> = ({ onSearch, isLoading = false, defaultQuery = '', defaultFilters }) => {
  const [query, setQuery] = useState(defaultQuery);
  const [filters, setFilters] = useState<SearchFilters>({
    open_access: defaultFilters?.open_access ?? false,
    has_full_text: defaultFilters?.has_full_text ?? false,
    article_type: defaultFilters?.article_type ?? '',
    sort: defaultFilters?.sort ?? '',
    source: defaultFilters?.source ?? 'europepmc',
  });
  const [typeOptions, setTypeOptions] = useState<ArticleTypeOption[]>(EUROPEPMC_TYPES);
  const [typesLoading, setTypesLoading] = useState(false);
  const [focused, setFocused] = useState(false);
  const [expanded, setExpanded] = useState(false);
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

  // Click-outside + Esc → collapse
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

  // Sync query from URL when navigating back
  React.useEffect(() => {
    setQuery(defaultQuery);
  }, [defaultQuery]);

  // Sync filters from URL when navigating back
  React.useEffect(() => {
    if (defaultFilters) {
      setFilters(prev => {
        if (
          prev.open_access === defaultFilters.open_access &&
          prev.has_full_text === defaultFilters.has_full_text &&
          prev.article_type === defaultFilters.article_type &&
          prev.sort === defaultFilters.sort &&
          prev.source === defaultFilters.source
        ) {
          return prev; // no change
        }
        return {
          open_access: defaultFilters.open_access ?? false,
          has_full_text: defaultFilters.has_full_text ?? false,
          article_type: defaultFilters.article_type ?? '',
          sort: defaultFilters.sort ?? '',
          source: defaultFilters.source ?? 'europepmc',
        };
      });
    }
  }, [defaultFilters]);

  // Fetch article types when source changes
  const fetchTypeOptions = useCallback(async (source: string) => {
    if (source === 'europepmc') {
      setTypeOptions(EUROPEPMC_TYPES);
      return;
    }
    if (source === 'openalex') {
      setTypesLoading(true);
      try {
        const data = await searchTypesApi.getTypes('openalex');
        // Prepend "Any Type" option
        const options = [{ key: '', display_name: 'All Types', count: null }, ...(data.types || [])];
        setTypeOptions(options);
      } catch (e) {
        console.error('Failed to fetch OpenAlex types:', e);
        // Fallback static list
        setTypeOptions([
          { key: '', display_name: 'All Types', count: null },
          { key: 'article', display_name: 'Article', count: null },
          { key: 'review', display_name: 'Review', count: null },
          { key: 'preprint', display_name: 'Preprint', count: null },
          { key: 'book-chapter', display_name: 'Book Chapter', count: null },
          { key: 'dataset', display_name: 'Dataset', count: null },
          { key: 'dissertation', display_name: 'Dissertation', count: null },
          { key: 'book', display_name: 'Book', count: null },
          { key: 'other', display_name: 'Other', count: null },
          { key: 'paratext', display_name: 'Paratext', count: null },
          { key: 'libguides', display_name: 'Libguides', count: null },
          { key: 'letter', display_name: 'Letter', count: null },
          { key: 'report', display_name: 'Report', count: null },
          { key: 'peer-review', display_name: 'Peer Review', count: null },
          { key: 'reference-entry', display_name: 'Reference Entry', count: null },
          { key: 'editorial', display_name: 'Editorial', count: null },
          { key: 'erratum', display_name: 'Erratum', count: null },
          { key: 'standard', display_name: 'Standard', count: null },
          { key: 'supplementary-materials', display_name: 'Supplementary Materials', count: null },
          { key: 'retraction', display_name: 'Retraction', count: null },
          { key: 'software', display_name: 'Software', count: null },
          { key: 'database', display_name: 'Database', count: null },
          { key: 'book-section', display_name: 'Book Section', count: null },
          { key: 'report-component', display_name: 'Report Component', count: null },
          { key: 'grant', display_name: 'Grant', count: null },
        ]);
      } finally {
        setTypesLoading(false);
      }
    }
  }, []);

  // Fetch types on initial mount and when source changes
  useEffect(() => {
    fetchTypeOptions(filters.source);
  }, [filters.source, fetchTypeOptions]);

  // Reset article_type to empty when source changes
  const handleSourceChange = (newSource: string) => {
    setFilters(prev => ({
      ...prev,
      source: newSource,
      article_type: '', // Reset to "All Types" / "Any Type"
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim(), filters);
      setExpanded(false);
      inputRef.current?.blur();
    }
  };



  return (
    <div ref={containerRef} className="max-w-4xl mx-auto mb-12">
      <div
        className={`
          rounded-2xl transition-all duration-300 ease-out
          ${expanded
            ? 'bg-white border border-blue-300/70 shadow-[0_0_0_4px_rgba(37,99,235,0.06),0_10px_30px_-12px_rgba(15,23,42,0.14)] p-5 sm:p-6'
            : 'bg-transparent border border-transparent shadow-none p-0'}
        `}
      >
        <form onSubmit={handleSubmit}>
          {/* ── Search pill ───────────────────────────────────────── */}
          <div
            role="search"
            onClick={() => {
              setExpanded(true);
              inputRef.current?.focus();
            }}
            className={`
              relative flex items-center gap-3
              w-full rounded-2xl
              bg-white border
              pl-5 pr-2 py-2
              transition-all duration-200
              ${expanded
                ? 'border-slate-200/70'
                : 'border-slate-200/60 shadow-[0_2px_14px_-4px_rgba(15,23,42,0.08)] hover:shadow-[0_4px_18px_-4px_rgba(15,23,42,0.10)]'}
            `}
          >
            <MagnifyingGlass
              size={20}
              weight="regular"
              className={`shrink-0 transition-colors duration-200 ${focused ? 'text-slate-500' : 'text-slate-400'}`}
            />

            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => { setFocused(true); setExpanded(true); }}
              onBlur={() => setFocused(false)}
              placeholder="Search scholarly publications (DOI, PMID, PMCID, or keywords)..."
              required
              autoComplete="off"
              spellCheck={false}
              className="
                flex-1 min-w-0
                bg-transparent border-none outline-none
                text-[15px] leading-tight text-slate-800
                placeholder:text-slate-400
                caret-blue-600
              "
            />

            {query && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setQuery('');
                  inputRef.current?.focus();
                }}
                aria-label="Clear search"
                className="
                  shrink-0 flex items-center justify-center
                  w-6 h-6 rounded-full
                  text-slate-400 hover:text-slate-700
                  bg-transparent hover:bg-slate-200/70
                  transition-colors duration-150
                "
              >
                <X size={11} weight="bold" />
              </button>
            )}

            <button
              type="submit"
              disabled={isLoading}
              className="
                shrink-0 inline-flex items-center justify-center
                h-10 px-7 rounded-full
                bg-blue-600 text-white
                text-[11.5px] font-bold uppercase tracking-[0.14em]
                hover:bg-blue-700 active:translate-y-px
                disabled:opacity-60 disabled:cursor-not-allowed
                shadow-[0_4px_12px_-3px_rgba(37,99,235,0.45)]
                transition-all duration-200
              "
            >
              {isLoading ? (
                <span className="inline-block w-3.5 h-3.5 border-[1.5px] border-white/40 border-t-white rounded-full animate-spin" />
              ) : (
                'Find'
              )}
            </button>

            {isLoading && (
              <span
                aria-hidden
                className="
                  absolute left-4 right-4 bottom-0
                  h-px rounded-full
                  bg-gradient-to-r from-transparent via-blue-500/50 to-transparent
                  animate-pulse
                "
              />
            )}
          </div>

          {/* ── Collapsible filters ───────────────────────────────── */}
          <div
            aria-hidden={!expanded}
            className={`
              overflow-hidden
              transition-[max-height,opacity,margin] duration-300 ease-out
              ${expanded ? 'max-h-[500px] opacity-100 mt-6' : 'max-h-0 opacity-0 mt-0'}
            `}
          >
            <div className="space-y-6">
              <div className="space-y-3">
                <div className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-400">
                  Search in
                </div>
                <div className="flex flex-wrap items-center gap-3">
                  <SourceChip
                    checked={filters.source === 'europepmc'}
                    onClick={() => handleSourceChange('europepmc')}
                    dotClass="bg-emerald-500"
                    tintBg="bg-emerald-50"
                    tintBorder="border-emerald-200"
                  >
                    Europe PMC
                  </SourceChip>
                  <SourceChip
                    checked={filters.source === 'openalex'}
                    onClick={() => handleSourceChange('openalex')}
                    dotClass="bg-orange-500"
                    tintBg="bg-orange-50"
                    tintBorder="border-orange-200"
                  >
                    OpenAlex
                  </SourceChip>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-x-7 gap-y-3 pt-1">
                <RoundCheck
                  checked={filters.open_access}
                  onChange={(v) => setFilters({ ...filters, open_access: v })}
                  label="Open Access"
                />
                <RoundCheck
                  checked={filters.has_full_text}
                  onChange={(v) => setFilters({ ...filters, has_full_text: v })}
                  label="Full Text"
                />
                <PillSelect
                  value={filters.article_type}
                  onChange={(v) => setFilters({ ...filters, article_type: v })}
                  disabled={typesLoading}
                  minWidth={130}
                  options={typeOptions.map((o) => ({ value: o.key, label: o.display_name }))}
                />
                <PillSelect
                  value={filters.sort}
                  onChange={(v) => setFilters({ ...filters, sort: v })}
                  minWidth={130}
                  options={[
                    { value: '', label: 'Relevance' },
                    { value: 'cited', label: 'Citations' },
                    { value: 'date', label: 'Date' },
                  ]}
                />
              </div>

              <div className="flex justify-end pt-1">
                <span className="text-[11.5px] text-slate-400 inline-flex items-center gap-1.5">
                  Press
                  <kbd className="
                    inline-flex items-center px-1.5 py-0.5
                    text-[10px] font-mono leading-none
                    bg-slate-100 text-slate-600
                    border border-slate-200/80 rounded
                  ">Esc</kbd>
                  or click outside to close filters
                </span>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SearchForm;
