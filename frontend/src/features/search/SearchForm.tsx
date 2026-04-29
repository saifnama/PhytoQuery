import React, { useState, useEffect, useCallback } from 'react';
import type { SearchFilters } from '../../types';
import { searchTypesApi } from '../../lib/api';

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
    }
  };



  return (
    <div className="saas-card max-w-4xl mx-auto p-10 mb-12">
      <form onSubmit={handleSubmit} className="space-y-8">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search scholarly publications (DOI, PMID, PMCID, or keywords)..."
            className="w-full bg-slate-50 border border-slate-200 rounded-xl py-4 px-6 focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 outline-none text-lg text-slate-900 transition-all shadow-inner"
            required
          />
          <button
            type="submit"
            disabled={isLoading}
            className="absolute right-2 top-2 bottom-2 bg-blue-600 text-white px-8 rounded-lg text-xs font-bold uppercase tracking-widest hover:bg-blue-700 transition shadow-md shadow-blue-200 disabled:opacity-50"
          >
            {isLoading ? 'Searching...' : 'Find'}
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-8 text-[11px] font-bold uppercase tracking-wider text-slate-400">
          <label className="flex items-center gap-2.5 cursor-pointer group">
            <input
              type="checkbox"
              checked={filters.open_access}
              onChange={(e) =>
                setFilters({ ...filters, open_access: e.target.checked })
              }
              className="w-4 h-4 text-blue-600 border-slate-300 rounded focus:ring-blue-500"
            />
            <span className="group-hover:text-slate-600 transition-colors">
              Open Access
            </span>
          </label>
          <label className="flex items-center gap-2.5 cursor-pointer group">
            <input
              type="checkbox"
              checked={filters.has_full_text}
              onChange={(e) =>
                setFilters({ ...filters, has_full_text: e.target.checked })
              }
              className="w-4 h-4 text-blue-600 border-slate-300 rounded focus:ring-blue-500"
            />
            <span className="group-hover:text-slate-600 transition-colors">
              Full Text
            </span>
          </label>
          <div className="h-4 w-px bg-slate-200" />
          <select
            value={filters.article_type}
            onChange={(e) =>
              setFilters({ ...filters, article_type: e.target.value })
            }
            disabled={typesLoading}
            className="bg-transparent border-none outline-none cursor-pointer hover:text-slate-600 transition-colors disabled:opacity-50"
          >
            {typeOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.display_name}
              </option>
            ))}
          </select>
          <div className="h-4 w-px bg-slate-200" />
          <select
            value={filters.source || 'europepmc'}
            onChange={(e) => handleSourceChange(e.target.value)}
            className="bg-transparent border-none outline-none cursor-pointer hover:text-slate-600 transition-colors"
          >
            <option value="europepmc">Europe PMC</option>
            <option value="openalex">OpenAlex</option>
          </select>
          <div className="h-4 w-px bg-slate-200" />
          <select
            value={filters.sort}
            onChange={(e) =>
              setFilters({ ...filters, sort: e.target.value })
            }
            className="bg-transparent border-none outline-none cursor-pointer hover:text-slate-600 transition-colors"
          >
            <option value="">Relevance</option>
            <option value="cited">Citations</option>
            <option value="date">Date</option>
          </select>
        </div>
      </form>
    </div>
  );
};

export default SearchForm;
