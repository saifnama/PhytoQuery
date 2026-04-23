import React, { useState } from 'react';
import type { SearchFilters } from '../../types';

interface SearchFormProps {
  onSearch: (query: string, filters: SearchFilters) => void;
  isLoading?: boolean;
  defaultQuery?: string;
  defaultFilters?: SearchFilters;
}

const SearchForm: React.FC<SearchFormProps> = ({ onSearch, isLoading = false, defaultQuery = '', defaultFilters }) => {
  const [query, setQuery] = useState(defaultQuery);
  const [filters, setFilters] = useState<SearchFilters>({
    open_access: defaultFilters?.open_access ?? false,
    has_full_text: defaultFilters?.has_full_text ?? false,
    article_type: defaultFilters?.article_type ?? '',
    sort: defaultFilters?.sort ?? '',
    page_size: defaultFilters?.page_size ?? 25,
  });

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
          prev.page_size === defaultFilters.page_size
        ) {
          return prev; // no change
        }
        return {
          open_access: defaultFilters.open_access ?? false,
          has_full_text: defaultFilters.has_full_text ?? false,
          article_type: defaultFilters.article_type ?? '',
          sort: defaultFilters.sort ?? '',
          page_size: defaultFilters.page_size ?? 25,
        };
      });
    }
  }, [defaultFilters]);

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

        <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-xs text-slate-500">
          <p className="font-semibold uppercase tracking-wider text-slate-600 mb-2">Advanced search examples</p>
          <div className="space-y-1">
            <p><code className="text-slate-700">"plant extract"</code> exact phrase</p>
            <p><code className="text-slate-700">turmeric AND review</code> boolean search</p>
            <p><code className="text-slate-700">(essential oil OR phytochemical) AND turmeric</code> grouped query</p>
            <p><code className="text-slate-700">AUTHOR:"Smith J"</code> fielded search</p>
          </div>
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
            className="bg-transparent border-none outline-none cursor-pointer hover:text-slate-600 transition-colors"
          >
            <option value="">Any Type</option>
            <option value="Research-article">Research Article</option>
            <option value="Review">Review</option>
          </select>
          <select
            value={filters.page_size}
            onChange={(e) =>
              setFilters({ ...filters, page_size: Number(e.target.value) })
            }
            className="bg-transparent border-none outline-none cursor-pointer hover:text-slate-600 transition-colors"
          >
            <option value={10}>10 / page</option>
            <option value={25}>25 / page</option>
            <option value={50}>50 / page</option>
            <option value={100}>100 / page</option>
          </select>
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
