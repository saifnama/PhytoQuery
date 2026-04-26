import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { MagnifyingGlass } from '@phosphor-icons/react';
import SearchForm from './SearchForm';
import { nerApi } from '../../lib/api';
import { formatTextWithFormatting } from '../../utils/sanitize';
import type { SearchFilters, SearchResult } from '../../types';

const NerPage: React.FC = () => {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState<{ total: number; page: number; hasMore: boolean; pageSize: number } | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [lastQuery, setLastQuery] = useState('');
  const [lastFilters, setLastFilters] = useState<SearchFilters | null>(null);
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Restore search from URL params on mount (back navigation)
  useEffect(() => {
    const q = searchParams.get('q');
    if (q && results.length === 0 && !isLoading) {
      const filters: SearchFilters = {
        open_access: searchParams.get('oa') === '1',
        has_full_text: searchParams.get('ft') === '1',
        article_type: searchParams.get('type') || '',
        sort: searchParams.get('sort') || '',
        source: searchParams.get('src') || 'europepmc',
      };
      doSearch(q, filters, 1);
    }
  }, []);

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

      setResults(data.results || []);
      setPagination(data.pagination || null);
      setCurrentPage(page);
      setLastQuery(query);
      setLastFilters(filters);

      // Persist search query in URL for back navigation
      const params = new URLSearchParams();
      params.set('q', query);
      if (filters.open_access) params.set('oa', '1');
      if (filters.has_full_text) params.set('ft', '1');
      if (filters.article_type) params.set('type', filters.article_type);
      if (filters.sort) params.set('sort', filters.sort);
      if (filters.source && filters.source !== 'europepmc') params.set('src', filters.source);
      navigate(`/?${params.toString()}`, { replace: true });
    } catch (err: any) {
      console.error('Search failed:', err);
      setError(err?.response?.data?.error || err?.response?.data?.detail || err?.message || 'Search failed. Please try again.');
      setResults([]);
      setPagination(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async (query: string, filters: SearchFilters) => {
    // Detect if query is an identifier pattern (DOI, PMCID, PMID, or URLs)
    const isIdentifier = (q: string) => {
      const trimmed = q.trim();
      return (
        /^10\.\d{4,}/.test(trimmed) || // DOI
        /^https?:\/\/(dx\.)?doi\.org\//.test(trimmed) || // DOI URL
        /^doi:/i.test(trimmed) || // doi: prefix
        /^PMC\d+/i.test(trimmed) || // PMCID
        /^https?:\/\/(www\.)?pubmed\.ncbi\.nlm\.nih\.gov\//.test(trimmed) || // PubMed URL
        /^https?:\/\/www\.ncbi\.nlm\.nih\.gov\/pmc\//.test(trimmed) || // NCBI PMC URL
        /^https?:\/\/pmc\.ncbi\.nlm\.nih\.gov\/articles\//.test(trimmed) || // PMC NCBI URL
        /^https?:\/\/europepmc\.org\/article\//.test(trimmed) || // Europe PMC URL
        /^\d+$/.test(trimmed) // Pure number (PMID)
      );
    };

    // If it looks like an identifier, navigate directly to the paper
    if (isIdentifier(query.trim())) {
      let cleanID = query.trim();
      // DOI URL
      if (cleanID.startsWith('http') && /(dx\.)?doi\.org\//.test(cleanID)) {
        cleanID = cleanID.replace(/^https?:\/\/(dx\.)?doi\.org\//, '');
      } else if (cleanID.toLowerCase().startsWith('doi:')) {
        cleanID = cleanID.substring(4).trim();
      } else if (/(www\.)?pubmed\.ncbi\.nlm\.nih\.gov/.test(cleanID)) {
        // Extract PMID from PubMed URL
        const m = cleanID.match(/\/(\d+)/);
        if (m) cleanID = m[1];
      } else if (/ncbi\.nlm\.nih\.gov\/pmc/.test(cleanID) || /pmc\.ncbi\.nlm\.nih\.gov/.test(cleanID)) {
        // Extract PMCID from NCBI URL
        const m = cleanID.match(/PMC\d+/i);
        if (m) cleanID = m[0].toUpperCase();
      } else if (/europepmc\.org\/article\/PMC\//.test(cleanID)) {
        const m = cleanID.match(/PMC\d+/i);
        if (m) cleanID = m[0].toUpperCase();
      } else if (/europepmc\.org\/article\/MED\//.test(cleanID)) {
        const m = cleanID.match(/MED\/(\d+)/i);
        if (m) cleanID = m[1];
      }
      navigate(`/paper/${encodeURIComponent(cleanID)}`);
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

  return (
    <div className="w-full py-12 px-0">
      <SearchForm
        onSearch={handleSearch}
        isLoading={isLoading}
        defaultQuery={searchParams.get('q') || ''}
        defaultFilters={{
          open_access: searchParams.get('oa') === '1',
          has_full_text: searchParams.get('ft') === '1',
          article_type: searchParams.get('type') || '',
          sort: searchParams.get('sort') || '',
          source: searchParams.get('src') || 'europepmc',
        }}
      />

      {/* Results Area */}
      <div id="ner-result-container" className="w-full">
        {error && (
          <div className="saas-card p-4 max-w-4xl mx-auto mb-6 bg-red-50 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {isLoading && (
          <div className="saas-card p-12 text-center max-w-4xl mx-auto">
            <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-sm text-slate-500 font-medium">Searching publications...</p>
          </div>
        )}

        {results.length === 0 && !isLoading && !error && (
          <div
            onClick={() => document.querySelector<HTMLInputElement>('input[type="text"]')?.focus()}
            className="saas-card p-20 text-center text-slate-400 border-dashed border-2 cursor-text hover:border-blue-300 hover:bg-blue-50/30 transition-colors"
          >
            <div className="w-16 h-16 bg-slate-50 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <MagnifyingGlass size={32} className="text-slate-200" weight="thin" />
            </div>
            <p className="text-sm font-medium">
              Start by searching above or entering a DOI
            </p>
          </div>
        )}

        {results.length > 0 && pagination && (
          <div className="max-w-4xl mx-auto mb-4 px-2">
            <span className="text-xs text-slate-500 font-medium">
              Showing {pagination.total.toLocaleString()} results
            </span>
          </div>
        )}

        {results.length > 0 && (
          <>
            <div className="space-y-4 max-w-4xl mx-auto">
              {results.map((result) => (
                  <div
                    key={result.id}
                    onClick={() => {
                      const paperId = result.doi || result.pmcid || result.pmid;
                      if (paperId) {
                        // Always pass source in URL - convert to lowercase
                        const src = (result.source || 'Europe PMC').toLowerCase();
                        const params = new URLSearchParams();
                        params.set('src', src);
                        navigate(`/paper/${encodeURIComponent(paperId)}?${params.toString()}`);
                      }
                    }}
                  className="saas-card p-6 hover:shadow-md transition-shadow cursor-pointer"
                >
                  <h3 
                    className="text-lg font-semibold text-slate-900 mb-2 title-font"
                    dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(result.title) }}
                  />
                  <p 
                    className="text-sm text-slate-600 mb-2"
                    dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(result.authors) }}
                  />
                  <div className="flex items-center gap-4 text-xs text-slate-400">
                    <span>{result.journal}</span>
                    <span>•</span>
                    <span>{result.year}</span>
                    {result.isOpenAccess && (
                      <span className="text-green-600">Open Access</span>
                    )}
                    {result.citationCount !== undefined && result.citationCount > 0 && (
                      <>
                        <span>•</span>
                        <span>{result.citationCount} citations</span>
                      </>
                    )}
                    {result.doi && (
                      <>
                        <span>•</span>
                        <a
                          href={`https://doi.org/${result.doi}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:underline"
                        >
                          {result.doi}
                        </a>
                      </>
                    )}
                  </div>
                  {result.abstract && (
                    <p 
                      className="text-sm text-slate-500 mt-3 line-clamp-3"
                      dangerouslySetInnerHTML={{ __html: formatTextWithFormatting(result.abstract) }}
                    />
                  )}
                </div>
              ))}
            </div>
            {pagination && pagination.hasMore && (
              <div className="flex items-center justify-center gap-3 pt-8 pb-4">
                <div className="flex items-center gap-2">
                  {currentPage > 1 && (
                    <button
                      onClick={handlePrevPage}
                      disabled={isLoading}
                      className="px-4 py-2 text-xs font-bold uppercase tracking-wider text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600 transition-all disabled:opacity-50"
                    >
                      Previous
                    </button>
                  )}
                  <button
                    onClick={handleNextPage}
                    disabled={isLoading}
                    className="px-4 py-2 text-xs font-bold uppercase tracking-wider text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600 transition-all disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default NerPage;
