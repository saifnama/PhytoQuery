import React, { useState, useEffect, useRef, lazy, Suspense } from 'react';
import { useNavigate, getRouteApi } from '@tanstack/react-router';
import SearchForm from './SearchForm';
import { nerApi } from '../../lib/api';
import { useSearchStore } from '../../stores/searchStore';

const route = getRouteApi('/');

const Dashboard = lazy(() => import('./Dashboard').then(m => ({ default: m.default })));
import { formatTextWithFormatting } from '../../utils/sanitize';
import type { SearchFilters } from '../../types';

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

const NerPage: React.FC = () => {
  // ── Persisted search state (sessionStorage-backed, per-tab) ─────────────
  const results = useSearchStore((s) => s.results);
  const pagination = useSearchStore((s) => s.pagination);
  const currentPage = useSearchStore((s) => s.currentPage);
  const lastQuery = useSearchStore((s) => s.lastQuery);
  const lastFilters = useSearchStore((s) => s.lastFilters);
  const scrollY = useSearchStore((s) => s.scrollY);
  const setSearchResult = useSearchStore((s) => s.setSearchResult);
  const setScrollY = useSearchStore((s) => s.setScrollY);
  // ── Transient (NOT persisted — would lock the page on stale spinner) ────
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const search = route.useSearch();
  const didInitFromUrl = useRef(false);

  // Restore on mount: if URL query matches what's cached, no fetch —
  // paint the cached results and restore scroll. Otherwise fetch.
  useEffect(() => {
    if (didInitFromUrl.current) return;
    didInitFromUrl.current = true;

    const q = search.q;
    if (!q) return;
    const filters: SearchFilters = {
      open_access: search.oa === '1',
      has_full_text: search.ft === '1',
      article_type: search.type ?? '',
      sort: search.sort ?? '',
      source: search.src ?? 'europepmc',
    };
    const cacheMatches =
      results.length > 0 && lastQuery === q && filtersEqual(lastFilters, filters);
    if (cacheMatches) {
      // Restore scroll on next paint — wait for the result list to be
      // measured before we try to set window.scrollY.
      requestAnimationFrame(() => window.scrollTo({ top: scrollY }));
      return;
    }
    doSearch(q, filters, 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Track scroll position so navigating back lands where we left off.
  // Throttled via rAF to avoid hammering the store on every wheel tick.
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
        results: data.results || [],
        pagination: data.pagination || null,
        currentPage: page,
        lastQuery: query,
        lastFilters: filters,
      });

      // Persist search query in URL for back navigation.
      // TanStack `search` is a typed object; omitted keys disappear from URL.
      navigate({
        to: '/',
        search: {
          q: query,
          oa: filters.open_access ? '1' : undefined,
          ft: filters.has_full_text ? '1' : undefined,
          type: filters.article_type || undefined,
          sort: filters.sort || undefined,
          src: filters.source && filters.source !== 'europepmc' ? filters.source : undefined,
        },
        replace: true,
      });
    } catch (err: any) {
      console.error('Search failed:', err);
      setError(err?.response?.data?.error || err?.response?.data?.detail || err?.message || 'Search failed. Please try again.');
      setSearchResult({
        results: [],
        pagination: null,
        currentPage: 1,
        lastQuery: query,
        lastFilters: filters,
      });
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
      // TanStack params are encoded automatically — don't double-encode.
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

  return (
    <div className="w-full pt-6 pb-4 px-0">
      <SearchForm
        onSearch={handleSearch}
        isLoading={isLoading}
        defaultQuery={search.q ?? ''}
        defaultFilters={{
          open_access: search.oa === '1',
          has_full_text: search.ft === '1',
          article_type: search.type ?? '',
          sort: search.sort ?? '',
          source: search.src ?? 'europepmc',
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
          <Suspense fallback={
            <div className="flex items-center justify-center py-12">
              <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
            </div>
          }>
            <Dashboard />
          </Suspense>
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
                        navigate({
                          to: '/paper/$doi',
                          params: { doi: paperId },
                          search: { src },
                        });
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
