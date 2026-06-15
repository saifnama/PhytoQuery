/**
 * Search page state — survives route navigation but clears on tab close.
 *
 * Persisted to ``sessionStorage`` via Zustand's ``persist`` middleware.
 * Why sessionStorage (not localStorage):
 *   - per-tab — two tabs don't bleed search results into each other
 *   - cleared on tab/browser close — matches the product's "fresh
 *     session each open" expectation, no GDPR'able long-lived state
 *
 * Why we still keep the query / filters in the URL (in addition to
 * here): URL is the canonical shareable state. The store is the
 * "warm cache" — when the user navigates back to ``/`` after visiting
 * Analyse or Chat, the store's results map back to what the URL says
 * and we skip an avoidable re-fetch. If the URL changes (user edits
 * params, opens a shared link), the URL wins and the store re-fills.
 *
 * What is NOT persisted: ``isLoading``, ``error`` — transient UI flags.
 * Loading state across navigation would lock the page on a stale
 * spinner forever.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { SearchFilters, SearchResult } from '../types';

export interface SearchPagination {
  total: number;
  page: number;
  hasMore: boolean;
  pageSize: number;
}

interface SearchState {
  results: SearchResult[];
  pagination: SearchPagination | null;
  currentPage: number;
  lastQuery: string;
  lastFilters: SearchFilters | null;
  /** Last known scroll position of the results container. Restored on
   * mount so navigation back to /search lands where you left off. */
  scrollY: number;

  /** Bundle-assign a successful fetch result. Replaces results +
   * pagination + currentPage + lastQuery + lastFilters atomically so
   * the consuming component doesn't render an in-between state. */
  setSearchResult: (args: {
    results: SearchResult[];
    pagination: SearchPagination | null;
    currentPage: number;
    lastQuery: string;
    lastFilters: SearchFilters;
  }) => void;
  setScrollY: (y: number) => void;
  /** Clear all persisted search state — used by the "Reset" affordance
   * and from external callers like a global "clean session" button. */
  resetSearch: () => void;
}

const INITIAL: Omit<
  SearchState,
  'setSearchResult' | 'setScrollY' | 'resetSearch'
> = {
  results: [],
  pagination: null,
  currentPage: 1,
  lastQuery: '',
  lastFilters: null,
  scrollY: 0,
};

export const useSearchStore = create<SearchState>()(
  persist(
    (set) => ({
      ...INITIAL,

      setSearchResult: ({
        results,
        pagination,
        currentPage,
        lastQuery,
        lastFilters,
      }) =>
        set({
          results,
          pagination,
          currentPage,
          lastQuery,
          lastFilters,
        }),

      setScrollY: (y) => set({ scrollY: y }),

      resetSearch: () => set(INITIAL),
    }),
    {
      name: 'pq_search_state',
      storage: createJSONStorage(() => sessionStorage),
      // ``partialize`` is the safe boundary between "what's in the store
      // at runtime" and "what gets written to sessionStorage". If a
      // future field is transient (Date, function, AbortController),
      // exclude it here to avoid blowing up the JSON serializer.
      partialize: (state) => ({
        results: state.results,
        pagination: state.pagination,
        currentPage: state.currentPage,
        lastQuery: state.lastQuery,
        lastFilters: state.lastFilters,
        scrollY: state.scrollY,
      }),
    },
  ),
);
