/**
 * Analyse page state — survives route navigation but clears on tab close.
 *
 * Persisted to ``sessionStorage`` via Zustand's ``persist`` middleware
 * (same backing as searchStore — per-tab, gone on close).
 *
 * Schema notes:
 *   - ``compareSelection`` is a ``string[]`` in the store, not a
 *     ``Set<string>``. Sets aren't JSON-serializable; storing as an
 *     array survives the persist round-trip and the consumer can build
 *     a Set on demand via ``new Set(compareSelection)`` if it needs O(1)
 *     membership checks.
 *   - ``selectedPaperId`` is a string, not the full ``UploadedPaper``
 *     object — that's the source of truth in ``papers`` and the page
 *     derives the selected paper from the id at render time.
 *   - Per-paper PDF preview state (``viewerSrc``, ``viewerError``,
 *     ``isViewerLoading``) intentionally stays in component-local
 *     state. ``viewerSrc`` is a ``blob:`` URL that's revoked when the
 *     viewer closes — persisting it would either explode the JSON
 *     storage or leave a dangling URL on the next page load.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface UploadedPaper {
  id: string;
  name: string;
  doi?: string;
  pdfUrl?: string | null;
  entities: Record<string, string[]>;
  entity_counts: Record<
    string,
    { text: string; count: number; canonical?: string; aliases?: string[] }[]
  >;
  entity_count: number;
}

interface AnalyseState {
  papers: UploadedPaper[];
  selectedPaperId: string | null;
  expandedGroups: Record<string, boolean>;
  isCompareMode: boolean;
  compareSelection: string[];

  // ── actions ────────────────────────────────────────────────────────────
  /** Prepend new papers to the list. Matches the current upload UX
   * (newest paper appears at the top of the sidebar). */
  addPapers: (papers: UploadedPaper[]) => void;
  removePaper: (id: string) => void;
  setSelectedPaperId: (id: string | null) => void;
  setExpandedGroups: (groups: Record<string, boolean>) => void;
  toggleGroup: (label: string) => void;
  setIsCompareMode: (compareMode: boolean) => void;
  setCompareSelection: (ids: string[]) => void;
  toggleCompareSelection: (id: string) => void;
  clearCompareSelection: () => void;
  resetAnalyse: () => void;
}

const INITIAL: Pick<
  AnalyseState,
  'papers' | 'selectedPaperId' | 'expandedGroups' | 'isCompareMode' | 'compareSelection'
> = {
  papers: [],
  selectedPaperId: null,
  expandedGroups: {},
  isCompareMode: false,
  compareSelection: [],
};

export const useAnalyseStore = create<AnalyseState>()(
  persist(
    (set) => ({
      ...INITIAL,

      addPapers: (newPapers) =>
        set((state) => ({ papers: [...newPapers, ...state.papers] })),

      removePaper: (id) =>
        set((state) => ({
          papers: state.papers.filter((p) => p.id !== id),
          selectedPaperId:
            state.selectedPaperId === id ? null : state.selectedPaperId,
          compareSelection: state.compareSelection.filter((s) => s !== id),
        })),

      setSelectedPaperId: (id) => set({ selectedPaperId: id }),

      setExpandedGroups: (groups) => set({ expandedGroups: groups }),

      toggleGroup: (label) =>
        set((state) => ({
          expandedGroups: {
            ...state.expandedGroups,
            [label]: !state.expandedGroups[label],
          },
        })),

      setIsCompareMode: (compareMode) => set({ isCompareMode: compareMode }),

      setCompareSelection: (ids) => set({ compareSelection: ids }),

      toggleCompareSelection: (id) =>
        set((state) => ({
          compareSelection: state.compareSelection.includes(id)
            ? state.compareSelection.filter((s) => s !== id)
            : [...state.compareSelection, id],
        })),

      clearCompareSelection: () => set({ compareSelection: [] }),

      resetAnalyse: () => set(INITIAL),
    }),
    {
      name: 'pq_analyse_state',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        papers: state.papers,
        selectedPaperId: state.selectedPaperId,
        expandedGroups: state.expandedGroups,
        isCompareMode: state.isCompareMode,
        compareSelection: state.compareSelection,
      }),
    },
  ),
);
