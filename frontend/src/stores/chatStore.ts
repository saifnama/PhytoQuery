/**
 * Chat page UI state — survives route navigation but clears on tab close.
 *
 * Persisted to ``sessionStorage`` via Zustand's ``persist`` middleware
 * (same backing as searchStore / analyseStore — per-tab, gone on close).
 *
 * What lives here:
 *   - ``parserType`` — user's PDF parser preference (pymupdf vs docling)
 *   - ``uploadedFiles`` — the per-user source list (merged from server +
 *     local optimistic adds); selection checkboxes live in this struct
 *   - ``sidebarCollapsed`` — chat sidebar collapsed/expanded
 *
 * What does NOT live here:
 *   - The chat thread itself — owned by assistant-ui's
 *     ``createSessionHistoryAdapter`` in ``./assistant/runtime.ts``
 *     (also sessionStorage, separate key ``pq_chat_history``).
 *   - In-flight upload progress — that's ``uploadStore.ts``.
 *   - The "currently-viewed PDF" pane (``activePdfFile``,
 *     ``activePdfUrl``) — ``activePdfUrl`` is a transient blob/HTTP URL,
 *     and the pane should re-open from a fresh click anyway.
 *   - The active citation popup — ephemeral hover-like UI.
 *
 * ``setUploadedFiles`` accepts either a value OR a ``(prev) => next``
 * function — same shape as React's ``useState`` setter — so the
 * existing call sites in RagPage migrate without changing their
 * functional-update patterns.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export interface UploadedFile {
  name: string;
  fileType: string;
  chunkCount: number;
  selected: boolean;
  parserType: 'pymupdf' | 'docling';
  authors?: string;
  doi?: string;
  journal?: string;
  summary?: string;
}

type Updater<T> = T | ((prev: T) => T);

interface ChatState {
  parserType: 'pymupdf' | 'docling';
  uploadedFiles: UploadedFile[];
  sidebarCollapsed: boolean;

  setParserType: (next: 'pymupdf' | 'docling') => void;
  setUploadedFiles: (next: Updater<UploadedFile[]>) => void;
  setSidebarCollapsed: (next: Updater<boolean>) => void;
  /** Wipe the uploadedFiles slice — used by RagPage's "Reset all"
   * handler. Other slices keep their values (parser preference, sidebar
   * collapsed) since those are pure UI preferences, not session data. */
  resetUploadedFiles: () => void;
}

const INITIAL: Pick<ChatState, 'parserType' | 'uploadedFiles' | 'sidebarCollapsed'> = {
  parserType: 'pymupdf',
  uploadedFiles: [],
  sidebarCollapsed: false,
};

function applyUpdater<T>(next: Updater<T>, prev: T): T {
  return typeof next === 'function' ? (next as (p: T) => T)(prev) : next;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      ...INITIAL,

      setParserType: (next) => set({ parserType: next }),

      setUploadedFiles: (next) =>
        set((state) => ({ uploadedFiles: applyUpdater(next, state.uploadedFiles) })),

      setSidebarCollapsed: (next) =>
        set((state) => ({ sidebarCollapsed: applyUpdater(next, state.sidebarCollapsed) })),

      resetUploadedFiles: () => set({ uploadedFiles: [] }),
    }),
    {
      name: 'pq_chat_state',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        parserType: state.parserType,
        uploadedFiles: state.uploadedFiles,
        sidebarCollapsed: state.sidebarCollapsed,
      }),
    },
  ),
);
