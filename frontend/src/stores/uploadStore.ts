/**
 * Shared upload state — read by both the chat page (RagPage.tsx) and
 * the sidebar's "Add Sources" button (Sidebar.tsx) so each surface
 * sees the same in-flight upload status, progress text, and current
 * job id.
 *
 * Why a store: before this, both components held their own
 * ``[isUploading, setIsUploading]`` and ``[uploadStatus, setStatus]``
 * pairs and ran independent ``setInterval`` polls — the user could
 * trigger an upload from one surface and the other would never
 * notice. Centralizing here is the canonical Zustand use case:
 * cross-component UI state that does not belong to any one component
 * tree.
 *
 * What does NOT live here:
 *   - The list of indexed files (server state — owned by the
 *     ``useIndexedFiles`` TanStack Query hook).
 *   - Per-component visual state (open menus, hover, etc.).
 *   - The chat thread (assistant-ui owns that).
 */
import { create } from 'zustand';

interface UploadState {
  /** True while a multipart POST is in flight OR while the indexing
   * background job is still polling. */
  isUploading: boolean;
  /** Human-readable status text rendered next to upload affordances.
   * Empty string when idle. */
  status: string;
  /** Latest backend job id we are polling. ``null`` when no job is
   * active. */
  currentJobId: string | null;

  setIsUploading: (next: boolean) => void;
  setStatus: (next: string) => void;
  setCurrentJobId: (jobId: string | null) => void;
  /** Reset the entire upload slice — useful on logout / wipe. */
  reset: () => void;
}

export const useUploadStore = create<UploadState>((set) => ({
  isUploading: false,
  status: '',
  currentJobId: null,
  setIsUploading: (next) => set({ isUploading: next }),
  setStatus: (next) => set({ status: next }),
  setCurrentJobId: (jobId) => set({ currentJobId: jobId }),
  reset: () => set({ isUploading: false, status: '', currentJobId: null }),
}));
