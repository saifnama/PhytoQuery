/**
 * Server state for the user's indexed RAG files.
 *
 * Wraps ``ragApi.listFiles`` in a TanStack Query so:
 *   - Multiple components asking for the list share one in-flight
 *     request (deduplication).
 *   - Stale data renders immediately while a background refetch
 *     keeps it fresh — no "loading…" flash on every navigation.
 *   - Cache invalidation after upload is one call
 *     (``queryClient.invalidateQueries({ queryKey: indexedFilesKey })``)
 *     instead of every component re-running its own fetch.
 *
 * Replaces the per-component ``loadIndexedFiles`` callback that used
 * to do ``ragApi.listFiles().then(setUploadedFiles)`` — that pattern
 * fired on every mount, didn't share between siblings, and had no
 * cache so a second tab/page hop re-downloaded the same list.
 */
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ragApi } from '../lib/api';
import type { IndexedFileInfo } from '../types';

/**
 * Stable cache key. Exported so callers can invalidate after a
 * mutation — e.g. after an upload completes:
 *
 *   queryClient.invalidateQueries({ queryKey: indexedFilesKey });
 */
export const indexedFilesKey = ['rag', 'indexed-files'] as const;

export function useIndexedFiles() {
  return useQuery<IndexedFileInfo[]>({
    queryKey: indexedFilesKey,
    queryFn: () => ragApi.listFiles(),
  });
}

/** Convenience helper used after upload completion — invalidates the
 * cache so the next read hits the server. Components can also import
 * ``indexedFilesKey`` directly and call ``invalidateQueries`` themselves. */
export function useInvalidateIndexedFiles() {
  const queryClient = useQueryClient();
  return () => queryClient.invalidateQueries({ queryKey: indexedFilesKey });
}
