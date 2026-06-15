/**
 * TanStack Query hook that polls a single upload job until terminal.
 *
 * Replaces the per-component ``setInterval`` + ``pollIntervalRef``
 * pattern that lived in ``RagPage.tsx`` and ``Sidebar.tsx``. Both
 * components ran independent polls — they could fire simultaneously
 * for the same job id, didn't share state on completion, and had
 * manual cleanup that was easy to leak on hot-reload. With the
 * single shared QueryClient we get free deduplication: any number of
 * components observing the same ``['upload-status', jobId]`` key
 * receive the same data from a single polling source.
 *
 * Polling stops automatically when the job reaches ``completed`` or
 * ``failed`` — ``refetchInterval`` returns ``false`` to halt further
 * fetches. The query then sits in cache; observers that mount later
 * see the terminal status without re-fetching.
 */
import { useQuery } from '@tanstack/react-query';
import { ragApi } from '../lib/api';
import type { UploadJobStatus } from '../types';

const POLL_INTERVAL_MS = 2000;

export function useUploadJobStatus(jobId: string | null) {
  return useQuery<UploadJobStatus>({
    queryKey: ['upload-status', jobId],
    queryFn: () => ragApi.getUploadStatus(jobId as string),
    enabled: !!jobId,
    // Poll while processing; stop the moment status is terminal.
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'completed' || status === 'failed') return false;
      return POLL_INTERVAL_MS;
    },
    // Don't re-fetch when the tab regains focus — completion side-
    // effects already fired; a duplicate fetch just adds noise.
    refetchOnWindowFocus: false,
    // Treat polled data as always-fresh for the purposes of cache
    // dedup; the refetchInterval already handles "is this stale".
    staleTime: POLL_INTERVAL_MS,
  });
}
