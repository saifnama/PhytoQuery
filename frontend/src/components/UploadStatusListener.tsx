/**
 * Top-level upload-job watcher.
 *
 * This component renders nothing visible — it exists to centralize
 * the side-effects that used to be duplicated inside ``RagPage`` and
 * ``Sidebar``'s ``startUploadPolling`` helpers:
 *   - Read ``currentJobId`` from the shared upload store
 *   - Poll the backend for that job's status (via TanStack Query)
 *   - On terminal status, update the store's ``status`` text and
 *     ``isUploading`` flag, invalidate the indexed-files cache so
 *     RagPage's merge effect refreshes, and clear ``currentJobId``
 *     so polling stops
 *
 * Mounted once at the App layout level so it observes uploads
 * regardless of which route the user is on. Components only need
 * to ``setCurrentJobId(jobId)`` after a POST returns ``processing``;
 * the listener handles the rest.
 */
import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { useUploadStore } from '../stores/uploadStore';
import { useUploadJobStatus } from '../hooks/useUploadJobStatus';
import { indexedFilesKey } from '../hooks/useIndexedFiles';

import { useChatStore, type UploadedFile } from '../stores/chatStore';

export function UploadStatusListener() {
  const currentJobId = useUploadStore((s) => s.currentJobId);
  const setCurrentJobId = useUploadStore((s) => s.setCurrentJobId);
  const setStatus = useUploadStore((s) => s.setStatus);
  const setIsUploading = useUploadStore((s) => s.setIsUploading);
  const queryClient = useQueryClient();

  const { data: jobStatus } = useUploadJobStatus(currentJobId);

  useEffect(() => {
    if (!jobStatus || !currentJobId) return;

    if (jobStatus.status === 'completed') {
      const count = jobStatus.files.length;
      setStatus(`Indexed ${count} file${count === 1 ? '' : 's'}.`);
      setIsUploading(false);

      // Instantly add completed files into chatStore sidebar state
      const chatStore = useChatStore.getState();
      chatStore.setUploadedFiles((prev) => {
        const existing = new Set(prev.map((f) => f.name));
        const newEntries: UploadedFile[] = (jobStatus.files || [])
          .filter((name) => !existing.has(name))
          .map((name) => ({
            name,
            fileType: '.pdf',
            chunkCount: 0,
            selected: true,
            parserType: (jobStatus.parser_type as 'pymupdf' | 'docling') || 'pymupdf',
            summary: '',
          }));
        return [...prev, ...newEntries];
      });

      // Refetch indexed files immediately so metadata/chunk counts populate
      void queryClient.refetchQueries({ queryKey: indexedFilesKey });

      // Stop polling.
      setCurrentJobId(null);
    } else if (jobStatus.status === 'failed') {
      const reason = jobStatus.error || 'Unknown error';
      setStatus(`Upload failed: ${reason}`);
      setIsUploading(false);
      toast.error('Upload failed', { description: reason });
      setCurrentJobId(null);
    } else if (jobStatus.status === 'processing') {
      // Keep the user informed about which file the background job
      // is working on. ``files`` here lists the inputs queued for
      // this batch, not the running progress (we'd need a per-file
      // counter on the backend for that). Truncate the join so the
      // status pill stays readable.
      const list = jobStatus.files.slice(0, 3).join(', ');
      const more = jobStatus.files.length > 3 ? ` (+${jobStatus.files.length - 3} more)` : '';
      setStatus(`Processing ${list}${more}…`);
    }
  }, [
    jobStatus,
    currentJobId,
    queryClient,
    setCurrentJobId,
    setStatus,
    setIsUploading,
  ]);

  return null;
}
