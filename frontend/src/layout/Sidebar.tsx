import React, { useRef, useCallback } from 'react';
import { Plus, SidebarSimple, FileText } from '@phosphor-icons/react';
import { ragApi } from '../lib/api';
import { useUploadStore } from '../stores/uploadStore';

interface SidebarProps {
  expanded: boolean;
  onCollapse: () => void;
  onResultReceived?: (result: unknown) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ expanded, onCollapse }) => {
  // Shared upload state — RagPage and Sidebar both read/write this
  // store so an upload triggered from one surface immediately
  // reflects in the other (see frontend/src/stores/uploadStore.ts).
  const uploadStatus = useUploadStore((s) => s.status);
  const setUploadStatus = useUploadStore((s) => s.setStatus);
  const isUploading = useUploadStore((s) => s.isUploading);
  const setIsUploading = useUploadStore((s) => s.setIsUploading);
  const setCurrentJobId = useUploadStore((s) => s.setCurrentJobId);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Trigger app-level polling by writing the job id into the store —
  // the App-mounted ``UploadStatusListener`` watches and handles
  // status text + completion side-effects centrally. Replaces the
  // previous local ``startUploadPolling`` + setInterval helper.
  const beginPollingJob = useCallback(
    (jobId: string) => {
      setIsUploading(true);
      setUploadStatus('Processing upload…');
      setCurrentJobId(jobId);
    },
    [setIsUploading, setUploadStatus, setCurrentJobId],
  );

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadStatus('');

    // Slice into 20-PDF batches when many files are selected so we
    // stay under reverse-proxy upload caps and don't hold a multi-GB
    // payload in browser memory.
    const fileArr = Array.from(files);
    const CHUNK_THRESHOLD = 20;

    try {
      if (fileArr.length > CHUNK_THRESHOLD) {
        // ``uploadFilesChunked`` returns the last batch's UploadResponse;
        // no need for a ``let`` + callback-capture pattern (TS flow
        // analysis can't see writes inside callbacks, which previously
        // narrowed ``lastResult`` to ``never`` after the await and
        // broke ``tsc -b``).
        const lastResult = await ragApi.uploadFilesChunked(
          fileArr,
          'docling',
          CHUNK_THRESHOLD,
          (idx, total, batchResult) => {
            setUploadStatus(
              `Queued batch ${idx + 1} of ${total} (${batchResult.files.length} file${
                batchResult.files.length > 1 ? 's' : ''
              })…`,
            );
          },
        );
        if (lastResult.status === 'processing' && lastResult.job_id) {
          beginPollingJob(lastResult.job_id);
        } else {
          setUploadStatus(
            `Indexed ${lastResult.files.length} file${lastResult.files.length > 1 ? 's' : ''}.`,
          );
          setIsUploading(false);
        }
      } else {
        const result = await ragApi.uploadFiles(fileArr);
        if (result.status === 'processing' && result.job_id) {
          beginPollingJob(result.job_id);
        } else {
          setUploadStatus(`Indexed ${result.files.length} file${result.files.length > 1 ? 's' : ''}.`);
          setIsUploading(false);
        }
      }
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadStatus('Upload failed. Please try again.');
      setIsUploading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <aside
      className={`bg-background sidebar-transition flex-shrink-0 flex flex-col border-r border-border ${
        expanded ? 'w-80' : 'w-0'
      }`}
    >
      {/* Sidebar Header */}
      <div className="h-14 flex items-center justify-between px-5 border-b border-surface-c">
        <div /> {/* Spacer */}
        <button
          onClick={onCollapse}
          className="text-on-surface-muted hover:text-on-surface-variant transition p-1.5 focus:outline-none"
          aria-label="Collapse sidebar"
        >
          <SidebarSimple size={22} weight="bold" />
        </button>
      </div>

      {/* Sidebar Content */}
      {expanded && (
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
          <div className="p-4 space-y-6">
            {/* Add Sources Button */}
            <div className="relative">
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                multiple
                accept=".pdf"
                className="hidden"
              />
              <button
                onClick={handleUploadClick}
                disabled={isUploading}
                className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-[#f0f4f9] hover:bg-[#e1e9f1] text-on-surface-variant rounded-full transition-all group border border-transparent hover:border-primary/20 disabled:opacity-50"
              >
                <Plus size={16} />
                <span className="text-sm font-medium">
                  {isUploading ? 'Uploading...' : 'Add sources'}
                </span>
              </button>
            </div>

            {/* Upload Status */}
            {uploadStatus && (
              <div
                className={`px-2 text-[10px] font-medium ${
                  uploadStatus.includes('failed')
                    ? 'text-red-400'
                    : 'text-green-400'
                }`}
              >
                {uploadStatus}
              </div>
            )}

            {/* Empty State Illustrations */}
            <div className="pt-20 text-center px-6">
              <div className="w-12 h-12 bg-surface-c/50 rounded-xl flex items-center justify-center mx-auto mb-4 border border-surface-c/50">
                <FileText size={24} className="text-on-surface-muted/30" />
              </div>
              <p className="text-[13px] text-on-surface-variant font-medium mb-1">
                Saved items will appear here
              </p>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
};

export default Sidebar;
