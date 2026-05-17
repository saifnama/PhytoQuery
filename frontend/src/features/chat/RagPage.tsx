import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useRouterState } from '@tanstack/react-router';
import { AssistantRuntimeProvider } from '@assistant-ui/react';
import {
  Plus,
  FileText,
  FilePdf,
  FileDoc,
  File as FileIcon,
  StackSimple,
  Check,
  Trash,
  Warning,
  SidebarSimple,
  Eye,
  X,
  ArrowsOutSimple,
} from '@phosphor-icons/react';
import { buildChatFileContentUrl, paperApi, ragApi } from '../../lib/api';
import { Thread } from './assistant/Thread';
import {
  clearPersistedChatHistory,
  usePhytoQueryRuntime,
  type Citation,
  type RagSource,
} from './assistant/runtime';
import { MarkdownPreviewPanel } from './MarkdownPreviewPanel';
import { useUploadStore } from '../../stores/uploadStore';
import { useIndexedFiles } from '../../hooks/useIndexedFiles';

interface UploadedFile {
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

interface RagLocationState {
  importPaperPdf?: {
    identifier: string;
    title?: string;
  };
}

/** Return the correct Phosphor icon for a given file extension. */
function FileTypeIcon({ ext, size = 20 }: { ext: string; size?: number }) {
  const className = 'text-blue-500 flex-shrink-0';
  switch (ext) {
    case '.pdf':
      return <FilePdf size={size} weight="duotone" className={className} />;
    case '.doc':
    case '.docx':
      return <FileDoc size={size} weight="duotone" className={className} />;
    case '.txt':
    case '.md':
      return <FileText size={size} weight="duotone" className={className} />;
    default:
      return <FileIcon size={size} weight="duotone" className={className} />;
  }
}

/** Custom styled checkbox matching the reference design. */
function CustomCheckbox({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onChange}
      className={`
        w-5 h-5 rounded flex-shrink-0 flex items-center justify-center
        transition-all duration-150 border
        ${
          checked
            ? 'bg-slate-600 border-slate-600 text-white'
            : 'bg-white border-slate-300 hover:border-slate-500'
        }
      `}
    >
      {checked && <Check size={14} weight="bold" />}
    </button>
  );
}

function SimplePdfViewer({
  pdfUrl,
}: {
  pdfUrl: string;
}) {
  return (
    <iframe
      key={pdfUrl}
      src={pdfUrl}
      title="PDF"
      className="h-full w-full rounded-xl border-0"
    />
  );
}

// Chat history is persisted by the assistant-ui runtime under
// `pq_chat_history` (see ./assistant/runtime.ts). The keys below are
// only for state outside the chat thread itself.
const SESSION_KEYS = {
  FILES: 'pq_chat_files',
  PARSER: 'pq_chat_parser',
  SIDEBAR: 'pq_chat_sidebar',
};

const RagPage: React.FC = () => {
  const navigate = useNavigate();
  const locationState = useRouterState({
    select: (s) => s.location.state as unknown as RagLocationState | null,
  });
  // Upload status + isUploading live in the shared Zustand store so
  // RagPage and the layout Sidebar always agree on whether an upload
  // is in flight (see frontend/src/stores/uploadStore.ts). The store
  // selectors use individual getters so re-renders only fire when
  // the slice the component reads actually changes.
  const uploadStatus = useUploadStore((s) => s.status);
  const setUploadStatus = useUploadStore((s) => s.setStatus);
  const isUploading = useUploadStore((s) => s.isUploading);
  const setIsUploading = useUploadStore((s) => s.setIsUploading);
  const [parserType, setParserType] = useState<'pymupdf' | 'docling'>(() => {
    const saved = sessionStorage.getItem(SESSION_KEYS.PARSER);
    return (saved as 'pymupdf' | 'docling' | null) || 'pymupdf';
  });
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>(() => {
    const saved = sessionStorage.getItem(SESSION_KEYS.FILES);
    return saved ? JSON.parse(saved) : [];
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    const saved = sessionStorage.getItem(SESSION_KEYS.SIDEBAR);
    return saved === 'true';
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  // ``triggerKey`` is bumped on every citation click so the markdown
  // preview can restart its flash animation even when the user clicks
  // the same [N] twice in a row (otherwise React would reconcile the
  // existing element and the CSS animation wouldn't replay).
  const [activeCitation, setActiveCitation] = useState<{
    source: RagSource;
    citation?: Citation;
    triggerKey: number;
  } | null>(null);
  const [activePdfFile, setActivePdfFile] = useState<UploadedFile | null>(null);
  const [activePdfUrl, setActivePdfUrl] = useState<string | null>(null);
  const importedPaperRef = useRef<string | null>(null);

  // Build the assistant-ui runtime. The selected-file getter is read on
  // every send so the user's checkbox state always reflects in the
  // outgoing /api/chat/query/json request.
  const uploadedFilesRef = useRef(uploadedFiles);
  uploadedFilesRef.current = uploadedFiles;
  const getSelectedFiles = useCallback(
    () =>
      uploadedFilesRef.current
        .filter((f) => f.selected)
        .map((f) => f.name),
    [],
  );
  const runtime = usePhytoQueryRuntime(
    useMemo(
      () => ({ getSelectedFiles, enableSessionPersistence: true }),
      [getSelectedFiles],
    ),
  );

  // (No manual interval cleanup needed — TanStack Query stops the
  // upload-status poll automatically when its query goes inactive,
  // and the App-level UploadStatusListener clears ``currentJobId``
  // on terminal status. The previous ``pollIntervalRef`` cleanup is
  // gone with the manual setInterval that needed it.)

  // Server state for the indexed-files list lives in TanStack Query
  // (see frontend/src/hooks/useIndexedFiles.ts). The query auto-fetches
  // on mount, dedupes across tabs/components, and is invalidated after
  // every upload completion. The selection-status merge below is the
  // same defensive logic we used before: if the backend reports an
  // empty list while we have files locally (e.g., during the eventual-
  // consistency window right after upload), we preserve the locals so
  // the UI doesn't flash empty.
  const { data: indexedFilesData, refetch: refetchIndexedFiles } = useIndexedFiles();

  // Merge server payload into the local ``uploadedFiles`` whenever the
  // query data changes. ``setUploadedFiles`` is a stable React setter
  // so this effect only refires when the backend payload changes.
  useEffect(() => {
    if (!indexedFilesData) return;
    setUploadedFiles((prev) => {
      if (indexedFilesData.length === 0 && prev.length > 0) {
        // Backend returned no files but we have locals — keep locals.
        return prev;
      }

      const selectionMap = new Map(prev.map((f) => [f.name, f.selected]));
      const summaryMap = new Map(prev.map((f) => [f.name, f.summary]));

      return indexedFilesData.map((f) => ({
        name: f.name,
        fileType: f.file_type,
        chunkCount: f.chunk_count,
        // If we had a selection status for this file before, preserve it; otherwise default to true
        selected: selectionMap.has(f.name) ? !!selectionMap.get(f.name) : true,
        parserType: f.parser_type as 'pymupdf' | 'docling',
        authors: f.authors || '',
        doi: f.doi || '',
        journal: f.journal || '',
        summary: f.summary || summaryMap.get(f.name) || '',
      }));
    });
  }, [indexedFilesData]);

  // Compatibility shim — call sites still use ``loadIndexedFiles()``;
  // forward to the query's refetch so behavior is unchanged. Awaiting
  // the refetch resolves once data has been refreshed.
  const loadIndexedFiles = useCallback(async () => {
    await refetchIndexedFiles();
  }, [refetchIndexedFiles]);

  const closePdfViewer = useCallback(() => {
    setActivePdfFile(null);
    setActivePdfUrl(null);
  }, []);

  const applyUploadResult = useCallback((result: { files: string[]; summaries?: Record<string, string> }, activeParserType: 'pymupdf' | 'docling') => {
    setUploadedFiles((prev) => {
      const existingNames = new Set(prev.map((f) => f.name));
      const newFiles: UploadedFile[] = result.files
        .filter((name) => !existingNames.has(name))
        .map((name) => {
          const ext = name.lastIndexOf('.') !== -1 ? name.slice(name.lastIndexOf('.')) : '.pdf';
          return {
            name,
            fileType: ext,
            chunkCount: 0,
            selected: true,
            parserType: activeParserType,
            summary: result.summaries?.[name] || '',
          };
        });
      return [...prev, ...newFiles];
    });
  }, []);

  // The upload-status polling that used to live here as
  // ``startUploadPolling`` (with its own setInterval, pollIntervalRef,
  // and inline status handling) has moved to a single App-level
  // ``UploadStatusListener`` driven by TanStack Query (see
  // frontend/src/components/UploadStatusListener.tsx). Trigger a poll
  // by writing the new ``job_id`` into the upload store via
  // ``setCurrentJobId(jobId)`` after the multipart POST returns. The
  // listener handles status text, completion side-effects, and cache
  // invalidation centrally — no per-component refs to manage.
  const setCurrentJobId = useUploadStore((s) => s.setCurrentJobId);
  const beginPollingJob = useCallback(
    (jobId: string) => {
      setIsUploading(true);
      setUploadStatus('Processing upload…');
      setCurrentJobId(jobId);
    },
    [setIsUploading, setUploadStatus, setCurrentJobId],
  );

  const openPdfViewer = useCallback((file: UploadedFile) => {
    setActivePdfFile(file);
    setActivePdfUrl(buildChatFileContentUrl(file.name));
  }, []);

  // Persist state changes to sessionStorage. Chat messages are no
  // longer persisted here — the assistant-ui runtime owns that via its
  // ThreadHistoryAdapter, see ./assistant/runtime.ts.
  useEffect(() => {
    sessionStorage.setItem(SESSION_KEYS.FILES, JSON.stringify(uploadedFiles));
  }, [uploadedFiles]);

  useEffect(() => {
    sessionStorage.setItem(SESSION_KEYS.PARSER, parserType);
  }, [parserType]);

  useEffect(() => {
    sessionStorage.setItem(SESSION_KEYS.SIDEBAR, String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  // The on-mount fetch that used to live here is gone — ``useIndexedFiles``
  // auto-fetches on mount via TanStack Query, so a second refetch here
  // would just duplicate the request. Polling/refresh after upload is
  // handled by ``invalidateIndexedFiles()`` in the upload completion
  // path; manual refetches still go through ``loadIndexedFiles()``.

  useEffect(() => {
    if (!activePdfFile) return;
    const stillExists = uploadedFiles.some((file) => file.name === activePdfFile.name);
    if (!stillExists) {
      closePdfViewer();
    }
  }, [activePdfFile, closePdfViewer, uploadedFiles]);

  useEffect(() => {
    const pendingImport = locationState?.importPaperPdf;
    if (!pendingImport?.identifier) {
      return;
    }
    if (importedPaperRef.current === pendingImport.identifier) {
      return;
    }

    importedPaperRef.current = pendingImport.identifier;

    const importPaperPdf = async () => {
      setIsUploading(true);
      setUploadStatus(`Importing PDF into RAG (${parserType === 'pymupdf' ? 'Fast' : 'Detailed'})...`);

      try {
        const { blob, filename } = await paperApi.fetchPdf(pendingImport.identifier);
        const safeBase = (pendingImport.title || filename || 'paper')
          .replace(/<[^>]+>/g, '')
          .replace(/[^a-zA-Z0-9._-]+/g, '_')
          .replace(/^_+|_+$/g, '')
          .slice(0, 120) || 'paper';
        const finalName = filename.toLowerCase().endsWith('.pdf') ? filename : `${safeBase}.pdf`;
        const file = new File([blob], finalName, { type: 'application/pdf' });
        const result = await ragApi.uploadFiles([file], parserType);
        if (result.status === 'processing' && result.job_id) {
          beginPollingJob(result.job_id);
        } else {
          applyUploadResult(result, parserType);
          await loadIndexedFiles();
          setUploadStatus(
            `Indexed ${result.files.length} file${result.files.length > 1 ? 's' : ''} (${parserType === 'pymupdf' ? 'Fast' : 'Detailed'}).`
          );
        }
      } catch (error) {
        console.error('Paper PDF import failed:', error);
        setUploadStatus('Paper PDF import failed. Please try downloading it manually.');
        setIsUploading(false);
      } finally {
        navigate({ to: '/chat', replace: true, state: {} });
      }
    };

    importPaperPdf();
  }, [applyUploadResult, loadIndexedFiles, locationState, navigate, parserType]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadStatus('');

    // For large multi-file uploads, slice into batches of 20 PDFs
    // each so we stay well under reverse-proxy body-size limits
    // (nginx and Cloudflare commonly cap at 100 MB) and the browser
    // does not have to hold a 5+ GB multipart payload in memory.
    // Each batch becomes its own background job on the server; we
    // poll the LAST batch's job_id for the completion summary.
    const fileArr = Array.from(files);
    const CHUNK_THRESHOLD = 20;

    try {
      if (fileArr.length > CHUNK_THRESHOLD) {
        // ``uploadFilesChunked`` already returns the LAST batch's
        // UploadResponse, so we don't need a ``let`` + callback-capture
        // pattern. The previous shape (``let lastResult = null;`` mutated
        // inside ``onBatch``) tripped TypeScript's flow analysis — TS
        // doesn't track writes inside callbacks, so post-await the type
        // narrowed to ``never`` and broke ``tsc -b``. Using the awaited
        // return value sidesteps the issue entirely.
        const lastResult = await ragApi.uploadFilesChunked(
          fileArr,
          parserType,
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
          applyUploadResult(lastResult, parserType);
          await loadIndexedFiles();
          setUploadStatus(
            `Indexed ${lastResult.files.length} file${lastResult.files.length > 1 ? 's' : ''} (${parserType === 'pymupdf' ? 'Fast' : 'Detailed'}).`,
          );
        }
      } else {
        const result = await ragApi.uploadFiles(fileArr, parserType);
        if (result.status === 'processing' && result.job_id) {
          beginPollingJob(result.job_id);
        } else {
          setUploadStatus(
            `Indexed ${result.files.length} file${result.files.length > 1 ? 's' : ''} (${
              parserType === 'pymupdf' ? 'Fast' : 'Detailed'
            }).`,
          );
          applyUploadResult(result, parserType);
          // Refresh from backend to get accurate chunk counts
          await loadIndexedFiles();
        }
      }
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadStatus('Upload failed. Please try again.');
      setIsUploading(false);
    } finally {
      // Reset file input so the same file can be re-uploaded
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  // --- Selection logic ---
  const allSelected = uploadedFiles.length > 0 && uploadedFiles.every((f) => f.selected);

  const handleDeleteFile = async (filename: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Don't toggle checkbox
    try {
      await ragApi.deleteFile(filename);
      if (activePdfFile?.name === filename) {
        closePdfViewer();
      }
      setUploadedFiles((prev) => prev.filter((f) => f.name !== filename));
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const toggleSelectAll = () => {
    const newVal = !allSelected;
    setUploadedFiles((prev) => prev.map((f) => ({ ...f, selected: newVal })));
  };

  const toggleFile = (name: string) => {
    setUploadedFiles((prev) =>
      prev.map((f) => (f.name === name ? { ...f, selected: !f.selected } : f))
    );
  };

  const handleResetAll = async () => {
    const confirmMessage =
      "Are you sure you want to permanently delete ALL chat history and source files? This cannot be undone.";
    if (window.confirm(confirmMessage)) {
      try {
        await ragApi.resetChat();
        closePdfViewer();
        setUploadedFiles([]);
        // Clear chat history (runtime-owned) + file list. Hard reload
        // afterwards so the runtime reinitializes with a fresh empty
        // thread.
        clearPersistedChatHistory();
        sessionStorage.removeItem(SESSION_KEYS.FILES);
        window.location.reload();
      } catch (error) {
        console.error('Reset failed:', error);
        alert('Failed to reset chat. Please try again.');
      }
    }
  };

  /** Strip file extension for cleaner display. */
  const displayName = (name: string) => {
    const dotIdx = name.lastIndexOf('.');
    return dotIdx > 0 ? name.slice(0, dotIdx) : name;
  };

  return (
    <div className="h-full flex px-0">
      {/* ─── Sources Sidebar ─── */}
      <aside
        className={`border-r border-slate-100 bg-white flex flex-col flex-shrink-0 transition-all duration-200 ${
          sidebarCollapsed ? 'w-14' : 'w-72'
        }`}
      >
        {sidebarCollapsed ? (
          /* ── Collapsed: icon strip ── */
          <div className="flex flex-col items-center py-3 space-y-3 h-full">
            {/* Toggle */}
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="p-2 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-colors"
              title="Expand sources"
            >
              <SidebarSimple size={18} />
            </button>

            <div className="w-6 border-t border-slate-200" />

            {/* Add */}
            <button
              onClick={handleUploadClick}
              disabled={isUploading}
              className="p-2 rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-colors disabled:opacity-50"
              title="Add sources"
            >
              <Plus size={18} />
            </button>

            {/* File icons */}
            {uploadedFiles.map((file) => (
              <button
                key={file.name}
                onClick={() => openPdfViewer(file)}
                className={`p-1.5 rounded-lg transition-colors ${
                  activePdfFile?.name === file.name
                    ? 'bg-blue-50 text-blue-500 ring-1 ring-blue-200'
                    : file.selected
                      ? 'bg-slate-100 text-blue-500'
                      : 'text-slate-300 hover:text-slate-500'
                }`}
                title={file.name}
              >
                <FileTypeIcon ext={file.fileType} size={18} />
              </button>
            ))}
          </div>
        ) : (
          /* ── Expanded: full panel ── */
          <>
            {/* Header */}
            <div className="px-5 pt-5 pb-3 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <StackSimple size={18} weight="duotone" className="text-slate-400" />
                <h3 className="text-sm font-bold text-slate-800 tracking-tight">Sources</h3>
              </div>
              <div className="flex items-center space-x-1">
                <span className="text-xs text-slate-400 font-medium tabular-nums">
                  {uploadedFiles.length}
                </span>
                <button
                  onClick={() => setSidebarCollapsed(true)}
                  className="p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors"
                  title="Collapse sidebar"
                >
                  <SidebarSimple size={16} />
                </button>
              </div>
            </div>

            {/* Upload controls */}
            <div className="px-4 pb-3 space-y-2">
              {/* Parser type toggle */}
              <div className="flex items-center bg-slate-100 rounded-lg p-1">
                <button
                  type="button"
                  onClick={() => setParserType('pymupdf')}
                  className={`flex-1 py-1.5 px-3 text-xs font-medium rounded-md transition-all ${
                    parserType === 'pymupdf'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  Fast
                </button>
                <button
                  type="button"
                  onClick={() => setParserType('docling')}
                  className={`flex-1 py-1.5 px-3 text-xs font-medium rounded-md transition-all ${
                    parserType === 'docling'
                      ? 'bg-white text-blue-600 shadow-sm'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  Detailed
                </button>
              </div>

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
                className="w-full flex items-center justify-center space-x-2 py-2.5 px-4 bg-blue-50 hover:bg-blue-100 text-blue-600 rounded-lg transition-all font-medium text-sm disabled:opacity-50"
              >
                <Plus size={16} />
                <span>{isUploading ? 'Uploading...' : 'Add Sources'}</span>
              </button>
              {uploadStatus && (
                <p
                  className={`text-xs ${
                    uploadStatus.includes('failed') ? 'text-red-500' : 'text-green-600'
                  }`}
                >
                  {uploadStatus}
                </p>
              )}
            </div>

            {/* Select All / File List */}
            <div className="flex-1 overflow-y-auto border-t border-slate-100">
              {uploadedFiles.length > 0 ? (
                <>
                  {/* Select All */}
                  <button
                    onClick={toggleSelectAll}
                    className="w-full flex items-center justify-between px-5 py-3 hover:bg-slate-50 transition-colors border-b border-slate-50"
                  >
                    <span className="text-xs font-medium text-slate-500">Select all sources</span>
                    <CustomCheckbox checked={allSelected} onChange={toggleSelectAll} />
                  </button>

                  {/* File rows */}
                  <div className="py-1">
                    {uploadedFiles.map((file) => (
                      <div
                        key={file.name}
                        onClick={() => openPdfViewer(file)}
                        className={`w-full flex items-center space-x-3 px-5 py-2.5 transition-colors group cursor-pointer ${
                          activePdfFile?.name === file.name
                            ? 'bg-blue-50/80 ring-1 ring-inset ring-blue-100'
                            : 'hover:bg-slate-50'
                        }`}
                      >
                        <FileTypeIcon ext={file.fileType} size={20} />
                        <div className="flex-1 min-w-0 text-left">
                          <p className="text-sm text-slate-700 truncate leading-tight">
                            {displayName(file.name)}
                          </p>
                          {(file.authors || file.journal) && (
                            <p className="text-[10px] text-slate-400 truncate leading-tight mt-0.5">
                              {file.authors && <span>{file.authors}</span>}
                              {file.authors && file.journal && <span> · </span>}
                              {file.journal && <span className="italic">{file.journal}</span>}
                            </p>
                          )}
                          {file.summary && (
                            <p className="text-[10px] text-slate-400 line-clamp-2 leading-snug mt-0.5" title={file.summary}>
                              {file.summary}
                            </p>
                          )}
                        </div>
                        <Eye size={14} className={`flex-shrink-0 transition-colors ${activePdfFile?.name === file.name ? 'text-blue-500' : 'text-slate-300 group-hover:text-slate-500'}`} />
                        <button
                          onClick={(e) => handleDeleteFile(file.name, e)}
                          className="text-slate-300 hover:text-red-500 transition-all p-1 rounded-md hover:bg-red-50 flex-shrink-0"
                          title={`Remove ${file.name}`}
                        >
                          <Trash size={14} />
                        </button>
                        <div onClick={(e) => e.stopPropagation()}>
                          <CustomCheckbox
                            checked={file.selected}
                            onChange={() => toggleFile(file.name)}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="text-center py-16 px-4">
                  <FileText size={36} className="text-slate-200 mx-auto mb-3" />
                  <p className="text-xs text-slate-400 leading-relaxed">
                    No sources added yet.
                    <br />
                    Upload research papers to get started.
                  </p>
                </div>
              )}
            </div>
            
            {/* Sidebar Footer: Clear All */}
            <div className="p-4 border-t border-slate-100 bg-slate-50 mt-auto">
              <button
                onClick={handleResetAll}
                className="w-full flex items-center justify-center space-x-2 py-2 px-4 bg-white border border-red-200 hover:bg-red-50 text-red-600 rounded-lg transition-all font-medium text-xs shadow-sm hover:shadow"
              >
                <Warning size={14} weight="bold" />
                <span>Clear All Data</span>
              </button>
            </div>
          </>
        )}
      </aside>

      {/* ─── Chat Area (assistant-ui Thread) ─── */}
      <div className="flex-1 flex flex-col relative">
        <AssistantRuntimeProvider runtime={runtime}>
          <Thread
            onCitationClick={(payload) => {
              // Ignore clicks where the chunk_id no longer resolves
              // to a known source (rare; can happen if a stored
              // assistant message references a chunk we've since
              // wiped via "Reset all").
              if (!payload.source) return;
              setActiveCitation({
                source: payload.source,
                citation: payload.citation,
                // Date.now() is monotonic enough for animation
                // restart purposes; using a counter would also work
                // but adds a useRef for no extra value.
                triggerKey: Date.now(),
              });
            }}
          />
        </AssistantRuntimeProvider>
      </div>

      {activePdfFile && (
        <aside className="w-[min(32rem,42vw)] min-w-[22rem] border-l border-slate-200 bg-white flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
            <div className="min-w-0">
              <h3 className="mt-1 text-sm font-semibold text-slate-800 truncate" title={activePdfFile.name}>
                {displayName(activePdfFile.name)}
              </h3>
            </div>
            <div className="flex items-center gap-2">
              {activePdfUrl && (
                <a
                  href={activePdfUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors"
                  title="Open PDF in new tab"
                >
                  <ArrowsOutSimple size={18} />
                </a>
              )}
              <button
                type="button"
                onClick={closePdfViewer}
                className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors"
                title="Close PDF viewer"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-auto bg-slate-100 p-2">
            {activePdfUrl ? (
              <SimplePdfViewer pdfUrl={activePdfUrl} />
            ) : (
              <div className="h-full min-h-[16rem] flex items-center justify-center text-sm text-slate-500">
                Select a PDF to preview it here.
              </div>
            )}
          </div>
        </aside>
      )}

      {/* ─── Citation markdown preview panel ─── */}
      {activeCitation && (
        <MarkdownPreviewPanel
          source={activeCitation.source}
          citation={activeCitation.citation}
          triggerKey={activeCitation.triggerKey}
          onClose={() => setActiveCitation(null)}
        />
      )}
    </div>
  );
};

export default RagPage;
