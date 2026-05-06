import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
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
  type RagSource,
} from './assistant/runtime';

// Source alias kept for callsite stability — same shape as the
// assistant-ui runtime's RagSource (source/section/parser_type/score/chunk_text).
type Source = RagSource;

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

// Chat history is now persisted by the assistant-ui runtime under
// `pq_chat_history` (see ./assistant/runtime.ts). The keys below are
// only for state outside the chat thread itself.
const SESSION_KEYS = {
  FILES: 'pq_chat_files',
  PARSER: 'pq_chat_parser',
  SIDEBAR: 'pq_chat_sidebar',
};

const RagPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const locationState = location.state as RagLocationState | null;
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
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
  const [previewSource, setPreviewSource] = useState<Source | null>(null);
  const [activePdfFile, setActivePdfFile] = useState<UploadedFile | null>(null);
  const [activePdfUrl, setActivePdfUrl] = useState<string | null>(null);
  const importedPaperRef = useRef<string | null>(null);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Load indexed files from backend and merge with persisted selection
  // status. The merge is *defensive*: if the backend reports an empty
  // list while we have files locally (e.g., a brief eventual-consistency
  // window right after upload, or a stale 404 on the existence probe),
  // we preserve the local list rather than wiping it. The "Reset all"
  // flow already clears local state explicitly, so this can't hide a
  // real reset.
  const loadIndexedFiles = useCallback(async () => {
    try {
      const files = await ragApi.listFiles();

      setUploadedFiles((prev) => {
        if (files.length === 0 && prev.length > 0) {
          // Backend returned no files but we have locals — keep locals.
          return prev;
        }

        const selectionMap = new Map(prev.map((f) => [f.name, f.selected]));
        const summaryMap = new Map(prev.map((f) => [f.name, f.summary]));

        return files.map((f) => ({
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
    } catch {
      // Silently ignore
    }
  }, []);

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

  const startUploadPolling = useCallback((jobId: string, parserType: 'pymupdf' | 'docling') => {
    // Clear any existing poll
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
    }
    setIsUploading(true);
    setUploadStatus(`Processing upload...`);

    const poll = async () => {
      try {
        const status = await ragApi.getUploadStatus(jobId);
        if (status.status === 'completed') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          setIsUploading(false);
          setUploadStatus(
            `Indexed ${status.files.length} file${status.files.length > 1 ? 's' : ''} (${parserType === 'pymupdf' ? 'Fast' : 'Detailed'}).`
          );
          applyUploadResult(
            { files: status.files, summaries: status.summaries },
            parserType
          );
          await loadIndexedFiles();
        } else if (status.status === 'failed') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          setIsUploading(false);
          setUploadStatus(`Upload failed: ${status.error || 'Unknown error'}`);
        } else {
          // still processing
          setUploadStatus(`Processing ${status.files.join(', ')}...`);
        }
      } catch (e) {
        // Keep polling on transient errors
        console.error('Poll error:', e);
      }
    };

    // Poll immediately, then every 2 seconds
    poll();
    pollIntervalRef.current = setInterval(poll, 2000);
  }, [applyUploadResult, loadIndexedFiles]);

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

  useEffect(() => {
    loadIndexedFiles();
  }, [loadIndexedFiles]);

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
          startUploadPolling(result.job_id, parserType);
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
        navigate(location.pathname, { replace: true, state: null });
      }
    };

    importPaperPdf();
  }, [applyUploadResult, loadIndexedFiles, location.pathname, locationState, navigate, parserType]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadStatus('');

    try {
      const result = await ragApi.uploadFiles(Array.from(files), parserType);
      if (result.status === 'processing' && result.job_id) {
        startUploadPolling(result.job_id, parserType);
      } else {
        setUploadStatus(
          `Indexed ${result.files.length} file${result.files.length > 1 ? 's' : ''} (${
            parserType === 'pymupdf' ? 'Fast' : 'Detailed'
          }).`
        );
        applyUploadResult(result, parserType);
        // Refresh from backend to get accurate chunk counts
        await loadIndexedFiles();
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
        // Chat history lives in the runtime now — clear its session key
        // and the file list key so a refresh shows a clean slate.
        clearPersistedChatHistory();
        sessionStorage.removeItem(SESSION_KEYS.FILES);
        // Wipe in-runtime thread state too. The runtime exposes
        // .thread.cancelRun and reset via thread-list, but the simplest
        // reliable path is a hard reload after the storage purge.
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
          <Thread onSourceClick={setPreviewSource} />
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

      {/* ─── Source Preview Panel ─── */}
      {previewSource && (
        <div className="fixed inset-0 z-50 flex justify-end">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/20 backdrop-blur-sm"
            onClick={() => setPreviewSource(null)}
          />
          {/* Panel */}
          <div className="relative w-full max-w-lg bg-white shadow-2xl border-l border-slate-200 flex flex-col animate-slide-up">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
              <div className="flex items-center space-x-3">
                <Eye size={18} className="text-blue-500" />
                <h3 className="text-sm font-bold text-slate-800">Source Preview</h3>
              </div>
              <button
                onClick={() => setPreviewSource(null)}
                className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors"
              >
                <X size={18} />
              </button>
            </div>
            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {/* Metadata row */}
              <div className="flex flex-wrap items-center gap-2">
                <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                  previewSource.score >= 80 ? 'bg-green-100 text-green-700'
                  : previewSource.score >= 60 ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-red-100 text-red-700'
                }`}>
                  {previewSource.score}% Match
                </span>
                <span className="text-xs text-slate-400 font-medium">
                  {previewSource.parser_type === 'pymupdf' ? 'Fast' : 'Detailed'} Parser
                </span>
              </div>
              {/* Source file */}
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">Source</p>
                <p className="text-sm text-slate-800 font-medium">{previewSource.source}</p>
                {previewSource.section && (
                  <p className="text-xs text-slate-500 mt-0.5">{previewSource.section}</p>
                )}
              </div>
              {/* Chunk text */}
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Retrieved Passage</p>
                <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                  <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                    {previewSource.chunk_text}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RagPage;
