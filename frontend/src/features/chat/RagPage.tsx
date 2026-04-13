import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Chats,
  Plus,
  ArrowUp,
  FileText,
  FilePdf,
  FileDoc,
  File as FileIcon,
  StackSimple,
  Check,
  Trash,
  DownloadSimple,
  Warning,
  SidebarSimple,
} from '@phosphor-icons/react';
import { ragApi } from '../../lib/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: Record<string, unknown>[];
}

function formatTimestamp(): string {
  return new Date().toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

interface UploadedFile {
  name: string;
  fileType: string;
  chunkCount: number;
  selected: boolean;
  parserType: 'pymupdf' | 'docling';
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

const SESSION_KEYS = {
  MESSAGES: 'pq_chat_messages',
  FILES: 'pq_chat_files',
  PARSER: 'pq_chat_parser',
  SIDEBAR: 'pq_chat_sidebar',
};

const RagPage: React.FC = () => {
  // Initialize state from sessionStorage
  const [messages, setMessages] = useState<Message[]>(() => {
    const saved = sessionStorage.getItem(SESSION_KEYS.MESSAGES);
    return saved ? JSON.parse(saved) : [];
  });
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [parserType, setParserType] = useState<'pymupdf' | 'docling'>(() => {
    const saved = sessionStorage.getItem(SESSION_KEYS.PARSER);
    return (saved as any) || 'pymupdf';
  });
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>(() => {
    const saved = sessionStorage.getItem(SESSION_KEYS.FILES);
    return saved ? JSON.parse(saved) : [];
  });
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    const saved = sessionStorage.getItem(SESSION_KEYS.SIDEBAR);
    return saved === 'true';
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-expand textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 192)}px`; // Max 192px (approx 8 lines)
    }
  }, [query]);

  // Load indexed files from backend and merge with persisted selection status
  const loadIndexedFiles = useCallback(async () => {
    try {
      const files = await ragApi.listFiles();

      setUploadedFiles((prev) => {
        // Create a map of existing selection statuses
        const selectionMap = new Map(prev.map((f) => [f.name, f.selected]));

        return files.map((f) => ({
          name: f.name,
          fileType: f.file_type,
          chunkCount: f.chunk_count,
          // If we had a selection status for this file before, preserve it; otherwise default to true
          selected: selectionMap.has(f.name) ? !!selectionMap.get(f.name) : true,
          parserType: f.parser_type as 'pymupdf' | 'docling',
        }));
      });
    } catch {
      // Silently ignore
    }
  }, []);

  // Persist state changes to sessionStorage
  useEffect(() => {
    sessionStorage.setItem(SESSION_KEYS.MESSAGES, JSON.stringify(messages));
  }, [messages]);

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

  // Cleanup user data when user closes/refreshes the page
  useEffect(() => {
    const handleBeforeUnload = async () => {
      // Call cleanup endpoint when user closes the page
      try {
        await ragApi.cleanupUserData();
      } catch (e) {
        // Ignore errors during cleanup - don't block page unload
        console.log('User data cleanup on close');
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    
    // Also handle pagehide for mobile browsers
    window.addEventListener('pagehide', handleBeforeUnload);

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('pagehide', handleBeforeUnload);
    };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadStatus('');

    try {
      const result = await ragApi.uploadFiles(Array.from(files), parserType);
      setUploadStatus(
        `Indexed ${result.files.length} file${result.files.length > 1 ? 's' : ''} (${
          parserType === 'pymupdf' ? 'Fast' : 'Detailed'
        }).`
      );

      // Add newly uploaded files to the sidebar (avoid duplicates)
      setUploadedFiles((prev) => {
        const existingNames = new Set(prev.map((f) => f.name));
        const newFiles: UploadedFile[] = result.files
          .filter((name) => !existingNames.has(name))
          .map((name) => {
            const ext = name.lastIndexOf('.') !== -1 ? name.slice(name.lastIndexOf('.')) : '.pdf';
            return { name, fileType: ext, chunkCount: 0, selected: true, parserType };
          });
        return [...prev, ...newFiles];
      });

      // Refresh from backend to get accurate chunk counts
      await loadIndexedFiles();
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadStatus('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
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

  const handleExportMarkdown = () => {
    let mdContent = `# Source Documents\n`;
    const selected = uploadedFiles.filter((f) => f.selected).map((f) => f.name);
    mdContent += selected.length > 0 ? selected.map(s => `- ${s}`).join('\n') : "None";
    mdContent += `\n\n---\n\n`;

    messages.forEach((msg) => {
      if (msg.type === 'user') {
        mdContent += `### You\n${msg.content}\n\n`;
      } else {
        mdContent += `### PhytoQuery Assistant\n${msg.content}\n\n`;
        if (msg.sources && msg.sources.length > 0) {
          mdContent += `> **Sources:**\n`;
          msg.sources.forEach((src: any) => {
            let srcText = `> - ${src.source}`;
            if (src.section) srcText += ` — ${src.section}`;
            if (src.parser_type) {
              srcText += ` (${src.parser_type === 'pymupdf' ? 'Fast' : 'Detailed'})`;
            }
            mdContent += `${srcText}\n`;
          });
          mdContent += `\n`;
        }
      }
    });

    const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `PhytoQuery_Chat_${new Date().toISOString().slice(0, 10)}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleResetAll = async () => {
    const confirmMessage =
      "Are you sure you want to permanently delete ALL chat history and source files? This cannot be undone.";
    if (window.confirm(confirmMessage)) {
      try {
        await ragApi.resetChat();
        setMessages([]);
        setUploadedFiles([]);
        sessionStorage.removeItem(SESSION_KEYS.MESSAGES);
        sessionStorage.removeItem(SESSION_KEYS.FILES);
      } catch (error) {
        console.error('Reset failed:', error);
        alert('Failed to reset chat. Please try again.');
      }
    }
  };

  // --- Chat logic ---
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    // Require at least one file selected
    const selectedFiles = uploadedFiles.filter((f) => f.selected).map((f) => f.name);
    if (uploadedFiles.length > 0 && selectedFiles.length === 0) {
      const errorMsg: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: 'Please select at least one source document before asking a question.',
        timestamp: formatTimestamp(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query.trim(),
      timestamp: formatTimestamp(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);
    scrollToBottom();

    try {
      // Pass selected files for filtered retrieval (or undefined for global search)
      const filterFiles = selectedFiles.length > 0 ? selectedFiles : undefined;
      const result = await ragApi.query(query.trim(), filterFiles);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: result.answer,
        timestamp: formatTimestamp(),
        sources: result.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('RAG query failed:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: formatTimestamp(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      scrollToBottom();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
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
                onClick={() => toggleFile(file.name)}
                className={`p-1.5 rounded-lg transition-colors ${
                  file.selected
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
                        onClick={() => toggleFile(file.name)}
                        className="w-full flex items-center space-x-3 px-5 py-2.5 hover:bg-slate-50 transition-colors group cursor-pointer"
                      >
                        <FileTypeIcon ext={file.fileType} size={20} />
                        <div className="flex-1 min-w-0 text-left">
                          <p className="text-sm text-slate-700 truncate leading-tight">
                            {displayName(file.name)}
                          </p>
                        </div>
                        <button
                          onClick={(e) => handleDeleteFile(file.name, e)}
                          className="text-slate-300 hover:text-red-500 transition-all p-1 rounded-md hover:bg-red-50 flex-shrink-0"
                          title={`Remove ${file.name}`}
                        >
                          <Trash size={14} />
                        </button>
                        <CustomCheckbox
                          checked={file.selected}
                          onChange={() => toggleFile(file.name)}
                        />
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

      {/* ─── Chat Area ─── */}
      <div className="flex-1 flex flex-col relative">
        {/* Chat header with export */}
        {messages.length > 0 && (
          <div className="flex items-center justify-end px-6 py-2 border-b border-slate-100 bg-white/80 backdrop-blur-sm">
            <button
              onClick={handleExportMarkdown}
              className="flex items-center space-x-1.5 text-xs font-medium text-slate-500 hover:text-slate-800 transition-colors px-3 py-1.5 rounded-lg hover:bg-slate-100 border border-transparent hover:border-slate-200"
            >
              <DownloadSimple size={14} />
              <span>Export Chat (.md)</span>
            </button>
          </div>
        )}
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-8 pb-32 space-y-6">
          {messages.length === 0 ? (
            <div className="text-center max-w-xl mx-auto mt-20">
              <div className="w-16 h-16 bg-blue-50 text-blue-300 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Chats size={32} weight="thin" />
              </div>
              <h2 className="text-xl font-bold text-slate-900 mb-2">RAG Assistant</h2>
              <p className="text-sm text-slate-500">
                Upload research papers and ask questions about them
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-5 py-3 ${
                      message.type === 'user'
                        ? 'text-slate-900'
                        : 'bg-white border border-slate-100 text-slate-900'
                    }`}
                    style={message.type === 'user' ? { backgroundColor: '#ffecf6' } : undefined}
                  >
                    <div
                      className={`text-sm prose prose-sm max-w-none`}
                    >
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-slate-100">
                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                          Sources
                        </p>
                        <div className="space-y-1">
                          {message.sources.map((source: any, idx) => (
                            <p key={idx} className="text-xs text-slate-500">
                              {String(source.source)}
                              {source.section ? ` — ${source.section}` : ''}
                              {source.parser_type && (
                                <span className="ml-1 text-[10px] text-slate-400 font-medium italic">
                                  ({source.parser_type === 'pymupdf' ? 'Fast' : 'Detailed'})
                                </span>
                              )}
                            </p>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white border border-slate-100 rounded-2xl px-5 py-3">
                    <div className="flex items-center space-x-1.5">
                      <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" />
                      <div
                        className="w-2 h-2 bg-slate-300 rounded-full animate-bounce"
                        style={{ animationDelay: '0.1s' }}
                      />
                      <div
                        className="w-2 h-2 bg-slate-300 rounded-full animate-bounce"
                        style={{ animationDelay: '0.2s' }}
                      />
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-gray-50 via-gray-50/90 to-transparent pointer-events-none">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto pointer-events-auto">
            <div className="bg-white border border-slate-200/80 rounded-[28px] p-1.5 pr-2.5 shadow-2xl shadow-slate-200/50 flex items-end">
              <textarea
                ref={textareaRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything..."
                className="flex-1 bg-transparent border-none focus:ring-0 text-base text-slate-800 placeholder:text-slate-400 resize-none py-3 px-4 min-h-[52px]"
                rows={1}
              />
              <div className="pb-1.5">
                <button
                  type="submit"
                  disabled={!query.trim() || isLoading}
                  className="w-10 h-10 rounded-full flex items-center justify-center transition-all shadow-md disabled:opacity-40 disabled:shadow-none hover:shadow-lg active:scale-95 group"
                  style={{ backgroundColor: '#ff6dba' }}
                >
                  <ArrowUp 
                    size={20} 
                    weight="bold" 
                    className="text-white group-hover:translate-y-[-1px] transition-transform" 
                  />
                </button>
              </div>
            </div>
            <p className="text-[10px] text-center text-slate-400 mt-3 tracking-wide">
              PhytoQuery can make mistakes. Verify important information.
            </p>
          </form>
        </div>
      </div>
    </div>
  );
};

export default RagPage;
