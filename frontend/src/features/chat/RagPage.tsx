import React, { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { Chats, Paperclip, PaperPlaneTilt, Plus, FileText } from '@phosphor-icons/react';
import { ragApi } from '../../lib/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: Record<string, unknown>[];
}

const RagPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadStatus('');

    try {
      const result = await ragApi.uploadFiles(Array.from(files));
      setUploadStatus(`Indexed ${result.files.length} files.`);
    } catch (error) {
      console.error('Upload failed:', error);
      setUploadStatus('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: query.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuery('');
    setIsLoading(true);
    scrollToBottom();

    try {
      const result = await ragApi.query(query.trim());
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: result.answer,
        sources: result.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('RAG query failed:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
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

  return (
    <div className="h-full flex">
      {/* Sidebar - Upload PDF */}
      <aside className="w-64 border-r border-slate-100 bg-white flex flex-col flex-shrink-0">
        <div className="p-4 border-b border-slate-100">
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Upload Papers</h3>
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
            <span>{isUploading ? 'Uploading...' : 'Add PDF'}</span>
          </button>
          {uploadStatus && (
            <p className={`mt-2 text-xs ${uploadStatus.includes('failed') ? 'text-red-500' : 'text-green-600'}`}>
              {uploadStatus}
            </p>
          )}
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <div className="text-center py-12">
            <FileText size={32} className="text-slate-200 mx-auto mb-3" />
            <p className="text-xs text-slate-400">No papers uploaded yet</p>
          </div>
        </div>
      </aside>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col relative">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-8 pb-32 space-y-6">
          {messages.length === 0 ? (
            <div className="text-center max-w-xl mx-auto mt-20">
              <div className="w-16 h-16 bg-blue-50 text-blue-300 rounded-2xl flex items-center justify-center mx-auto mb-6">
                <Chats size={32} weight="thin" />
              </div>
              <h2 className="text-xl font-bold text-slate-900 mb-2">RAG Assistant</h2>
              <p className="text-sm text-slate-500">Upload research papers and ask questions about them</p>
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
                        ? 'bg-blue-600 text-white'
                        : 'bg-white border border-slate-100 text-slate-900'
                    }`}
                  >
                    <div className={`text-sm prose prose-sm max-w-none ${message.type === 'user' ? 'prose-invert' : ''}`}>
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-slate-100">
                        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Sources</p>
                        <div className="space-y-1">
                          {message.sources.map((source, idx) => (
                            <p key={idx} className="text-xs text-slate-500">
                              {String(source.source)}{source.section ? ` - ${source.section}` : ''}
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
                      <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
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
            <div className="bg-white border border-slate-200 rounded-2xl p-3 shadow-lg">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your uploaded papers..."
                className="w-full bg-transparent border-none focus:ring-0 text-sm text-slate-900 placeholder:text-slate-400 resize-none py-2 max-h-32"
                rows={1}
              />
              <div className="flex items-center justify-between pt-2 border-t border-slate-100">
                <button type="button" className="text-slate-400 hover:text-blue-600 transition-colors">
                  <Paperclip size={16} />
                </button>
                <button
                  type="submit"
                  disabled={!query.trim() || isLoading}
                  className="bg-blue-600 hover:bg-blue-700 text-white p-2 rounded-xl transition-all disabled:opacity-50"
                >
                  <PaperPlaneTilt size={16} weight="bold" />
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default RagPage;
