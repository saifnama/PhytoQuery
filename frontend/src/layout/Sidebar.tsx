import React, { useState, useRef } from 'react';
import { Plus, SidebarSimple, FileText } from '@phosphor-icons/react';
import { ragApi } from '../lib/api';

interface SidebarProps {
  expanded: boolean;
  onCollapse: () => void;
  onResultReceived?: (result: unknown) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ expanded, onCollapse }) => {
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  return (
    <aside
      className={`bg-white sidebar-transition flex-shrink-0 flex flex-col border-r border-slate-200 ${
        expanded ? 'w-80' : 'w-0'
      }`}
    >
      {/* Sidebar Header */}
      <div className="h-14 flex items-center justify-between px-5 border-b border-slate-100">
        <div /> {/* Spacer */}
        <button
          onClick={onCollapse}
          className="text-slate-400 hover:text-slate-600 transition p-1.5 focus:outline-none"
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
                className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-[#f0f4f9] hover:bg-[#e1e9f1] text-slate-600 rounded-full transition-all group border border-transparent hover:border-blue-100 disabled:opacity-50"
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
              <div className="w-12 h-12 bg-slate-50 rounded-xl flex items-center justify-center mx-auto mb-4 border border-slate-100/50">
                <FileText size={24} className="text-slate-300" />
              </div>
              <p className="text-[13px] text-slate-600 font-medium mb-1">
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
