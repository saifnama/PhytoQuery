import { useState, useMemo, useRef, useEffect } from 'react';
import { Plus, X, TrashSimple, FileArrowUp, ListNumbers, Graph, SidebarSimple, DotsThreeVertical, ChartBar, CaretDown, CaretUp, SpinnerGap } from '@phosphor-icons/react';
import { KnowledgeGraph, type KnowledgeGraphHandle } from '../reader/KnowledgeGraph';
import { downloadGraphHtml } from '../../utils/exportGraphHtml';
import { CompareMatrix } from './CompareMatrix';
import type { Entity } from '../../types';
import { useShallow } from 'zustand/react/shallow';
import { useAnalyseStore, type UploadedPaper } from '../../stores/analyseStore';

const PdfIcon = ({ size = 24, className = "" }: { size?: number, className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 160 160" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    <rect width="160" height="160" rx="28" fill="#E21101"/>
    <text
      x="50%"
      y="54%"
      textAnchor="middle"
      fill="#FFFFFF"
      fontFamily="Arial, Helvetica, sans-serif"
      fontSize="56"
      fontWeight="700"
      dominantBaseline="middle"
    >
      PDF
    </text>
  </svg>
);

const ENTITY_GROUP_ORDER = [
  'CHEMICAL',
  'SPECIES',
  'PLANT_PART',
  'DEVELOPMENT_STAGE',
  'EXTRACTION_METHOD',
  'ANALYTICAL_TECHNIQUE',
  'BIOACTIVITY',
  'DISEASE',
  'SEASON',
  'LOCATION',
] as const;

const LABEL_NAMES: Record<string, string> = {
  CHEMICAL: 'Chemical',
  SPECIES: 'Species',
  PLANT_PART: 'Plant Part',
  ANALYTICAL_TECHNIQUE: 'Analytical Technique',
  EXTRACTION_METHOD: 'Extraction Method',
  BIOACTIVITY: 'Bioactivity',
  DEVELOPMENT_STAGE: 'Development Stage',
  SEASON: 'Season',
  DISEASE: 'Disease',
  LOCATION: 'Location',
};

const getEntityAccentVar = (label: string) => `--entity-${label.toLowerCase().replace(/_/g, '-')}`;
const getEntityAccentColor = (label: string) => `var(${getEntityAccentVar(label)})`;

const AnalysePage = () => {
  // ── Persisted analyse state (sessionStorage-backed, per-tab) ────────────
  // useShallow for array/object slices avoids getSnapshot loop
  const papers = useAnalyseStore(useShallow((s) => s.papers));
  const selectedPaperId = useAnalyseStore((s) => s.selectedPaperId);
  const expandedGroups = useAnalyseStore(useShallow((s) => s.expandedGroups));
  const isCompareMode = useAnalyseStore((s) => s.isCompareMode);
  const compareSelection = useAnalyseStore(useShallow((s) => s.compareSelection));
  const addPapers = useAnalyseStore((s) => s.addPapers);
  const removePaper = useAnalyseStore((s) => s.removePaper);
  const setSelectedPaperId = useAnalyseStore((s) => s.setSelectedPaperId);
  const setExpandedGroups = useAnalyseStore((s) => s.setExpandedGroups);
  const toggleGroup = useAnalyseStore((s) => s.toggleGroup);
  const setIsCompareMode = useAnalyseStore((s) => s.setIsCompareMode);
  const setCompareSelection = useAnalyseStore((s) => s.setCompareSelection);
  const toggleCompareSelection = useAnalyseStore((s) => s.toggleCompareSelection);
  const clearCompareSelection = useAnalyseStore((s) => s.clearCompareSelection);

  // ── Derived: selectedPaper is looked up from id at render time ──────────
  const selectedPaper = useMemo(
    () => papers.find((p) => p.id === selectedPaperId) ?? null,
    [papers, selectedPaperId],
  );

  // ── Transient state (intentionally NOT persisted) ───────────────────────
  // viewerSrc is a blob: URL revoked on close → persisting would leave
  // a dangling reference next tab. Upload progress + click-outside menu
  // are pure UI affordances.
  const [viewerPaper, setViewerPaper] = useState<UploadedPaper | null>(null);
  const [viewerSrc, setViewerSrc] = useState<string | null>(null);
  const [viewerError, setViewerError] = useState<string | null>(null);
  const [isViewerLoading, setIsViewerLoading] = useState(false);
  const [viewerLoadingPaperId, setViewerLoadingPaperId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({ current: 0, total: 0 });
  
  // Dashboard layout states
  const [leftSidebarCollapsed, setLeftSidebarCollapsed] = useState(false);
  const [rightSidebarCollapsed, setRightSidebarCollapsed] = useState(false);
  const [rightSidebarMode, setRightSidebarMode] = useState<'entities' | 'graph'>('entities');
  const [hoverRightSidebarMode, setHoverRightSidebarMode] = useState<'entities' | 'graph' | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  
  const exportRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const graphRef = useRef<KnowledgeGraphHandle | null>(null);

  // close the export menu on outside click
  useEffect(() => {
    if (!exportOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [exportOpen]);

  // Check localStorage queue for papers added from PaperPage
  useEffect(() => {
    const queueKey = 'phytoquery_mypapers_queue';
    const raw = localStorage.getItem(queueKey);
    if (!raw) return;
    try {
      const queued: UploadedPaper[] = JSON.parse(raw);
      if (queued.length > 0) {
        addPapers(queued);
        // Only auto-select if nothing is currently selected.
        if (!selectedPaperId) {
          setSelectedPaperId(queued[0].id);
        }
        localStorage.removeItem(queueKey);
      }
    } catch {
      // Ignore malformed queue
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    return () => {
      if (viewerSrc) {
        URL.revokeObjectURL(viewerSrc);
      }
    };
  }, [viewerSrc]);

  // Auto-load PDF when active paper changes is moved down


  const closeViewer = () => {
    if (viewerSrc) {
      URL.revokeObjectURL(viewerSrc);
    }
    setViewerSrc(null);
    setViewerError(null);
    setIsViewerLoading(false);
    setViewerLoadingPaperId(null);
    setViewerPaper(null);
  };

  const openViewer = async (paper: UploadedPaper) => {
    if (!paper.pdfUrl) return;

    if (viewerSrc) {
      URL.revokeObjectURL(viewerSrc);
      setViewerSrc(null);
    }

    setViewerPaper(paper);
    setViewerError(null);
    setIsViewerLoading(true);
    setViewerLoadingPaperId(paper.id);

    try {
      const res = await fetch(paper.pdfUrl);

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || 'Failed to load PDF preview');
      }

      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      setViewerSrc(objectUrl);
    } catch (error) {
      console.error('Failed to open PDF viewer:', error);
      setViewerError('Unable to load this PDF preview right now.');
    } finally {
      setIsViewerLoading(false);
      setViewerLoadingPaperId(null);
    }
  };

  const deletePaper = async (paper: UploadedPaper) => {
    if (!paper.pdfUrl) {
      // Just remove from store if no backend PDF — the store action
      // also drops it from compareSelection and clears selectedPaperId
      // if this was the selected one.
      removePaper(paper.id);
      return;
    }

    try {
      const storedFilename = paper.pdfUrl.split('/').pop()?.split('?')[0];
      if (storedFilename) {
        const res = await fetch(`/ner/uploaded/${storedFilename}`, {
          method: 'DELETE',
          credentials: 'same-origin',
        });
        if (!res.ok) {
          console.error('Failed to delete PDF from backend:', await res.text());
        }
      }
    } catch (err) {
      console.error('Delete paper failed:', err);
    } finally {
      removePaper(paper.id);
    }
  };

  const processFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) return;

    setIsUploading(true);
    setUploadProgress({ current: 0, total: pdfFiles.length });
    setIsCompareMode(false);
    clearCompareSelection();

    const uploadedPapers: UploadedPaper[] = [];

    for (let i = 0; i < pdfFiles.length; i++) {
      const file = pdfFiles[i];
      setUploadProgress({ current: i + 1, total: pdfFiles.length });

      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/ner/upload/json', {
          method: 'POST',
          credentials: 'same-origin',
          body: formData,
        });
        if (!res.ok) continue;
        const data = await res.json();

        const paper: UploadedPaper = {
          id: `${Date.now()}_${i}`,
          name: data.metadata.title || file.name,
          doi: data.metadata.doi,
          pdfUrl: data.pdf_url || null,
          entities: data.entities,
          entity_counts: data.entity_counts || {},
          entity_count: data.entity_count,
        };
        uploadedPapers.push(paper);
      } catch {
        // Skip failed uploads
      }
    }

    if (uploadedPapers.length > 0) {
      addPapers(uploadedPapers);
      setSelectedPaperId(uploadedPapers[0].id);
      const initial: Record<string, boolean> = {};
      ENTITY_GROUP_ORDER.forEach(k => initial[k] = false);
      setExpandedGroups(initial);
    }

    setIsUploading(false);
    setUploadProgress({ current: 0, total: 0 });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    processFiles(e.target.files);
    e.target.value = ''; // Reset input so same files can be selected again
  };

  const activePapers = useMemo(() => {
    if (isCompareMode && compareSelection.length >= 2) {
      const selected = new Set(compareSelection);
      return papers.filter(p => selected.has(p.id));
    }
    return selectedPaper ? [selectedPaper] : [];
  }, [isCompareMode, compareSelection, selectedPaper, papers]);

  const groupedEntities = useMemo(() => {
    if (activePapers.length === 0) return [];

    const mergedCounts: Record<string, Map<string, { text: string; count: number; papers: Set<string>; canonical: string; maxSingleCount: number }>> = {};

    activePapers.forEach(paper => {
      Object.entries(paper.entity_counts).forEach(([label, items]) => {
        if (!mergedCounts[label]) mergedCounts[label] = new Map();
        items.forEach(item => {
          const canonical = item.canonical || item.text;
          const key = canonical.toLowerCase();
          const existing = mergedCounts[label].get(key);
          if (existing) {
            existing.count += item.count;
            existing.papers.add(paper.id);
            // Keep display text from the paper with highest individual count for this canonical
            if (item.count > existing.maxSingleCount) {
              existing.maxSingleCount = item.count;
              existing.text = item.text;
            }
          } else {
            mergedCounts[label].set(key, {
              text: item.text,
              count: item.count,
              papers: new Set([paper.id]),
              canonical,
              maxSingleCount: item.count,
            });
          }
        });
      });
    });

    return ENTITY_GROUP_ORDER.map((label) => {
      const itemsMap = mergedCounts[label];
      if (!itemsMap || itemsMap.size === 0) {
        return { label, items: [] as { text: string; count: number; papers: Set<string> }[], totalCount: 0, termCount: 0 };
      }
      const sortedItems = Array.from(itemsMap.values()).sort((a, b) => b.count - a.count);
      return {
        label,
        items: sortedItems,
        totalCount: sortedItems.reduce((sum, item) => sum + item.count, 0),
        termCount: sortedItems.length,
      };
    });
  }, [activePapers]);

  const { graphEntities, paperIdentifiers, entityPaperMap } = useMemo(() => {
    if (activePapers.length === 0) {
      return { graphEntities: [] as Entity[], paperIdentifiers: [] as { type: string; value: string }[], entityPaperMap: {} as Record<string, string[]> };
    }

    const allEntities: (Entity & { count: number })[] = [];
    const paperMap: Record<string, string[]> = {};
    const pids: { type: string; value: string }[] = [];

    activePapers.forEach(paper => {
      pids.push({ type: 'doi', value: paper.doi || paper.id });
      Object.entries(paper.entity_counts).forEach(([label, items]) => {
        items.forEach(item => {
          allEntities.push({ text: item.text, label, score: 1, count: item.count });
          const canonical = item.canonical || item.text;
          const key = `${label}-${canonical.toLowerCase()}`;
          if (!paperMap[key]) paperMap[key] = [];
          const paperValue = paper.doi || paper.id;
          if (!paperMap[key].includes(paperValue)) {
            paperMap[key].push(paperValue);
          }
        });
      });
    });

    return { graphEntities: allEntities, paperIdentifiers: pids, entityPaperMap: paperMap };
  }, [activePapers]);

  const exportCSV = () => {
    if (activePapers.length === 0) return;
    const lines = ['Entity Type,Entity Name,Count,Papers'];
    groupedEntities.forEach(group => {
      group.items.forEach(item => {
        const paperNames = activePapers
          .filter(p => item.papers.has(p.id))
          .map(p => p.name.replace(/"/g, '""'))
          .join('; ');
        lines.push(`"${group.label}","${item.text.replace(/"/g, '""')}",${item.count},"${paperNames}"`);
      });
    });
    const csv = lines.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const filename = isCompareMode
      ? `compared_entities_${activePapers.length}_papers`
      : (activePapers[0]?.name.replace(/[^a-zA-Z0-9]/g, '_') || 'entities');
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const exportGraph = () => {
    if (activePapers.length === 0) return;
    // Exact copy when the live graph is mounted; plain fallback otherwise.
    if (graphRef.current?.exportSnapshot()) {
      return;
    }
    const nodes: any[] = graphEntities.map(e => ({
      id: `${e.label}-${e.text.toLowerCase()}`,
      label: e.text,
      group: e.label,
    }));
    const edges: { from: string, to: string }[] = [];
    activePapers.forEach(paper => {
      const pid = `paper-${paper.doi || paper.id}`;
      nodes.push({ id: pid, label: paper.name, group: 'PAPER' });
      Object.entries(paper.entity_counts).forEach(([label, items]) => {
        items.forEach(item => {
          edges.push({
            from: pid,
            to: `${label}-${(item.canonical || item.text).toLowerCase()}`
          });
        });
      });
    });

    // ponytail: standalone offline HTML export (no CDN), shared helper.
    const title = isCompareMode ? `${activePapers.length} Papers` : activePapers[0]?.name || 'Graph';
    void downloadGraphHtml({
      nodes,
      edges,
      filename: `graph_export.html`,
      title: `Knowledge Graph - ${title}`,
      subtitle: `Entities linked to ${title}`,
    });
  };

  const entityConfig: Record<string, { accentVar: string }> = {};
  ENTITY_GROUP_ORDER.forEach((label) => {
    entityConfig[label] = { accentVar: getEntityAccentVar(label) };
  });

  const isComparing = isCompareMode && compareSelection.length >= 2;

  // Auto-load PDF when active paper changes
  useEffect(() => {
    if (!isCompareMode && activePapers.length === 1) {
      const paper = activePapers[0];
      if (paper.pdfUrl && paper.id !== viewerPaper?.id && paper.id !== viewerLoadingPaperId) {
        void openViewer(paper);
      } else if (!paper.pdfUrl) {
        closeViewer();
      }
    } else if (isCompareMode) {
      closeViewer();
    }
  }, [activePapers, isCompareMode, viewerPaper?.id, viewerLoadingPaperId]);

  return (
    <div className="flex h-screen bg-background text-on-background overflow-hidden flex-col">
      <main className="flex flex-1 overflow-hidden">
        
        {/* Left Sidebar: PDF Library & Actions */}
        {papers.length > 0 && (
          <aside className={`sidebar-transition border-r border-outline-variant bg-surface-bright flex flex-col shrink-0 relative ${leftSidebarCollapsed ? 'sidebar-collapsed' : 'w-72'}`} id="library-sidebar">
            {/* Unified Top Header Bar */}
            <div className="px-3.5 h-14 border-b border-outline-variant/40 flex items-center justify-between shrink-0">
              <span className={`!text-[17px] !font-bold text-slate-900 tracking-tight whitespace-nowrap transition-opacity duration-200 ${leftSidebarCollapsed ? 'hidden' : 'block'}`}>
                Sources
              </span>
              <button 
                className={`p-1.5 text-slate-600 hover:text-slate-900 rounded-md hover:bg-surface-low transition-colors outline-none ${leftSidebarCollapsed ? 'mx-auto' : ''}`} 
                onClick={() => setLeftSidebarCollapsed(!leftSidebarCollapsed)}
                title={leftSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              >
                <SidebarSimple size={20} />
              </button>
            </div>
            
            {/* Mini View (Icons only) */}
            <div className="sidebar-mini-view flex-col items-center pt-3 gap-2.5 h-full w-full overflow-y-auto custom-scrollbar pb-6">
              {/* Mini Action Buttons */}
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="p-2 rounded-lg text-slate-700 hover:bg-slate-100 hover:text-slate-900 transition-colors flex items-center justify-center"
                title="Add sources"
              >
                {isUploading ? (
                  <SpinnerGap size={20} className="animate-spin text-slate-900" />
                ) : (
                  <Plus size={20} weight="bold" />
                )}
              </button>

              {papers.length >= 2 && (
                <button
                  type="button"
                  onClick={() => {
                    if (isCompareMode) {
                      setIsCompareMode(false);
                      clearCompareSelection();
                    } else {
                      setIsCompareMode(true);
                      if (selectedPaper) setCompareSelection([selectedPaper.id]);
                    }
                  }}
                  className={`p-2 rounded-lg transition-colors flex items-center justify-center ${
                    isCompareMode
                      ? 'bg-violet-50 text-violet-700'
                      : 'text-slate-700 hover:bg-slate-100 hover:text-slate-900'
                  }`}
                  title={isCompareMode ? 'Cancel Compare' : 'Compare'}
                >
                  <ChartBar size={20} weight="bold" />
                </button>
              )}

              {/* Divider before paper stack */}
              <div className="w-6 h-[1.5px] bg-slate-200 rounded-full mx-auto my-1.5" />

              {papers.map((paper) => {
                const isSelected = selectedPaper?.id === paper.id;
                const isInCompare = compareSelection.includes(paper.id);
                const active = (isSelected && !isCompareMode) || (isInCompare && isCompareMode);
                return (
                  <button 
                    key={paper.id}
                    title={paper.name}
                    onClick={() => {
                        if (isCompareMode) {
                          toggleCompareSelection(paper.id);
                        } else {
                          setSelectedPaperId(paper.id);
                          const initial: Record<string, boolean> = {};
                          ENTITY_GROUP_ORDER.forEach(k => initial[k] = false);
                          setExpandedGroups(initial);
                        }
                    }}
                    className={`w-11 h-11 rounded-2xl flex items-center justify-center transition-all ${active ? 'bg-[#edf2fa] opacity-100' : 'hover:bg-slate-50 opacity-70 hover:opacity-100'}`}
                  >
                    <PdfIcon size={26} />
                  </button>
                );
              })}
            </div>

            {/* Main Sidebar Content */}
            <div className={`sidebar-content h-full flex flex-col overflow-hidden w-72 transition-opacity duration-200 ${leftSidebarCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
              <div className="pt-3 space-y-2 px-4 pb-3">
                <label className="block relative">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    multiple
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <div className="w-full py-2 px-4 bg-white border border-slate-200 text-slate-900 hover:bg-slate-50 rounded-full flex items-center justify-center gap-2 text-sm font-medium transition-colors cursor-pointer">
                    {isUploading ? (
                      <SpinnerGap size={16} className="animate-spin text-slate-900" />
                    ) : (
                      <Plus size={16} weight="bold" className="text-slate-900" />
                    )}
                    {isUploading ? `Uploading ${uploadProgress.current}/${uploadProgress.total}` : 'Add sources'}
                  </div>
                </label>
                
                {papers.length >= 2 && (
                  <button 
                    onClick={() => {
                      if (isCompareMode) {
                        setIsCompareMode(false);
                        clearCompareSelection();
                      } else {
                        setIsCompareMode(true);
                        if (selectedPaper) setCompareSelection([selectedPaper.id]);
                      }
                    }}
                    className={`w-full py-2 px-4 rounded-full flex items-center justify-center gap-2 text-sm transition-colors border ${
                      isCompareMode 
                        ? 'bg-violet-50 border-violet-200 text-violet-700 font-semibold' 
                        : 'bg-white border-slate-200 text-slate-900 hover:bg-slate-50 font-medium'
                    }`}
                  >
                    {isCompareMode ? <X size={16} weight="bold" /> : <ChartBar size={16} weight="bold" className="text-slate-900" />}
                    {isCompareMode ? 'Cancel Compare' : 'Compare'}
                  </button>
                )}
              </div>

              {/* Subtle divider before paper stack in uncollapsed state */}
              <div className="mx-4 mb-2.5 h-[1px] bg-slate-200/60 shrink-0" />

              <div className="flex-1 overflow-y-auto custom-scrollbar px-3 pb-6 space-y-1.5">
                {papers.map((paper) => {
                  const isSelected = selectedPaper?.id === paper.id;
                  const isInCompare = compareSelection.includes(paper.id);
                  const active = (isSelected && !isCompareMode) || (isInCompare && isCompareMode);

                  return (
                    <div 
                      key={paper.id}
                      onClick={() => {
                        if (isCompareMode) {
                          toggleCompareSelection(paper.id);
                        } else {
                          setSelectedPaperId(paper.id);
                          const initial: Record<string, boolean> = {};
                          ENTITY_GROUP_ORDER.forEach(k => initial[k] = false);
                          setExpandedGroups(initial);
                        }
                      }}
                      className={`p-3 relative group cursor-pointer transition-all flex items-center gap-3 rounded-2xl ${active ? 'bg-[#edf2fa] opacity-100' : 'bg-white hover:bg-slate-50 opacity-70 hover:opacity-100'}`}
                    >
                      <PdfIcon size={24} className="shrink-0" />
                      <div className="flex-1 min-w-0 pr-6">
                        <h4 className={`text-[14px] line-clamp-2 leading-tight ${active ? 'font-bold text-slate-900' : 'font-medium text-slate-700'}`}>{paper.name}</h4>
                      </div>
                      
                      {/* Delete Menu Trigger */}
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          void deletePaper(paper);
                        }}
                        className="absolute right-3 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 p-1 text-slate-400 hover:text-red-500 transition-colors"
                        title="Delete paper"
                      >
                        <TrashSimple size={16} weight="regular" />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          </aside>
        )}

        {/* Center Viewport: Document Reader or Upload Hero */}
        <section className="flex-1 bg-surface-dim overflow-y-auto custom-scrollbar flex flex-col items-center relative">
          {papers.length === 0 ? (
            <div className="flex-1 relative dot-pattern flex flex-col items-center justify-center overflow-hidden w-full h-full bg-background">
              {/* Abstract Background Shapes */}
              <div className="absolute top-[20%] left-[10%] w-64 h-64 bg-teal-500 opacity-10 blur-[100px] rounded-full pointer-events-none" />
              <div className="absolute bottom-[20%] right-[10%] w-96 h-96 bg-purple-500 opacity-10 blur-[120px] rounded-full pointer-events-none" />
              
              <div className="relative z-10 w-full max-w-2xl px-6">
                <label 
                  className={`upload-dashed pulse-border relative w-full aspect-[1.6/1] bg-surface-lowest/50 flex flex-col items-center justify-center transition-all duration-500 cursor-pointer group hover:bg-surface-c/80 hover:scale-[1.005] ${isUploading ? 'bg-surface-high' : ''}`}
                >
                  <input
                    type="file"
                    accept=".pdf"
                    multiple
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  
                  {!isUploading ? (
                    <div className="flex flex-col items-center transition-all duration-500 relative z-10">
                      <div className="mb-6 group-hover:scale-110 transition-transform duration-500">
                        <FileArrowUp size={64} weight="light" className="text-teal-600" />
                      </div>
                      <h1 className="text-2xl font-bold text-primary mb-2 tracking-tight font-display">Upload Research Papers</h1>
                      <p className="text-base text-on-surface-variant">Analyze documents and extract structured entities automatically.</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center w-full max-w-md relative z-10">
                      <div className="w-20 h-20 bg-slate-100 flex items-center justify-center rounded-xl mb-6 shadow-sm border border-slate-200">
                        <SpinnerGap size={40} className="text-slate-900 animate-spin" />
                      </div>
                      <h2 className="text-xl font-semibold text-slate-900 mb-1">Processing Papers...</h2>
                      <p className="text-sm text-on-surface-variant mb-8">
                        {uploadProgress.current} of {uploadProgress.total} documents
                      </p>
                      <div className="w-full h-1.5 bg-surface-variant rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-slate-900 transition-all duration-300 ease-out" 
                          style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}
                </label>
              </div>
            </div>
          ) : activePapers.length > 0 ? (
            <>
              {/* Doc Header Toolbar */}
              <div className="w-full bg-surface-bright border-b border-outline-variant px-6 py-4 sticky top-0 z-40 shadow-sm flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {isCompareMode ? (
                    <h3 className="font-serif font-semibold text-on-surface">Comparing {activePapers.length} Papers</h3>
                  ) : (
                    <div className="flex flex-col">
                      <h3 className="font-serif font-semibold text-on-surface truncate max-w-2xl">{activePapers[0]?.name}</h3>
                      {activePapers[0]?.doi && (
                        <span className="text-[11px] text-on-surface-variant mt-0.5">{activePapers[0].doi}</span>
                      )}
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-4">
                  {/* Export removed from here, now in right sidebar */}
                </div>
              </div>
              
              {/* Main Content Area */}
              <div className="flex-1 w-full flex flex-col min-h-0">
                {isCompareMode && activePapers.length >= 2 ? (
                  <CompareMatrix papers={activePapers} />
                ) : (
                  <div className="flex-1 bg-white border border-outline-variant overflow-hidden flex flex-col w-full h-[min(85vh,1200px)] relative">
                    {isViewerLoading ? (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-surface-c/50 backdrop-blur-sm z-10 gap-3.5" style={{ fontFamily: 'var(--font-google-sans)' }}>
                        <SpinnerGap size={46} className="animate-spin text-slate-900" />
                        <span className="text-[17px] text-on-surface-variant font-medium">Loading...</span>
                      </div>
                    ) : viewerError ? (
                      <div className="absolute inset-0 flex items-center justify-center bg-red-50 text-red-600 text-sm font-medium">
                        {viewerError}
                      </div>
                    ) : null}
                    <iframe
                      src={viewerSrc || undefined}
                      title={`PDF viewer`}
                      className="w-full h-full border-none"
                    />
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center text-on-surface-muted text-sm">
              {isCompareMode
                ? 'Select at least 2 papers to compare'
                : 'Select a paper to view its analysis'}
            </div>
          )}
        </section>

        {/* Right Sidebar: Insights & Entity Index */}
        {activePapers.length > 0 && (
          <aside className={`sidebar-transition border-l border-outline-variant bg-surface flex flex-col shrink-0 relative ${rightSidebarCollapsed ? 'sidebar-collapsed' : 'w-[400px]'}`} id="insights-sidebar">
            {/* Integrated Sidebar Toggle (Top Left) */}
            <button 
              className="toggle-btn-right absolute left-3 top-3 z-50 p-1.5 text-on-surface-variant rounded-full flex items-center justify-center hover:bg-surface-low transition-all outline-none focus:outline-none focus:ring-0"
              onClick={() => setRightSidebarCollapsed(!rightSidebarCollapsed)}
            >
              <SidebarSimple size={20} className="rotate-180" />
            </button>

            {/* Mini View (Icons only) */}
            <div className={`sidebar-mini-view ${rightSidebarCollapsed ? 'flex opacity-100' : 'hidden opacity-0 pointer-events-none'} flex-col items-center pt-16 gap-6 h-full w-full transition-opacity duration-200`}>
              {!isCompareMode && (
                <button onClick={() => { setRightSidebarMode('entities'); setRightSidebarCollapsed(false); }} className="p-2 rounded-full hover:bg-surface-c transition-colors" title="Entity Index">
                  <ListNumbers className={`text-xl ${rightSidebarMode === 'entities' ? 'text-primary' : 'text-on-surface-variant'}`} />
                </button>
              )}
              <button onClick={() => { setRightSidebarMode('graph'); setRightSidebarCollapsed(false); }} className="p-2 rounded-full hover:bg-surface-c transition-colors" title="Graph View">
                <Graph className={`text-xl ${rightSidebarMode === 'graph' ? 'text-primary' : 'text-on-surface-variant'}`} />
              </button>
            </div>

            {/* Main Sidebar Content */}
            <div className={`sidebar-content h-full flex flex-col overflow-hidden w-[400px] transition-opacity duration-200 ${rightSidebarCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
              <div style={{ position: "relative", display: "flex", justifyContent: "center", minHeight: 44, marginBottom: 14, marginTop: 56 }}>
                {!isCompareMode ? (
                  <div style={{
                    display: "inline-flex", gap: 2,
                    padding: 5, borderRadius: 999,
                    border: "none", background: "var(--surface-c)"
                  }}>
                    {(
                      [
                        { id: 'entities', icon: ListNumbers, label: 'Entity Index' },
                        { id: 'graph', icon: Graph, label: 'Graph View' }
                      ] as const
                    ).map(({ id, icon: Icon, label }) => {
                      const isActive = rightSidebarMode === id;
                      const expanded = hoverRightSidebarMode === id;
                      return (
                        <button key={id}
                          data-tab={id}
                          onClick={() => setRightSidebarMode(id)}
                          onMouseEnter={() => setHoverRightSidebarMode(id)}
                          onMouseLeave={() => setHoverRightSidebarMode(null)}
                          style={{
                            display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 7,
                            height: 34, borderRadius: 999,
                            width: expanded ? "auto" : 38,
                            padding: expanded ? "0 14px" : 0,
                            background: isActive ? "#FFFFFF" : "transparent",
                            boxShadow: isActive ? "0 1px 2px rgba(0,0,0,.08)" : "none",
                            border: "none", cursor: "pointer",
                            fontSize: 13, fontWeight: isActive ? 600 : 500,
                            color: isActive ? "var(--on-surface)" : "var(--on-surface-variant)",
                            whiteSpace: "nowrap", overflow: "hidden",
                            transition: "width .22s ease, padding .22s ease, background .15s, color .15s"
                          }}>
                            {expanded ? <span>{label}</span> : <Icon size={16} />}
                          </button>
                      );
                    })}
                  </div>
                ) : (
                  <div className="flex items-center justify-center">
                    <span className="text-[14px] font-semibold text-slate-800 tracking-tight">Graph View</span>
                  </div>
                )}

                {/* Export options pinned right */}
                <div
                  ref={exportRef}
                  style={{ position: "absolute", right: 0, top: 0 }}
                >
                  <button 
                    onClick={() => setExportOpen((o) => !o)} 
                    title="Export options" 
                    style={{
                      width: 40, height: 40, borderRadius: 999,
                      background: exportOpen ? "var(--surface-c)" : "transparent",
                      color: "var(--on-surface-variant)",
                      border: "none", cursor: "pointer",
                      display: "grid", placeItems: "center",
                      transition: "background .15s"
                    }}
                  >
                    <DotsThreeVertical size={20} weight="bold" />
                  </button>
                  {exportOpen && (
                    <div style={{
                      position: "absolute", top: "calc(100% + 4px)", right: 0,
                      width: "max-content",
                      background: "#FFFFFF",
                      border: "1px solid var(--border)", borderRadius: 12,
                      boxShadow: "0 2px 6px 2px rgba(0, 0, 0, 0.08)",
                      padding: 5, zIndex: 100,
                      animation: "fadeUp .16s ease"
                    }}>
                      <button 
                        className="export-opt" 
                        onClick={() => {
                          setExportOpen(false);
                          exportCSV();
                        }}
                      >
                        <span>Export CSV</span>
                      </button>
                      <button 
                        className="export-opt" 
                        onClick={() => {
                          setExportOpen(false);
                          exportGraph();
                        }}
                      >
                        <span>Export Graph</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Entity Index + Graph share one anchored scroll region, so switching
                  tabs or expanding groups never moves surrounding content */}
              <div className="flex-1 min-h-0 overflow-y-auto scrollbar-hide">
              {(!isCompareMode && rightSidebarMode === 'entities') && (
                <>
                  <div className="px-4 pb-8" style={{ display: "flex", flexDirection: "column" }}>
                    {groupedEntities.map((group) => {
                      const isExpanded = expandedGroups[group.label] !== false;
                      const accentColor = getEntityAccentColor(group.label);
                      const empty = group.termCount === 0;

                      return (
                        <div 
                          key={group.label} 
                          style={{ borderBottom: "1px solid var(--border)" }}
                        >
                          {/* Accordion Row (.ent-cat) */}
                          <button 
                            type="button"
                            className="ent-cat group"
                            onClick={empty ? undefined : () => {
                              toggleGroup(group.label);
                            }}
                            style={{
                              display: "flex", alignItems: "center", gap: 12,
                              width: "100%", padding: "13px 4px",
                              background: "transparent", border: "none",
                              cursor: empty ? "default" : "pointer",
                              opacity: empty ? 0.5 : 1
                            }}
                          >
                            {/* Left category accent dot bar */}
                            <span style={{
                              width: 4, height: 20, borderRadius: 999,
                              background: accentColor, flexShrink: 0,
                              opacity: empty ? 0.3 : 1
                            }} />
                            
                            <span style={{
                              flex: 1, textAlign: "left", fontSize: 14, fontWeight: 600,
                              color: isExpanded ? accentColor : "var(--on-surface)",
                              transition: "color .15s",
                              textTransform: "capitalize"
                            }}>
                              {LABEL_NAMES[group.label] || group.label.toLowerCase()}
                            </span>

                            {/* Hover caret transitions slot */}
                            <span style={{ flexShrink: 0 }}>
                              {empty ? (
                                <span style={{ fontSize: 14, color: "var(--on-surface-variant)" }}>–</span>
                              ) : isExpanded ? (
                                <span className="relative flex items-center justify-center min-w-[24px] h-[20px] text-on-surface-variant">
                                  <CaretUp size={16} weight="bold" />
                                </span>
                              ) : (
                                <span className="relative flex items-center justify-center min-w-[24px] h-[20px]">
                                  {/* Entity Count: visible without cursor, fades out on hover */}
                                  <span className="transition-all duration-200 text-[13px] font-semibold text-on-surface-variant group-hover:opacity-0 group-hover:scale-75">
                                    {group.termCount}
                                  </span>
                                  
                                  {/* Caret: animates into view on hover */}
                                  <span className="absolute inset-0 m-auto flex items-center justify-center transition-all duration-200 text-on-surface-variant opacity-0 scale-75 group-hover:opacity-100 group-hover:scale-100">
                                    <CaretDown size={16} weight="bold" />
                                  </span>
                                </span>
                              )}
                            </span>
                          </button>

                          {/* Values expanded container — animated open/close */}
                          {!empty && (
                            <div
                              className="grid transition-[grid-template-rows,opacity] duration-300 ease-out"
                              style={{
                                gridTemplateRows: isExpanded ? '1fr' : '0fr',
                                opacity: isExpanded ? 1 : 0,
                              }}
                            >
                            <div className="min-h-0 overflow-hidden">
                            <div style={{ padding: "0 4px 12px 16px", display: "flex", flexDirection: "column", gap: 2 }}>
                              {group.items.map((ent, eIdx) => {
                                const name = ent.text;
                                return (
                                  <div 
                                    key={eIdx}
                                    className="ent-row"
                                    style={{
                                      display: "flex", alignItems: "center", gap: 10,
                                      padding: "6px 10px", borderRadius: 7, cursor: "default",
                                      background: "transparent"
                                    }}
                                  >
                                    {/* Item dot */}
                                    <span style={{ width: 7, height: 7, borderRadius: 999, background: accentColor, flexShrink: 0 }} />
                                    
                                    {/* Item name */}
                                    <span style={{
                                      flex: 1, fontSize: 13.5,
                                      color: "var(--on-surface)",
                                      fontStyle: group.label === "SPECIES" ? "italic" : "normal"
                                    }}>
                                      {name}
                                    </span>

                                    <span className="mono" style={{ fontSize: 12, color: "var(--on-surface-variant)" }}>
                                      {ent.count}
                                    </span>
                                  </div>
                                );
                              })}
                            </div>
                            </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
              {/* Always mounted (hidden when inactive) so Export Graph snapshots the live view from any tab */}
              <div
                className="flex flex-col"
                style={{ display: !isCompareMode && rightSidebarMode === 'entities' ? 'none' : undefined }}
              >
                <KnowledgeGraph
                  ref={graphRef}
                  active={isCompareMode || rightSidebarMode === 'graph'}
                  entities={graphEntities}
                    paperIdentifier={isComparing ? undefined : { type: 'doi', value: activePapers[0]?.doi || '' }}
                    paperIdentifiers={isComparing ? paperIdentifiers : undefined}
                    entityConfig={entityConfig}
                    entityPaperMap={isComparing ? entityPaperMap : undefined}
                  />
                </div>
              </div>
            </div>
          </aside>
        )}
      </main>
    </div>
  );
};

export default AnalysePage;
