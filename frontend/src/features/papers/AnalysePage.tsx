import { useState, useMemo, useRef, useEffect } from 'react';
import { Plus, FlowerLotus, Table, ChartBar, X, DotsThreeVertical, Trash } from '@phosphor-icons/react';
import { KnowledgeGraph } from '../reader/KnowledgeGraph';
import type { Entity } from '../../types';
import { useAnalyseStore, type UploadedPaper } from '../../stores/analyseStore';

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
  const papers = useAnalyseStore((s) => s.papers);
  const selectedPaperId = useAnalyseStore((s) => s.selectedPaperId);
  const expandedGroups = useAnalyseStore((s) => s.expandedGroups);
  const isCompareMode = useAnalyseStore((s) => s.isCompareMode);
  const compareSelection = useAnalyseStore((s) => s.compareSelection);
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
  const [activePaperMenu, setActivePaperMenu] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setActivePaperMenu(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
      setActivePaperMenu(null);
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
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const filename = isCompareMode
      ? `compared_entities_${activePapers.length}_papers`
      : (activePapers[0]?.name.replace(/[^a-zA-Z0-9]/g, '_') || 'entities');
    a.download = `${filename}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const entityConfig: Record<string, { accentVar: string }> = {};
  ENTITY_GROUP_ORDER.forEach((label) => {
    entityConfig[label] = { accentVar: getEntityAccentVar(label) };
  });

  const showGraph = graphEntities.length > 0;
  const isComparing = isCompareMode && compareSelection.length >= 2;

  return (
    <div className="flex h-full">
      {/* Left Sidebar */}
      <div className="w-56 flex-shrink-0 border-r border-slate-200 bg-white flex flex-col">
        <div className="p-3 border-b border-slate-100">
          <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
            <FlowerLotus size={18} className="text-pink-600" />
            Analyse
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">Upload PDFs · Extract entities</p>
        </div>

        <div className="px-3 py-2 space-y-2">
          {papers.length >= 2 && (
            <button
              onClick={() => {
                if (isCompareMode) {
                  setIsCompareMode(false);
                  clearCompareSelection();
                } else {
                  setIsCompareMode(true);
                  if (selectedPaper) {
                    setCompareSelection([selectedPaper.id]);
                  }
                }
              }}
              className={`flex items-center justify-center gap-1.5 w-full py-2 px-3 rounded-lg text-xs font-medium transition-colors ${
                isCompareMode
                  ? 'bg-violet-100 text-violet-700 border border-violet-200'
                  : 'border border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              {isCompareMode ? <X size={14} /> : <ChartBar size={14} />}
              {isCompareMode ? 'Cancel Compare' : 'Compare Papers'}
            </button>
          )}

          <label className="block relative">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              onChange={handleFileChange}
              className="hidden"
            />
            <div className={`flex items-center justify-center gap-1.5 py-2 px-3 border border-dashed rounded-lg cursor-pointer transition-colors text-xs font-medium ${
              isUploading
                ? 'border-blue-300 bg-blue-50 text-blue-700'
                : 'border-slate-300 hover:border-blue-400 hover:bg-blue-50 text-blue-600'
            }`}>
              {isUploading ? (
                <>
                  <svg className="animate-spin h-3.5 w-3.5 text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>{uploadProgress.current} / {uploadProgress.total}</span>
                </>
              ) : (
                <>
                  <Plus size={14} />
                  <span>Upload PDFs</span>
                </>
              )}
            </div>
          </label>
        </div>

        <div className="flex-1 overflow-y-auto">
          {papers.map(paper => {
            const isSelected = selectedPaper?.id === paper.id;
            const isInCompare = compareSelection.includes(paper.id);

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
                className={`group flex items-center gap-2 px-3 py-2 cursor-pointer border-l-2 transition-colors ${
                  isSelected && !isCompareMode
                    ? 'border-blue-500 bg-blue-50'
                    : isInCompare && isCompareMode
                    ? 'border-violet-500 bg-violet-50'
                    : 'border-transparent hover:bg-slate-50'
                }`}
              >
                {isCompareMode && (
                  <div className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 ${
                    isInCompare ? 'bg-violet-600 border-violet-600' : 'border-slate-300'
                  }`}>
                    {isInCompare && (
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                        <polyline points="1.5,5 4,7.5 8.5,2.5" stroke="#fff" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    )}
                  </div>
                )}
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    void openViewer(paper);
                  }}
                  disabled={!paper.pdfUrl}
                  title={paper.pdfUrl ? `Open PDF for ${paper.name}` : 'PDF viewer unavailable for this paper'}
                  className={`w-6 h-7 rounded flex items-center justify-center text-[10px] font-bold flex-shrink-0 transition-colors ${
                    paper.pdfUrl
                      ? 'bg-red-100 text-red-600 hover:bg-red-200 cursor-pointer'
                      : 'bg-slate-100 text-slate-300 cursor-not-allowed'
                  }`}
                >
                  {viewerLoadingPaperId === paper.id ? (
                    <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-red-200 border-t-red-600" />
                  ) : (
                    'PDF'
                  )}
                </button>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-slate-900 truncate">{paper.name}</p>
                  <p className="text-[10px] text-slate-400 truncate">{paper.doi || 'No DOI'}</p>
                </div>

                {/* Three-dots menu */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      setActivePaperMenu(activePaperMenu === paper.id ? null : paper.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 inline-flex h-6 w-6 items-center justify-center rounded text-slate-400 transition-opacity hover:text-slate-600"
                    title="Paper options"
                  >
                    <DotsThreeVertical size={16} weight="bold" />
                  </button>

                  {activePaperMenu === paper.id && (
                    <div
                      ref={menuRef}
                      className="absolute right-0 top-7 z-20 w-44 rounded-lg border border-slate-200 bg-white py-1 shadow-lg"
                    >
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          void deletePaper(paper);
                        }}
                        className="flex w-full items-center gap-2 px-3 py-2 text-xs text-red-600 hover:bg-red-50 transition-colors"
                      >
                        <Trash size={14} weight="bold" />
                        <span>Delete</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          {papers.length === 0 && (
            <p className="text-xs text-slate-400 text-center py-8">No papers uploaded</p>
          )}
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activePapers.length > 0 ? (
          <>
            <div className="border-b border-slate-200 p-4">
              {isComparing ? (
                <div>
                  <h3 className="text-base font-semibold text-slate-900 font-serif">
                    Comparing {activePapers.length} Papers
                  </h3>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {activePapers.map(p => (
                      <span key={p.id} className="text-[10px] bg-violet-50 text-violet-700 px-2 py-0.5 rounded-full border border-violet-100 truncate max-w-[200px]">
                        {p.name}
                      </span>
                    ))}
                  </div>
                </div>
              ) : (
                <>
                  <h3 className="text-base font-semibold text-slate-900 font-serif">{activePapers[0]?.name}</h3>
                  <p className="text-xs text-slate-500 mt-1 font-mono">{activePapers[0]?.doi || 'No DOI'}</p>
                </>
              )}
            </div>

            <div className="flex-1 flex overflow-hidden">
              {/* Entity Index */}
              <div className="w-[380px] flex-shrink-0 border-r border-slate-200 flex flex-col bg-slate-50/20 overflow-hidden">
                <div className="flex flex-col flex-1 p-8 space-y-6 overflow-y-auto custom-scrollbar">
                  <button
                    onClick={exportCSV}
                    className="w-full px-6 py-3 rounded-xl font-bold transition-all flex items-center justify-center space-x-2 text-[11px] uppercase tracking-widest shadow-lg bg-emerald-600 hover:bg-emerald-700 text-white shadow-emerald-100 cursor-pointer"
                  >
                    <Table size={16} weight="bold" />
                    <span>Export Entities</span>
                  </button>

                  <div className="flex items-center justify-between pb-3 border-b border-slate-200">
                    <span className="text-[14px] font-semibold text-slate-900 font-display">Entity Index</span>
                    <span className="text-[10px] text-slate-400">
                      {activePapers.reduce((sum, p) => sum + p.entity_count, 0).toLocaleString()} entities
                    </span>
                  </div>

                  <div className="flex-1 overflow-y-auto pr-2 -mr-2">
                    <div className="flex flex-col bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
                      {groupedEntities.map((group) => {
                        const isExpanded = expandedGroups[group.label] !== false;
                        const accentColor = getEntityAccentColor(group.label);
                        const accentRgbVar = getEntityAccentVar(group.label) + '-rgb';

                        return (
                          <div
                            key={group.label}
                            className="relative border-b-[0.5px] border-slate-200 last:border-0"
                            style={{ borderLeft: isExpanded ? `2.5px solid ${accentColor}` : '2.5px solid transparent' }}
                          >
                            <div
                              className="flex items-center gap-3 py-2.5 px-3 cursor-pointer transition-colors hover:bg-slate-50"
                              onClick={() => toggleGroup(group.label)}
                            >
                              <div
                                className="w-[3px] h-8 rounded-sm shrink-0"
                                style={{ backgroundColor: accentColor, opacity: group.termCount === 0 ? 0.3 : 1 }}
                              />
                              <div className="flex-1 flex flex-col gap-[2px] min-w-0">
                                <span 
                                  className="text-[11px] font-semibold text-slate-800 font-display truncate"
                                  style={{ color: isExpanded ? accentColor : undefined }}
                                >
                                  {LABEL_NAMES[group.label] || group.label}
                                </span>
                                {group.termCount > 0 && (
                                  <div
                                    className="h-2 w-full rounded-[2px] transition-opacity"
                                    style={{
                                      backgroundColor: `rgb(var(${accentRgbVar}) / 0.13)`,
                                      opacity: 1
                                    }}
                                  />
                                )}
                              </div>
                              <div className="flex items-center gap-1.5 shrink-0">
                                <span
                                  className="text-[10px] font-mono text-right min-w-[1.5rem]"
                                  style={{ color: group.termCount === 0 ? '#cbd5e1' : accentColor }}
                                >
                                  {group.termCount || '—'}
                                </span>
                                {group.termCount > 0 && (
                                  <button
                                    type="button"
                                    className={`text-[9px] text-slate-300 transition-transform duration-200 px-1.5 py-1 hover:text-slate-500 ${isExpanded ? 'rotate-90' : ''}`}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleGroup(group.label);
                                    }}
                                  >
                                    ▶
                                  </button>
                                )}
                              </div>
                            </div>

                            <div className={`grid transition-[grid-template-rows] duration-300 ease-in-out bg-white ${isExpanded && group.items.length > 0 ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}`}>
                              <div className="overflow-hidden min-h-0">
                                <div className="flex flex-col py-1 border-t-[0.5px] border-slate-100">
                                  {group.items.map((ent, eIdx) => (
                                    <div key={eIdx} className="flex flex-col border-b-[0.5px] border-slate-50 last:border-0">
                                      <div className="flex items-center gap-2 py-1.5 pr-3 pl-8 transition-colors hover:bg-slate-50">
                                        <div className="w-[6px] h-[6px] rounded-full shrink-0" style={{ backgroundColor: accentColor }} />
                                        <div className="flex-1 min-w-0 pr-2">
                                          <span 
                                            className="text-[11px] font-display block truncate text-slate-600"
                                          >
                                            {ent.text}
                                          </span>
                                        </div>
                                        <span className="text-[9px] font-mono text-slate-900 font-semibold">{ent.count}</span>
                                      </div>
                                      {isComparing && ent.papers.size > 0 && (
                                        <div className="pl-8 pr-3 pb-1 flex flex-wrap gap-1">
                                          {activePapers
                                            .filter(p => ent.papers.has(p.id))
                                            .map(p => (
                                              <span key={p.id} className="text-[9px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded truncate max-w-[120px]" title={p.name}>
                                                {p.name}
                                              </span>
                                            ))}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>

              {/* Knowledge Graph — Right Side */}
              {showGraph && (
                <div className="flex-1 bg-white overflow-hidden flex flex-col">
                  <div className="flex items-center justify-between px-4 py-2 border-b border-slate-100">
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Entity View</span>
                    {isComparing && (
                      <span className="text-[10px] text-slate-400">Compare mode</span>
                    )}
                  </div>

                  <div className="flex-1 overflow-hidden">
                    <KnowledgeGraph
                      entities={graphEntities}
                      paperIdentifier={isComparing ? undefined : { type: 'doi', value: activePapers[0]?.doi || '' }}
                      paperIdentifiers={isComparing ? paperIdentifiers : undefined}
                      entityConfig={entityConfig}
                      entityPaperMap={isComparing ? entityPaperMap : undefined}
                    />
                  </div>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
            {isCompareMode
              ? 'Select at least 2 papers to compare'
              : 'Select a paper or upload a new PDF'}
          </div>
        )}
      </div>

      {viewerPaper && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4" onClick={closeViewer}>
          <div
            className="flex h-[min(92vh,56rem)] w-[min(96vw,72rem)] flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
              <div className="min-w-0">
                <h3 className="truncate text-sm font-semibold text-slate-900">{viewerPaper.name}</h3>
              </div>
              <button
                type="button"
                onClick={closeViewer}
                className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-slate-200 text-slate-500 transition-colors hover:border-slate-300 hover:text-slate-700"
                title="Close PDF viewer"
              >
                <X size={18} weight="bold" />
              </button>
            </div>

            <div className="flex-1 bg-slate-100 p-3">
              {isViewerLoading ? (
                <div className="flex h-full items-center justify-center rounded-xl border border-slate-200 bg-white text-sm text-slate-500">
                  <div className="flex items-center gap-3">
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-slate-700" />
                    <span>Loading PDF preview...</span>
                  </div>
                </div>
              ) : viewerError ? (
                <div className="flex h-full items-center justify-center rounded-xl border border-red-100 bg-white px-6 text-center text-sm text-red-500">
                  {viewerError}
                </div>
              ) : (
                <iframe
                  src={viewerSrc || undefined}
                  title={`PDF viewer for ${viewerPaper.name}`}
                  className="h-full w-full rounded-xl border border-slate-200 bg-white"
                />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysePage;
