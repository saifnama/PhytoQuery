import { useState, useMemo } from 'react';
import { Plus, FlowerLotus, Table, ChartBar, X } from '@phosphor-icons/react';
import { KnowledgeGraph } from '../reader/KnowledgeGraph';
import type { Entity } from '../../types';

interface UploadedPaper {
  id: string;
  name: string;
  doi?: string;
  entities: Record<string, string[]>;
  entity_counts: Record<string, { text: string; count: number; canonical?: string; aliases?: string[] }[]>;
  entity_count: number;
}

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

const MyPapersPage = () => {
  const [papers, setPapers] = useState<UploadedPaper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<UploadedPaper | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const [isCompareMode, setIsCompareMode] = useState(false);
  const [compareSelection, setCompareSelection] = useState<Set<string>>(new Set());

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      const selectedFile = e.target.files[0];
      const formData = new FormData();
      formData.append('file', selectedFile);
      fetch('/ner/upload/json', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
          const paper: UploadedPaper = {
            id: Date.now().toString(),
            name: data.metadata.title || selectedFile.name,
            doi: data.metadata.doi,
            entities: data.entities,
            entity_counts: data.entity_counts || {},
            entity_count: data.entity_count,
          };
          setPapers(prev => [paper, ...prev]);
          setSelectedPaper(paper);
          const initial: Record<string, boolean> = {};
          ENTITY_GROUP_ORDER.forEach(k => initial[k] = false);
          setExpandedGroups(initial);
          setIsCompareMode(false);
          setCompareSelection(new Set());
        })
        .catch(() => {});
    }
  };

  const toggleComparePaper = (paperId: string) => {
    setCompareSelection(prev => {
      const next = new Set(prev);
      if (next.has(paperId)) {
        next.delete(paperId);
      } else {
        next.add(paperId);
      }
      return next;
    });
  };

  const toggleGroup = (label: string) => {
    setExpandedGroups(prev => ({ ...prev, [label]: !prev[label] }));
  };

  const activePapers = useMemo(() => {
    if (isCompareMode && compareSelection.size >= 2) {
      return papers.filter(p => compareSelection.has(p.id));
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
  const isComparing = isCompareMode && compareSelection.size >= 2;

  return (
    <div className="flex h-full">
      {/* Left Sidebar */}
      <div className="w-56 flex-shrink-0 border-r border-slate-200 bg-white flex flex-col">
        <div className="p-3 border-b border-slate-100">
          <h2 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
            <FlowerLotus size={18} className="text-pink-600" />
            My Papers
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">Upload PDFs · Extract entities</p>
        </div>

        <div className="px-3 py-2 space-y-2">
          {papers.length >= 2 && (
            <button
              onClick={() => {
                if (isCompareMode) {
                  setIsCompareMode(false);
                  setCompareSelection(new Set());
                } else {
                  setIsCompareMode(true);
                  if (selectedPaper) {
                    setCompareSelection(new Set([selectedPaper.id]));
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

          <label className="block">
            <input type="file" accept=".pdf" onChange={handleFileChange} className="hidden" />
            <div className="flex items-center justify-center gap-1.5 py-2 px-3 border border-dashed border-slate-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors text-xs text-blue-600 font-medium">
              <Plus size={14} />
              Upload PDF
            </div>
          </label>
        </div>

        <div className="flex-1 overflow-y-auto">
          {papers.map(paper => {
            const isSelected = selectedPaper?.id === paper.id;
            const isInCompare = compareSelection.has(paper.id);

            return (
              <div
                key={paper.id}
                onClick={() => {
                  if (isCompareMode) {
                    toggleComparePaper(paper.id);
                  } else {
                    setSelectedPaper(paper);
                    const initial: Record<string, boolean> = {};
                    ENTITY_GROUP_ORDER.forEach(k => initial[k] = false);
                    setExpandedGroups(initial);
                  }
                }}
                className={`flex items-center gap-2 px-3 py-2 cursor-pointer border-l-2 transition-colors ${
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
                <div className="w-6 h-7 rounded bg-red-100 flex items-center justify-center text-[10px] font-bold text-red-600 flex-shrink-0">PDF</div>
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-slate-900 truncate">{paper.name}</p>
                  <p className="text-[10px] text-slate-400 truncate">{paper.doi || 'No DOI'}</p>
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
                    <span>Export CSV</span>
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

              {/* Knowledge Graph - Right Side */}
              {showGraph && (
                <div className="flex-1 bg-white overflow-hidden">
                  <KnowledgeGraph
                    entities={graphEntities}
                    paperIdentifier={isComparing ? undefined : { type: 'doi', value: activePapers[0]?.doi || '' }}
                    paperIdentifiers={isComparing ? paperIdentifiers : undefined}
                    entityConfig={entityConfig}
                    entityPaperMap={isComparing ? entityPaperMap : undefined}
                  />
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
    </div>
  );
};

export default MyPapersPage;