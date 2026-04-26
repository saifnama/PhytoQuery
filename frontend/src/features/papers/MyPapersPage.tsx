import { useState, useMemo } from 'react';
import { Plus, FlowerLotus, Table } from '@phosphor-icons/react';
import { KnowledgeGraph } from '../reader/KnowledgeGraph';
import type { Entity } from '../../types';

interface UploadedPaper {
  id: string;
  name: string;
  doi?: string;
  entities: Record<string, string[]>;
  entity_counts: Record<string, { text: string; count: number }[]>;
  entity_count: number;
}

const ENT_COLORS: Record<string, string> = {
  CHEMICAL: '#2563EB',         // blue
  SPECIES: '#16A34A',          // green
  PLANT_PART: '#84CC16',       // lime green
  EXTRACTION_METHOD: '#7C3AED', // purple
  DEVELOPMENT_STAGE: '#F97316',  // orange
  SEASON: '#EAB308',           // yellow
  LOCATION: '#06B6D4',        // cyan
  BIOACTIVITY: '#EC4899',       // pink
  ANALYTICAL_TECHNIQUE: '#64748B', // slate
  DISEASE: '#DC2626',          // red
};

// Entity label display names - matching search page
const LABEL_NAMES: Record<string, string> = {
  CHEMICAL: 'Chemical',
  SPECIES: 'Species',
  PLANT_PART: 'Plant Part',
  ANALYTICAL_TECHNIQUE: 'Analytical Technique',
  EXTRACTION_METHOD: 'Extraction Method',
  BIOACTIVITY: 'Bioactivity',
  DEVELOPMENT_STAGE: 'Development Stage',
  SEASON: 'Season',
};

const MyPapersPage = () => {
  const [papers, setPapers] = useState<UploadedPaper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<UploadedPaper | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});

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
          // Expand all groups by default
          const all: Record<string, boolean> = {};
          Object.keys(data.entities).forEach(k => all[k] = true);
          setExpandedGroups(all);
        })
        .catch(() => {});
    }
  };

  const toggleGroup = (label: string) => {
    setExpandedGroups(prev => ({ ...prev, [label]: !prev[label] }));
  };

  const toggleAll = () => {
    const allExpanded = Object.values(expandedGroups).every(v => v);
    const all: Record<string, boolean> = {};
    Object.keys(selectedPaper?.entities || {}).forEach(k => all[k] = !allExpanded);
    setExpandedGroups(all);
  };

  // Group entities with counts
  const groupedEntities = useMemo(() => {
    if (!selectedPaper?.entity_counts) return [];
    return Object.entries(selectedPaper.entity_counts).map(([label, items]) => ({
      label,
      items: items.map(item => ({ text: item.text, count: item.count })),
      totalCount: items.reduce((sum, item) => sum + item.count, 0),
      termCount: items.length,
    }));
  }, [selectedPaper]);

  // CSV export
  const exportCSV = () => {
    if (!selectedPaper) return;
    const lines = ['Entity Type,Entity Name,Count'];
    Object.entries(selectedPaper.entities).forEach(([label, texts]) => {
      texts.forEach(text => {
        lines.push(`"${label}","${text.replace(/"/g, '""')}",1`);
      });
    });
    const csv = lines.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedPaper.name.replace(/[^a-zA-Z0-9]/g, '_')}_entities.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const graphEntities: Entity[] = selectedPaper
    ? Object.entries(selectedPaper.entities).flatMap(([label, texts]) =>
        texts.map(text => ({ text, label, score: 1 }))
      )
    : [];

  const paperIdentifier = { type: 'doi', value: selectedPaper?.doi || '' };
  const entityConfig: Record<string, { accentVar: string }> = {};
  Object.keys(ENT_COLORS).forEach(k => { entityConfig[k] = { accentVar: `--entity-${k.toLowerCase().replace('_', '-')}` }; });

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

        <div className="px-3 py-2">
          <label className="block">
            <input type="file" accept=".pdf" onChange={handleFileChange} className="hidden" />
            <div className="flex items-center justify-center gap-1.5 py-2 px-3 border border-dashed border-slate-300 rounded-lg cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors text-xs text-blue-600 font-medium">
              <Plus size={14} />
              Upload PDF
            </div>
          </label>
        </div>

        <div className="flex-1 overflow-y-auto">
          {papers.map(paper => (
            <div
              key={paper.id}
              onClick={() => {
                setSelectedPaper(paper);
                const all: Record<string, boolean> = {};
                Object.keys(paper.entities).forEach(k => all[k] = true);
                setExpandedGroups(all);
              }}
              className={`flex items-center gap-2 px-3 py-2 cursor-pointer border-l-2 transition-colors ${
                selectedPaper?.id === paper.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-transparent hover:bg-slate-50'
              }`}
            >
              <div className="w-6 h-7 rounded bg-red-100 flex items-center justify-center text-[10px] font-bold text-red-600 flex-shrink-0">PDF</div>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-slate-900 truncate">{paper.name}</p>
                <p className="text-[10px] text-slate-400 truncate">{paper.doi || 'No DOI'}</p>
              </div>
            </div>
          ))}
          {papers.length === 0 && (
            <p className="text-xs text-slate-400 text-center py-8">No papers uploaded</p>
          )}
        </div>
      </div>

      {/* Main Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedPaper ? (
          <>
            {/* Header: Title + DOI */}
            <div className="border-b border-slate-200 p-4">
              <h3 className="text-base font-semibold text-slate-900 font-serif">{selectedPaper.name}</h3>
              <p className="text-xs text-slate-500 mt-1 font-mono">{selectedPaper.doi || 'No DOI'}</p>
            </div>

            {/* Entity Index + Knowledge Graph */}
            <div className="flex-1 flex overflow-hidden">
              {/* Entity Index Sidebar */}
              <div className="w-80 flex-shrink-0 border-r border-slate-200 flex flex-col bg-white overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between p-3 border-b border-slate-200">
                  <span className="text-sm font-semibold text-slate-900">Entity Index</span>
                  <button
                    onClick={toggleAll}
                    className="text-[10px] text-slate-500 hover:text-slate-700"
                  >
                    {Object.values(expandedGroups).every(v => v) ? 'Collapse All' : 'Expand All'}
                  </button>
                </div>

                {/* Entity Groups */}
                <div className="flex-1 overflow-y-auto p-2">
                  {groupedEntities.map(group => {
                    const isExpanded = expandedGroups[group.label] !== false;
                    const color = ENT_COLORS[group.label] || '#64748b';
                    
                    return (
                      <div key={group.label} className="mb-2">
                        {/* Group Header */}
                        <div
                          onClick={() => toggleGroup(group.label)}
                          className="flex items-center gap-2 py-2 px-2 rounded cursor-pointer hover:bg-slate-50"
                          style={{ borderLeft: `3px solid ${color}` }}
                        >
                          <span className="text-[10px] text-slate-400">▶</span>
                          <span className="flex-1 text-xs font-semibold text-slate-800 capitalize" style={{ color }}>
                            {LABEL_NAMES[group.label] || group.label.toLowerCase()}
                          </span>
                          <span className="text-[10px] font-mono text-slate-500">{group.totalCount}</span>
                        </div>
                        
                        {/* Entity List */}
                        {isExpanded && (
                          <div className="ml-4 flex flex-col gap-0.5 py-1">
                            {group.items.map((item, idx) => (
                              <div key={idx} className="flex items-center gap-2 py-1 px-2">
                                <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
                                <span className="flex-1 text-[11px] text-slate-600 truncate">{item.text}</span>
                                <span className="text-[10px] font-mono text-slate-400">{item.count}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* Export Button */}
                <div className="p-3 border-t border-slate-200">
                  <button
                    onClick={exportCSV}
                    className="flex items-center justify-center gap-1.5 w-full py-2 px-3 border border-slate-200 rounded-lg text-xs text-slate-600 hover:bg-slate-50"
                  >
                    <Table size={14} />
                    Export CSV
                  </button>
                </div>
              </div>

              {/* Knowledge Graph */}
              <div className="flex-1 overflow-hidden">
                <KnowledgeGraph
                  entities={graphEntities}
                  paperIdentifier={paperIdentifier}
                  entityConfig={entityConfig}
                />
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
            Select a paper or upload a new PDF
          </div>
        )}
      </div>
    </div>
  );
};

export default MyPapersPage;