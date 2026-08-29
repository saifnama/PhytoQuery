import { useMemo } from 'react';
import type { UploadedPaper } from '../../stores/analyseStore';

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

export const CompareMatrix = ({ papers }: { papers: UploadedPaper[] }) => {
  // Aggregate data
  const matrixData = useMemo(() => {
    // 1. Collect all unique entity names per category
    const categoryEntities: Record<string, Set<string>> = {};
    
    // 2. Map of (entityName + category) -> paperId -> count
    const entityCountsMap: Record<string, Record<string, number>> = {};
    
    // 3. Track universal max count across the dataset for universal opacity scaling
    let globalMaxCount = 1;
    
    // 4. Track total mentions per paper
    const paperTotals: Record<string, number> = {};
    papers.forEach(p => paperTotals[p.id] = 0);

    papers.forEach(paper => {
      if (!paper.entity_counts) return;
      
      Object.entries(paper.entity_counts).forEach(([category, entities]) => {
        if (!categoryEntities[category]) categoryEntities[category] = new Set();
        
        entities.forEach(entity => {
          const name = entity.canonical || entity.text;
          categoryEntities[category].add(name);
          
          const key = `${category}::${name}`;
          if (!entityCountsMap[key]) entityCountsMap[key] = {};
          
          const existingCount = entityCountsMap[key][paper.id] || 0;
          const totalCount = existingCount + entity.count;
          entityCountsMap[key][paper.id] = totalCount;
          
          if (totalCount > globalMaxCount) {
            globalMaxCount = totalCount;
          }
          
          paperTotals[paper.id] += entity.count;
        });
      });
    });
    
    // Prepare rows based on defined group order
    type RowType = 
      | { type: 'group', category: string } 
      | { type: 'entity', category: string, name: string, counts: Record<string, number>, numPapersWithMention: number };
      
    const aggregatedRows: RowType[] = [];
    
    for (const category of ENTITY_GROUP_ORDER) {
      const entities = categoryEntities[category];
      if (!entities || entities.size === 0) continue;
      
      aggregatedRows.push({ type: 'group', category });
      
      // Sort entities alphabetically for consistency
      const sortedEntities = Array.from(entities).sort((a, b) => a.localeCompare(b));
      
      sortedEntities.forEach(name => {
        const key = `${category}::${name}`;
        const counts = entityCountsMap[key];
        const numPapersWithMention = Object.keys(counts).length;
        
        aggregatedRows.push({
          type: 'entity',
          category,
          name,
          counts,
          numPapersWithMention
        });
      });
    }
    
    return { rows: aggregatedRows, globalMaxCount, paperTotals };
  }, [papers]);

  if (papers.length === 0) return null;

  return (
    <div className="flex-1 w-full max-w-[1400px] mx-auto p-4 md:p-6 flex flex-col min-h-0 h-full overflow-hidden">
      <div className="bg-white border border-outline-variant rounded-md overflow-hidden flex-1 flex flex-col min-h-0 shadow-sm">
        <div className="overflow-auto custom-scrollbar flex-1 relative">
          <table className="w-full text-left border-collapse table-fixed min-w-[1200px]">
            <thead className="sticky top-0 z-40">
              <tr className="bg-white border-b border-outline-variant shadow-[0_1px_2px_rgba(0,0,0,0.05)]">
                <th className="p-3 text-[10px] font-bold uppercase tracking-wider text-on-surface-variant w-56 sticky left-0 bg-white z-50 border-r border-outline-variant shadow-[2px_0_4px_rgba(0,0,0,0.03)]">
                  Entity Index
                </th>
                {papers.map((p) => (
                  <th key={p.id} className="p-3 text-[10px] font-medium text-center border-r border-outline-variant bg-white min-w-[160px]">
                    <div className="flex flex-col items-center gap-1">
                      <span className="font-bold text-on-surface line-clamp-2 leading-tight px-2" title={p.name}>{p.name}</span>
                      {p.doi && <span className="opacity-70 text-[9px]">{p.doi}</span>}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-xs text-on-surface-variant">
              {matrixData.rows.map((row) => {
                if (row.type === 'group') {
                  const accentColor = `var(${getEntityAccentVar(row.category)}, var(--primary))`;
                  return (
                    <tr 
                      key={`group-${row.category}`} 
                      style={{ backgroundColor: `color-mix(in srgb, ${accentColor} 8%, #ffffff)` }}
                    >
                      <td 
                        className="px-4 py-2 font-bold text-[10px] uppercase tracking-widest border-b border-r border-outline-variant/60 sticky left-0 z-30 shadow-[2px_0_4px_rgba(0,0,0,0.03)]"
                        style={{ backgroundColor: `color-mix(in srgb, ${accentColor} 12%, #ffffff)`, color: accentColor }}
                      >
                        {LABEL_NAMES[row.category] || row.category}
                      </td>
                      {papers.map(p => (
                        <td key={`group-spacer-${p.id}`} className="border-b border-r border-outline-variant/60" style={{ backgroundColor: 'inherit' }}></td>
                      ))}
                    </tr>
                  );
                }
                
                // Entity Row
                const entityRow = row as { type: 'entity', category: string, name: string, counts: Record<string, number>, numPapersWithMention: number };
                const accentColor = `var(${getEntityAccentVar(entityRow.category)}, var(--primary))`;
                const maxCount = matrixData.globalMaxCount || 1;
                
                return (
                  <tr key={`entity-${entityRow.category}-${entityRow.name}`} className="border-b border-outline-variant/40 hover:bg-slate-50 transition-colors group">
                    <td className="p-3 font-medium border-r border-outline-variant sticky left-0 bg-white group-hover:bg-slate-50 transition-colors z-20 shadow-[2px_0_4px_rgba(0,0,0,0.03)]">
                      <span className="text-on-surface truncate block pr-2" title={entityRow.name}>{entityRow.name}</span>
                    </td>
                    {papers.map(p => {
                      const count = entityRow.counts[p.id];
                      if (!count) {
                        return <td key={`cell-${entityRow.name}-${p.id}`} className="p-1.5 px-2 border-r border-outline-variant/40 text-center text-on-surface-variant/40 text-xs">—</td>;
                      }
                      
                      // Universal dynamic opacity percentage (18% to 90%) calculated from entity mention number
                      const norm = maxCount > 1 ? count / maxCount : 0.5;
                      const opacityPercent = Math.min(90, Math.max(18, Math.round(18 + 72 * Math.pow(norm, 0.5))));
                      // High-contrast, fairly visible numbers on both light and saturated backgrounds
                      const textColor = opacityPercent > 55 ? '#ffffff' : `color-mix(in srgb, ${accentColor} 85%, #0f172a)`;
                      
                      return (
                        <td key={`cell-${entityRow.name}-${p.id}`} className="p-1.5 px-2 border-r border-outline-variant/40 text-center">
                          <div 
                            className="w-full h-7 rounded-md flex items-center justify-center font-bold text-xs transition-all shadow-[0_1px_2px_rgba(0,0,0,0.03)]"
                            style={{ 
                              backgroundColor: `color-mix(in srgb, ${accentColor} ${opacityPercent}%, transparent)`,
                              color: textColor 
                            }}
                            title={`${count} mentions`}
                          >
                            {count}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
            <tfoot className="bg-slate-50 border-t-2 border-outline-variant sticky bottom-0 z-40 shadow-[0_-2px_4px_rgba(0,0,0,0.04)]">
              <tr className="font-bold text-on-surface text-xs bg-slate-50">
                <td className="p-3 border-r border-outline-variant sticky left-0 bg-slate-50 z-50 shadow-[2px_0_4px_rgba(0,0,0,0.03)]">
                  Total mentions
                </td>
                {papers.map(p => (
                  <td key={`total-${p.id}`} className="p-3 border-r border-outline-variant text-center bg-slate-50 font-bold text-slate-800">
                    {matrixData.paperTotals[p.id].toLocaleString()}
                  </td>
                ))}
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </div>
  );
};
