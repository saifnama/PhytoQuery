import React, { useState, useEffect, useRef, useCallback } from 'react';
import { sanitizeHtml } from '../../utils/sanitize';
import { PencilSimple } from '@phosphor-icons/react';
import type { Entity, TocItem } from '../../types';


interface PaperViewerProps {
  doi: string;
  mode: 'full_text' | 'abstract';
  title: string;
  html: string;
  toc: TocItem[];
  entities?: Entity[];
  isExtracted?: boolean;
  isExtracting?: boolean;
  extractionError?: string | null;
  summary?: Record<string, { text: string; count: number; avg_score: number }[]>;
  fallbackSource?: { source: string; url: string };
  isFetchingFallback?: boolean;
  paperAuthors?: string[];
  paperJournal?: string;
  paperDate?: string;
  onExtract?: () => void;
}

interface GroupedEntities {
  [label: string]: { text: string; count: number }[];
}

const PaperViewer: React.FC<PaperViewerProps> = ({
   doi,
   mode,
   title,
   html,
   toc,
   entities = [],
   isExtracted = false,
   isExtracting = false,
   extractionError = null,
   fallbackSource,
   isFetchingFallback = false,
   paperAuthors = [],
   paperJournal,
   paperDate,
   onExtract,
 }) => {
  const [activeHeading, setActiveHeading] = useState<string | null>(null);
  const [currentSectionIdx, setCurrentSectionIdx] = useState(0);
  const htmlContainerRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Group entities by label and calculate true frequency from frontend HTML
  const groupedEntities = useCallback(() => {
    const grouped: GroupedEntities = {};
    if (!entities || entities.length === 0) return grouped;

    // Create a plain text version of the HTML to count occurrences accurately without HTML tag interference
    let fullText = '';
    if (html) {
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = html;
      fullText = tempDiv.textContent || tempDiv.innerText || '';
    }

    // Group entities case-insensitively
    const uniqueEntities: Record<string, { label: string, originalTexts: Record<string, number> }> = {};

    entities.forEach((entity) => {
      // Strip any stray HTML tags like <em> that might have been extracted by the AI
      const cleanText = entity.text.replace(/<[^>]+>/g, '').trim();
      const lowerText = cleanText.toLowerCase();
      if (!uniqueEntities[lowerText]) {
        uniqueEntities[lowerText] = { label: entity.label, originalTexts: {} };
      }
      uniqueEntities[lowerText].originalTexts[cleanText] = (uniqueEntities[lowerText].originalTexts[cleanText] || 0) + 1;
    });

    Object.keys(uniqueEntities).forEach(lowerText => {
      const data = uniqueEntities[lowerText];
      
      // Determine most frequent original casing
      let bestText = lowerText;
      let maxCount = 0;
      for (const [text, count] of Object.entries(data.originalTexts)) {
        if (count > maxCount) {
          maxCount = count;
          bestText = text;
        }
      }

      // Count frequency in full text using safe boundary regex
      let trueCount = 0;
      if (fullText) {
        try {
          const escapedText = lowerText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          const regex = new RegExp(`(?:^|\\W)(${escapedText})(?:$|\\W)`, 'gi');
          while (regex.exec(fullText) !== null) {
            trueCount++;
            // Rewind 1 character to handle overlapping boundaries (e.g. "virus, ebola, virus")
            if (regex.lastIndex > 0) {
              regex.lastIndex -= 1;
            }
          }
        } catch (e) {
          // Ultimate fallback to basic string count
          trueCount = fullText.toLowerCase().split(lowerText).length - 1;
        }
      }

      // Fallback to at least 1 if it was extracted by AI
      if (trueCount === 0) {
        trueCount = 1;
      }

      if (!grouped[data.label]) {
        grouped[data.label] = [];
      }
      grouped[data.label].push({ text: bestText, count: trueCount });
    });

    // Sort by count descending
    Object.keys(grouped).forEach(label => {
      grouped[label].sort((a, b) => b.count - a.count);
    });

    return grouped;
  }, [entities, html]);

  // Setup scroll spy for sub-headings (based on HTML blob sections)
  useEffect(() => {
    if (!htmlContainerRef.current) return;
    const container = htmlContainerRef.current;
    const sectionNodes = container.querySelectorAll('section');
    if (sectionNodes.length === 0) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.intersectionRatio > 0) {
            const id = entry.target.id;
            const match = id.match(/^section-(\d+)$/);
            if (match) {
              setCurrentSectionIdx(parseInt(match[1], 10));
            } else {
              const idx = Array.from(sectionNodes).indexOf(entry.target as HTMLElement);
              if (idx >= 0) setCurrentSectionIdx(idx);
            }
            setActiveHeading(id);
          }
        });
      },
      { rootMargin: '-20% 0px -70% 0px', threshold: 0 }
    );

    sectionNodes.forEach((el: Element) => observerRef.current?.observe(el as HTMLElement));

    return () => {
      observerRef.current?.disconnect();
    };
  }, [html]);

  // Scroll to element by ID
  const scrollToId = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // Keyboard shortcuts (navigate through TOC items)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((toc?.length ?? 0) === 0) return;
      if (e.key === 'ArrowRight') {
        if (currentSectionIdx < (toc?.length ?? 0) - 1) {
          const nextId = toc?.[currentSectionIdx + 1]?.id;
          if (nextId) scrollToId(nextId);
        }
      } else if (e.key === 'ArrowLeft') {
        if (currentSectionIdx > 0) {
          const prevId = toc?.[currentSectionIdx - 1]?.id;
          if (prevId) scrollToId(prevId);
        }
      } else if (e.key === 'ArrowUp') {
        window.scrollBy({ top: -100, behavior: 'smooth' });
      } else if (e.key === 'ArrowDown') {
        window.scrollBy({ top: 100, behavior: 'smooth' });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentSectionIdx, toc?.length]);

  // Citation click handler - scroll to reference
  useEffect(() => {
    const handleCitationClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const cite = target.closest('.citation');
      if (cite) {
        // Support both data-rid and data-ref as the citation identifier
        const refId = cite.getAttribute('data-rid') ?? cite.getAttribute('data-ref');
        if (refId) {
          const refEl = document.getElementById(`ref-${refId}`);
          if (refEl) {
            e.preventDefault();
            refEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            // Highlight the reference briefly
            refEl.classList.add('bg-yellow-100');
            setTimeout(() => {
              refEl.classList.remove('bg-yellow-100');
            }, 1800);
          }
        }
      }
    };

    document.addEventListener('click', handleCitationClick);
    return () => document.removeEventListener('click', handleCitationClick);
  }, []);

  // LEGEND_ITEMS for entity highlighting
  const LEGEND_ITEMS = [
    { label: 'CHEMICAL', className: 'legend-chemical' },
    { label: 'SPECIES', className: 'legend-species' },
    { label: 'PLANT PART', className: 'legend-plant-part' },
    { label: 'EXTRACTION METHOD', className: 'legend-extraction-method' },
    { label: 'LOCATION', className: 'legend-location' },
    { label: 'CHEMICAL ACTIVITY', className: 'legend-chemical-activity' },
    { label: 'ISOLATION METHOD', className: 'legend-isolation-method' },
    { label: 'DISEASE', className: 'legend-disease' },
    { label: 'DRUG', className: 'legend-drug' },
    { label: 'CHEMICAL LIGAND', className: 'legend-chemical-ligand' },
  ];

  return (
    <div className="flex flex-col lg:flex-row min-h-screen bg-white w-full max-w-full overflow-x-hidden">
      {/* Left Sidebar: Table of Contents */}
      <aside className="hidden lg:flex flex-col w-[260px] border-r border-slate-100 p-6 space-y-6 h-screen sticky top-0 overflow-y-auto shrink-0 bg-white custom-scrollbar">
          <div className="mb-4">
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-2 title-font">
            Table of Contents
          </h3>
            <p className="text-[11px] font-semibold text-slate-700 leading-snug line-clamp-3 title-font">
            {title || 'Untitled Paper'}
          </p>
        </div>

        <nav className="flex flex-col space-y-0.5">
          {toc.map((item, idx) => (
            <button
              key={item.id}
              onClick={() => {
                setCurrentSectionIdx(idx);
                setActiveHeading(null);
                scrollToId(item.id);
              }}
              data-toc-id={`toc-${item.id}`}
              className={`toc-lnk w-full text-left text-[11px] font-semibold uppercase tracking-wider transition-all duration-200 py-2.5 px-4 rounded-xl
                ${
                  idx === currentSectionIdx && !activeHeading
                    ? 'toc-item-active bg-blue-50 text-blue-600'
                    : idx === currentSectionIdx
                    ? 'text-blue-500'
                    : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
                }`}
              style={{ paddingLeft: `${(item.level || 1) * 0.5 + 1}rem` }}
            >
              {item.text}
            </button>
          ))}

          {/* Keyboard shortcuts */}
        <div className="hidden md:flex items-center justify-end space-x-2 text-[9px] text-slate-400">
            <span className="flex items-center">
              <span className="mr-1">←</span> Prev
            </span>
            <span className="mx-2">|</span>
            <span className="flex items-center">
              <span className="mr-1">→</span> Next
            </span>
            <span className="mx-2">|</span>
            <span className="flex items-center">
              <span className="mr-1">↑</span> Up
            </span>
            <span className="mx-2">|</span>
            <span className="flex items-center">
              <span className="mr-1">↓</span> Down
            </span>
          </div>
        </nav>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-white border-r border-slate-100 relative">
        {/* Header */}
        <div className="p-10 lg:p-14 border-b border-slate-50 flex justify-between items-center bg-white sticky top-0 z-20">
          <div>
            <div className="flex items-center mb-2">
              <div className="flex items-center space-x-3">
                <span className="text-[9px] font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full uppercase tracking-widest">
                  {mode}
                </span>
                {fallbackSource && (
                  <span
                    className="text-[9px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full uppercase tracking-widest"
                  >
                    {fallbackSource.source}
                  </span>
                )}
                <a
                  href={`https://doi.org/${doi}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[9px] text-slate-400 font-medium tracking-wider uppercase hover:text-blue-600 transition-colors"
                >
                  DOI: {doi}
                </a>
              </div>
              {paperDate && (
                <span className="text-[9px] text-slate-400 font-medium tracking-wider ml-auto">
                  {paperDate}
                </span>
              )}
            </div>
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight title-font" dangerouslySetInnerHTML={{ __html: sanitizeHtml(title || 'Untitled Paper') }} />
            {(paperAuthors.length > 0 || paperJournal) && (
              <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-500">
                {paperAuthors.length > 0 && (
                  <span className="truncate max-w-md" title={paperAuthors.join(', ')}>
                    {paperAuthors.slice(0, 3).join(', ')}{paperAuthors.length > 3 ? ' et al.' : ''}
                  </span>
                )}
                {paperJournal && (
                  <>
                    <span className="text-slate-300">•</span>
                    <span>{paperJournal}</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Section Content - Continuous Scroll (HTML blob) */}
        <div
           ref={htmlContainerRef}
           id="section-content-area"
           className="p-10 lg:px-20 lg:py-14 min-h-screen article-prose"
           dangerouslySetInnerHTML={{ __html: sanitizeHtml(html) }}
         />
        {isFetchingFallback && (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-slate-200 border-t-emerald-600 mx-auto mb-3" />
              <p className="text-xs text-slate-400">Fetching abstract from alternative sources...</p>
            </div>
          </div>
        )}
      </main>

      {/* Right Sidebar: Legend & Mentions */}
      <aside className="w-full lg:w-[380px] p-8 space-y-10 bg-slate-50/20 h-screen sticky top-0 overflow-y-auto custom-scrollbar shrink-0 relative z-30">
        {/* Find Key Terms Button */}
        <button
          onClick={() => {
            if (onExtract && !isExtracting && !isExtracted) onExtract();
          }}
          disabled={isExtracted || isExtracting}
          className={`w-full px-6 py-3 rounded-xl font-bold transition-all flex items-center justify-center space-x-2 text-[11px] uppercase tracking-widest shadow-lg ${
            isExtracted 
              ? 'bg-slate-200 text-slate-500 cursor-not-allowed' 
              : isExtracting
              ? 'bg-blue-100 text-blue-400 cursor-wait animate-pulse'
              : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-100 cursor-pointer'
          }`}
        >
          {isExtracting ? (
            <div className="animate-spin rounded-full h-3 w-3 border-2 border-blue-400 border-t-blue-600 mr-2" />
          ) : (
            <PencilSimple size={16} weight="bold" />
          )}
          <span>
            {isExtracted ? 'Entities Extracted' : isExtracting ? 'Extracting Terms...' : 'Find Key Terms'}
          </span>
        </button>

        {extractionError && (
          <div className="p-4 bg-red-50 border border-red-100 rounded-xl">
            <p className="text-[10px] text-red-600 font-bold uppercase tracking-wider mb-1">Extraction Error</p>
            <p className="text-[11px] text-red-500 leading-relaxed font-medium">{extractionError}</p>
          </div>
        )}

        {/* Legend */}
        <div className="space-y-6">
          <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-6 border-b border-slate-100 pb-4 title-font">
            Legend
          </h3>
          <div className="grid grid-cols-1 gap-4">
            {LEGEND_ITEMS.map((item) => (
              <label key={item.label} className="flex items-center group cursor-pointer">
                <input type="checkbox" checked className="hidden peer" />
                <div
                  className={`w-3.5 h-3.5 rounded-md border border-slate-200 mr-3 transition-all peer-checked:border-transparent ${item.className}`}
                  style={{ backgroundColor: 'currentColor' }}
                />
                <span className={`text-[11px] font-bold transition-all ${item.className} uppercase tracking-wider`}>
                  {item.label}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Mentions */}
        <div className="space-y-6">
          <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-6 border-b border-slate-100 pb-4 title-font">
            Extracted Mentions
          </h3>

          {!isExtracted ? (
            <div className="text-center py-10 bg-slate-100/50 rounded-2xl border border-dashed border-slate-200">
              <p className="text-[10px] text-slate-400 uppercase tracking-widest italic font-medium">
                Pending analysis
              </p>
            </div>
          ) : (
            <div className="space-y-8 pr-2">
              {Object.entries(groupedEntities()).map(([label, ents], idx) => (
                <div key={label} className="animate-fade-in" style={{ animationDelay: `${idx * 0.05}s` }}>
                  <div className="flex items-center space-x-2 mb-4">
                    <div
                      className={`w-1 h-3 rounded-full legend-${label.toLowerCase().replace(/[\s_]/g, '-')}`}
                      style={{ backgroundColor: 'currentColor' }}
                    />
                    <span className={`text-[10px] font-bold tracking-widest legend-${label.toLowerCase().replace(/[\s_]/g, '-')} uppercase`}>
                      {label}
                    </span>
                  </div>

                  <div className="space-y-1.5">
                    {ents
                      .sort((a, b) => b.count - a.count)
                      .map((ent, eIdx) => (
                        <div
                          key={eIdx}
                          className="flex justify-between items-center py-2.5 px-4 bg-white border border-slate-100 rounded-xl transition-all hover:border-blue-100 group/item"
                        >
                          <span className="text-[10px] text-slate-600 font-bold uppercase truncate pr-2">
                            {ent.text}
                          </span>
                          <span className="text-[10px] font-extrabold text-blue-600">
                            {ent.count}
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </aside>

      <style>{`
        .font-display {
          font-family: 'Inter', sans-serif !important;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in {
          animation: fadeIn 0.4s ease-out forwards;
        }
        .toc-lnk:focus-visible {
          outline: 2px solid #2563eb;
          outline-offset: 2px;
        }
        .scroll-mt-20 {
          scroll-margin-top: 5rem;
        }
      `}</style>
    </div>
  );
};

export default PaperViewer;
