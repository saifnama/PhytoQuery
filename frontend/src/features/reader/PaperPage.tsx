import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft } from '@phosphor-icons/react';
import PaperViewer from './PaperViewer';
import { doiApi, nerApi } from '../../lib/api';
import type { PaperData, Entity } from '../../types';

const PaperPage: React.FC = () => {
  const { doi } = useParams<{ doi: string }>();
  const navigate = useNavigate();
  const [paperData, setPaperData] = useState<PaperData | null>(null);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [isExtracted, setIsExtracted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionError, setExtractionError] = useState<string | null>(null);
  const [fallbackSource, setFallbackSource] = useState<{ source: string; url: string } | null>(null);
  const [fallbackLoading, setFallbackLoading] = useState(false);

  const isExplicitDoi = (value: string) => {
    const trimmed = value.trim();
    return /^10\.\d{4,}/i.test(trimmed) || /^https?:\/\/(dx\.)?doi\.org\//i.test(trimmed) || /^doi:/i.test(trimmed);
  };

  const getLookupIdentifier = () => {
    const raw = doi?.trim();
    if (!raw) return undefined;

    if (raw.toUpperCase().startsWith('PMC')) {
      return {
        type: 'pmcid' as const,
        value: raw.toUpperCase(),
        href: `https://pmc.ncbi.nlm.nih.gov/articles/${raw.toUpperCase()}/`,
      };
    }

    if (/^\d+$/.test(raw)) {
      return {
        type: 'pmid' as const,
        value: raw,
        href: `https://pubmed.ncbi.nlm.nih.gov/${raw}/`,
      };
    }

    if (isExplicitDoi(raw)) {
      return {
        type: 'doi' as const,
        value: raw,
        href: `https://doi.org/${raw}`,
      };
    }

    return undefined;
  };

  const getPaperIdentifier = () => {
    const lookupIdentifier = getLookupIdentifier();
    if (lookupIdentifier) {
      return lookupIdentifier;
    }

    const resolvedDoi = paperData?.doi?.trim();
    if (resolvedDoi && isExplicitDoi(resolvedDoi)) {
      return {
        type: 'doi' as const,
        value: resolvedDoi,
        href: `https://doi.org/${resolvedDoi}`,
      };
    }

    const resolvedPmcid = paperData?.pmcid?.trim();
    if (resolvedPmcid) {
      return {
        type: 'pmcid' as const,
        value: resolvedPmcid,
        href: `https://pmc.ncbi.nlm.nih.gov/articles/${resolvedPmcid}/`,
      };
    }

    if (doi?.trim().startsWith('PMC')) {
      return {
        type: 'pmcid' as const,
        value: doi.trim(),
        href: `https://pmc.ncbi.nlm.nih.gov/articles/${doi.trim()}/`,
      };
    }

    if (doi && /^\d+$/.test(doi.trim())) {
      return {
        type: 'pmid' as const,
        value: doi.trim(),
        href: `https://pubmed.ncbi.nlm.nih.gov/${doi.trim()}/`,
      };
    }

    return undefined;
  };

  useEffect(() => {
    const fetchPaper = async () => {
      if (!doi) return;

      setIsLoading(true);
      setError(null);

      try {
        const data = await nerApi.analyzePaper(doi, false);
        
        if (!data || 'error' in data) {
          setError(data.error || 'Paper not found');
          return;
        }
        
        setPaperData(data);
        if (data.entities) {
          setEntities(data.entities);
        }
        // Track if this came from a fallback source
        if (data.fallback_source) {
          setFallbackSource({
            source: data.fallback_source,
            url: data.fallback_url || '',
          });
        }
      } catch (err) {
        console.error('[PaperPage] Failed to fetch paper:', err);
        if (!isExplicitDoi(doi)) {
          setError('Failed to load paper from the primary source.');
          return;
        }

        setFallbackLoading(true);
        try {
          const data = await doiApi.getAbstract(doi);
          setFallbackLoading(false);
          if (data?.abstract) {
            setPaperData({
              doi: data.doi || doi,
              mode: 'abstract' as const,
              title: data.title || '',
              html: `<section id="section-0"><h2>Abstract</h2><p>${data.abstract}</p></section>`,
              sections: [{ title: 'Abstract', content: data.abstract, headings: [] }],
              references: {},
              pmcid: '',
              authors: data.authors,
              year: data.year,
            } as PaperData);
            setFallbackSource({ source: data.source, url: data.url });
            return;
          }
          setError(`No abstract found. <a href="https://doi.org/${doi}" target="_blank" class="text-blue-600 underline">View on publisher</a>`);
        } catch {
          setFallbackLoading(false);
          setError('Failed to load paper from any source.');
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchPaper();
  }, [doi]);

  const handleExtract = async () => {
    if (!doi) return;

    setIsExtracting(true);
    setExtractionError(null);

    try {
      const data = await nerApi.analyzePaper(doi, true);

      if (!data || 'error' in data) {
        throw new Error(data.error || 'Extraction failed');
      }

      if (data.entities) {
        setEntities(data.entities);
        setPaperData(data);
        setIsExtracted(true);
      }
    } catch (err: any) {
      console.error('NER extraction failed:', err);
      setExtractionError(err.message || 'Extraction timed out or failed. Please try again.');
    } finally {
      setIsExtracting(false);
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'e' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const tag = (e.target as HTMLElement).tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if (!isExtracted && !isExtracting) {
          e.preventDefault();
          handleExtract();
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [doi, isExtracted, isExtracting]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-slate-200 border-t-blue-600 mx-auto mb-4" />
          <p className="text-sm text-slate-500">Loading paper...</p>
        </div>
      </div>
    );
  }

  if (fallbackLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-slate-200 border-t-emerald-600 mx-auto mb-4" />
          <p className="text-sm text-slate-500">Fetching DOI fallback sources...</p>
        </div>
      </div>
    );
  }

  if (error || !paperData) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center p-8 bg-red-50 rounded-xl">
          <p className="text-sm text-red-600 mb-4" dangerouslySetInnerHTML={{ __html: error || 'Paper not found' }} />
          <button
            onClick={() => navigate('/')}
            className="text-blue-600 hover:underline flex items-center gap-2 mx-auto"
          >
            <ArrowLeft size={16} /> Back to search
          </button>
        </div>
      </div>
    );
  }

  const htmlBlob: string = paperData.html ??
    (paperData.sections?.length
      ? paperData.sections.map((s, i) => `<section id="section-${i}"><h2>${s.title}</h2>${s.content}</section>`).join('')
      : '');
  const tocList = paperData.toc ?? (paperData.sections?.map((s, i) => ({ id: `section-${i}`, text: s.title, level: 1 })) ?? []);

  // Add spaces around inline tags when touching letters
  // e.g. "of<i>Lantana</i>L." → "of <i>Lantana</i> L."
  // Works for any tag: <i>, <b>, <em>, <sub>, <sup>, etc.
  const formattedTitle = paperData.title
    ? (paperData.title.includes('<')
        ? paperData.title
        : paperData.title
            .replace(/([a-zA-Z])(<[a-zA-Z])/g, '$1 $2')
            .replace(/(<\/[a-zA-Z]+>)([a-zA-Z])/g, '$1 $2'))
    : '';

  return (
    <PaperViewer
      paperIdentifier={getPaperIdentifier()}
      mode={paperData.mode}
      title={formattedTitle || 'Untitled Paper'}
      html={htmlBlob}
      toc={tocList}
      entities={entities}
      isExtracted={isExtracted}
      isExtracting={isExtracting}
      extractionError={extractionError}
      fallbackSource={fallbackSource ?? undefined}
      isFetchingFallback={false}
      paperAuthors={paperData.authors || []}
      paperJournal={paperData.journal}
      paperDate={paperData.date}
      onExtract={handleExtract}
    />
  );
};

export default PaperPage;
