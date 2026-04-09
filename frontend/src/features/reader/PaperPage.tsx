import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft } from '@phosphor-icons/react';
import PaperViewer from './PaperViewer';
import { nerApi } from '../../lib/api';
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

  useEffect(() => {
    const fetchPaper = async () => {
      if (!doi) return;

      setIsLoading(true);
      setError(null);

      try {
        const data = await nerApi.analyzePaper(doi, false);
        
        if (!data || (data as any).error) {
          setError((data as any).error || 'Paper not found');
          return;
        }
        
        setPaperData(data as PaperData);
        if (data.entities) {
          setEntities(data.entities);
        }
        // Track if this came from a fallback source
        if ((data as any).fallback_source) {
          setFallbackSource({
            source: (data as any).fallback_source,
            url: (data as any).fallback_url || '',
          });
        }
      } catch (err) {
        console.error('[PaperPage] Failed to fetch paper:', err);
        // If request timed out, show fallback loading
        setFallbackLoading(true);
        try {
          const resp = await fetch(`/doi/abstract?doi=${encodeURIComponent(doi)}`);
          setFallbackLoading(false);
          if (resp.ok) {
            const data = await resp.json();
            if (data?.abstract) {
              setPaperData({
                mode: 'abstract' as const,
                title: data.title || '',
                html: `<section id="section-0"><h2>Abstract</h2><p>${data.abstract}</p></section>`,
                sections: [{ title: 'Abstract', content: data.abstract }],
                references: {},
                pmcid: '',
              } as PaperData);
              setFallbackSource({ source: data.source, url: data.url });
              return;
            }
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
      
      if (!data || (data as any).error) {
        throw new Error((data as any).error || 'Extraction failed');
      }

      if (data.entities) {
        setEntities(data.entities);
        setPaperData(data as PaperData);
        setIsExtracted(true);
      }
    } catch (err: any) {
      console.error('NER extraction failed:', err);
      setExtractionError(err.message || 'Extraction timed out or failed. Please try again.');
    } finally {
      setIsExtracting(false);
    }
  };

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
          <p className="text-sm text-slate-500">Fetching from OpenAlex...</p>
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
    ? paperData.title
        .replace(/([a-zA-Z])(<[a-zA-Z])/g, '$1 $2')  // letter before opening tag
        .replace(/(<\/[a-zA-Z]+>)([a-zA-Z])/g, '$1 $2')  // closing tag before letter
    : '';

  return (
    <PaperViewer
      doi={(paperData as any).doi || doi || ''}
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
      paperDate={(paperData as any).date}
      onExtract={handleExtract}
    />
  );
};

export default PaperPage;
