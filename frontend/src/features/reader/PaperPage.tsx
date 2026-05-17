import React, { useState, useEffect } from 'react';
import { useNavigate, getRouteApi } from '@tanstack/react-router';
import { ArrowLeft } from '@phosphor-icons/react';
import PaperViewer from './PaperViewer';
import { doiApi, nerApi, paperApi, dbApi } from '../../lib/api';
import type { PaperData, Entity } from '../../types';

const route = getRouteApi('/paper/$doi');

const PaperPage: React.FC = () => {
  const { doi } = route.useParams();
  const { src } = route.useSearch();
  const navigate = useNavigate();
  const searchSource = src ?? '';
  const [paperData, setPaperData] = useState<PaperData | null>(null);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [isExtracted, setIsExtracted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExtracting, setIsExtracting] = useState(false);
  const [extractionError, setExtractionError] = useState<string | null>(null);
  const [fallbackSource, setFallbackSource] = useState<{ source: string; url: string } | null>(null);
  const [fallbackLoading, setFallbackLoading] = useState(false);
  const [pdfActionError, setPdfActionError] = useState<string | null>(null);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [isUploadingToRag, setIsUploadingToRag] = useState(false);
  const [isAddingToMyPapers, setIsAddingToMyPapers] = useState(false);
  const [myPapersActionError, setMyPapersActionError] = useState<string | null>(null);

  const isExplicitDoi = (value: string) => {
    const trimmed = value.trim();
    return /^10\.\d{4,}/i.test(trimmed) || /^https?:\/\/(dx\.)?doi\.org\//i.test(trimmed) || /^doi:/i.test(trimmed);
  };

  const normalizeDoi = (value: string): string => {
    const trimmed = value.trim();
    // Extract DOI from full URLs: https://doi.org/10.xxxx/xxx or https://dx.doi.org/10.xxxx/xxx
    const urlMatch = trimmed.match(/^https?:\/\/(?:dx\.)?doi\.org\/(.+)$/i);
    if (urlMatch) return urlMatch[1];
    // Strip doi: prefix
    if (/^doi:/i.test(trimmed)) return trimmed.substring(4).trim();
    return trimmed;
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
      const normalized = normalizeDoi(raw);
      return {
        type: 'doi' as const,
        value: normalized,
        href: `https://doi.org/${normalized}`,
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
        const dataPromise = nerApi.analyzePaper(doi, false, searchSource);
        const dbEntitiesPromise = dbApi.getPaperEntities(doi).catch(() => ({ entities: [] }));

        const [data, dbData] = await Promise.all([dataPromise, dbEntitiesPromise]);
        
        if (!data || 'error' in data) {
          setError(data.error || 'Paper not found');
          return;
        }
        
        setPaperData(data);
        
        // Use DB entities if they exist, otherwise fallback to NER output if any
        if (dbData && dbData.entities && dbData.entities.length > 0) {
          setEntities(dbData.entities);
          setIsExtracted(true);
        } else if (data.entities && data.entities.length > 0) {
          setEntities(data.entities);
          setIsExtracted(true);
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
      const data = await nerApi.analyzePaper(doi, true, searchSource);

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

  const pdfIdentifier = getPaperIdentifier()?.value || paperData?.pmcid || paperData?.doi || doi;
  // Allow PDF if: full_text mode, OR OpenAlex has direct PDF URL
  const canUsePdfActions = (paperData?.mode === 'full_text' || Boolean(paperData?.pdfUrl)) && Boolean(pdfIdentifier);

  const handleDownloadPdf = async () => {
    // OpenAlex has direct PDF URL - open in new tab
    if (paperData?.pdfUrl) {
      window.open(paperData.pdfUrl, '_blank');
      return;
    }
    
    if (!pdfIdentifier || isDownloadingPdf) return;
    setPdfActionError(null);
    setIsDownloadingPdf(true);

    try {
      const { blob, filename } = await paperApi.fetchPdf(pdfIdentifier);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      console.error('PDF download failed:', err);
      setPdfActionError(err?.response?.data?.detail || 'PDF download is not available for this paper.');
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const handleSendPdfToRag = async () => {
    if (!pdfIdentifier || isUploadingToRag) return;
    setPdfActionError(null);
    setIsUploadingToRag(true);

    try {
      const result = await paperApi.fetchAndUploadToRag(pdfIdentifier);
      if (result.status === 'success') {
        setPdfActionError(null); // Success - no error message
      } else {
        setPdfActionError(result.message || 'Failed to upload to RAG');
      }
    } catch (err: any) {
      console.error('PDF upload to RAG failed:', err);
      setPdfActionError(err?.message || 'Failed to upload PDF to RAG');
    } finally {
      setIsUploadingToRag(false);
    }
  };

  const handleAddToMyPapers = async () => {
    if (!pdfIdentifier || isAddingToMyPapers) return;
    setMyPapersActionError(null);
    setIsAddingToMyPapers(true);

    try {
      const { blob, filename } = paperData?.pdfUrl
        ? await paperApi.fetchPdfFromUrl(paperData.pdfUrl)
        : await paperApi.fetchPdf(pdfIdentifier);
      const file = new File([blob], filename || `${pdfIdentifier}.pdf`, {
        type: 'application/pdf',
      });
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/ner/upload/json', {
        method: 'POST',
        credentials: 'same-origin',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to process paper for My Papers');
      }

      const data = await res.json();

      // Save to localStorage queue so MyPapersPage can pick it up
      const queueKey = 'phytoquery_mypapers_queue';
      const existing = JSON.parse(localStorage.getItem(queueKey) || '[]');
      const paperEntry = {
        id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        name: data.metadata?.title || filename || pdfIdentifier,
        doi: data.metadata?.doi || paperData?.doi,
        pdfUrl: data.pdf_url || null,
        entities: data.entities || {},
        entity_counts: data.entity_counts || {},
        entity_count: data.entity_count || 0,
      };
      localStorage.setItem(queueKey, JSON.stringify([paperEntry, ...existing]));
    } catch (err: any) {
      console.error('Add to My Papers failed:', err);
      setMyPapersActionError(err?.message || 'Failed to add paper to My Papers');
    } finally {
      setIsAddingToMyPapers(false);
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
            onClick={() => navigate({ to: '/' })}
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

  // Format title with preserved formatting (italic/bold) for display
  const displayTitle = paperData.title || 'Untitled Paper';

  return (
    <PaperViewer
      paperIdentifier={getPaperIdentifier()}
      mode={paperData.mode}
      title={displayTitle}
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
      canUsePdfActions={canUsePdfActions}
      isDownloadingPdf={isDownloadingPdf}
      isUploadingToRag={isUploadingToRag}
      isAddingToMyPapers={isAddingToMyPapers}
      pdfActionError={pdfActionError}
      myPapersActionError={myPapersActionError}
      onDownloadPdf={handleDownloadPdf}
      onSendPdfToRag={handleSendPdfToRag}
      onAddToMyPapers={handleAddToMyPapers}
      onExtract={handleExtract}
    />
  );
};

export default PaperPage;
