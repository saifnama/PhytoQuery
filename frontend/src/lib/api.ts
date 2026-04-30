import axios from 'axios';
import type {
  NERResponse,
  QueryResponse,
  UploadResponse,
  PaperData,
  SearchResult,
  SearchFilters,
  Entity,
  IndexedFileInfo,
} from '../types';

type PaperApiResponse = (PaperData & { entities?: Entity[] }) | { error: string; sections: unknown[] };

export interface PaperPdfResponse {
  blob: Blob;
  filename: string;
}

const API_BASE = ''; // Uses Vite proxy in dev, same origin in production

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000, 
  withCredentials: true,
});

// NER API
export const nerApi = {
  /**
   * Search papers by query
   * @param source - "europepmc", "openalex", or "" (both/merged)
   */
  async search(query: string, filters: SearchFilters, page: number = 1, source: string = ""): Promise<{ results: SearchResult[]; pagination: { total: number; page: number; hasMore: boolean; pageSize: number } ; error?: string }> {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('open_access', String(filters.open_access));
    formData.append('has_full_text', String(filters.has_full_text));
    formData.append('article_type', filters.article_type);
    formData.append('sort', filters.sort);
    formData.append('page', String(page));
    if (source) {
      formData.append('source', source);
    }

    const response = await api.post('/search/json', formData);
    return response.data;
  },

  /**
   * Fetch paper data with optional NER extraction
   */
  async analyzePaper(doi: string, runNer: boolean = false, source: string = ""): Promise<PaperApiResponse> {
    const formData = new FormData();
    formData.append('doi', doi);
    formData.append('run_ner', String(runNer));
    if (source) {
      formData.append('source', source);
    }

    const response = await api.post<PaperApiResponse>('/paper/json', formData, {
      timeout: runNer ? 600000 : 120000,
    });
    return response.data;
  },

  /**
   * Extract entities from paper (JSON endpoint)
   */
  async extractEntities(doi: string): Promise<NERResponse> {
    const response = await api.post('/ner/doi/json', { doi });
    return response.data;
  },

  /**
   * Switch to a different section in paper viewer
   */
  async switchSection(doi: string, sectionIdx: number): Promise<{ content: string; highlighted: string }> {
    const formData = new FormData();
    formData.append('doi', doi);
    formData.append('section_idx', String(sectionIdx));

    const response = await api.post('/paper/section/json', formData);
    return response.data;
  },

};

const extractFilenameFromDisposition = (contentDisposition?: string): string => {
  if (!contentDisposition) return 'paper.pdf';
  const match = contentDisposition.match(/filename="?([^";]+)"?/i);
  return match?.[1]?.trim() || 'paper.pdf';
};

export const paperApi = {
  getPdfUrl(identifier: string): string {
    return `/paper/pdf?identifier=${encodeURIComponent(identifier)}`;
  },

  async fetchPdf(identifier: string): Promise<PaperPdfResponse> {
    const response = await api.get('/paper/pdf', {
      params: { identifier },
      responseType: 'blob',
      timeout: 120000,
    });

    return {
      blob: response.data,
      filename: extractFilenameFromDisposition(response.headers['content-disposition']),
    };
  },

   /**
    * Fetch PDF and upload it directly to RAG for indexing.
    * Used for silent "Upload to RAG" from paper page without navigation.
    */
   async fetchAndUploadToRag(identifier: string): Promise<{ status: string; message: string; filename?: string }> {
     const { blob, filename } = await paperApi.fetchPdf(identifier);
     const file = new File([blob], filename || `${identifier}.pdf`, { type: 'application/pdf' });
     
     const result = await ragApi.uploadFiles([file], 'pymupdf');
     // result.files is string[] - get first file
     const fileList = result.files || [];
     return {
       status: result.status,
       message: result.message,
       filename: fileList[0],
     };
   },

    /**
     * Upload an already-fetched PDF File to RAG.
     * Used when we have a direct PDF URL (OpenAlex/Semantic Scholar) and want to upload it.
     */
    async uploadPdfToRag(file: File): Promise<{ status: string; message: string; filename?: string }> {
      const result = await ragApi.uploadFiles([file], 'pymupdf');
      const fileList = result.files || [];
      return {
        status: result.status,
        message: result.message,
        filename: fileList[0],
      };
    },

    /**
     * Fetch PDF from an external URL via backend proxy (bypasses CORS).
     * Used for OpenAlex/Semantic Scholar direct PDF URLs.
     */
    async fetchPdfFromUrl(pdfUrl: string): Promise<PaperPdfResponse> {
      const response = await api.get('/paper/pdf-proxy', {
        params: { url: pdfUrl },
        responseType: 'blob',
        timeout: 120000,
      });
      return {
        blob: response.data,
        filename: extractFilenameFromDisposition(response.headers['content-disposition']),
      };
    },
  };

// RAG API
export const ragApi = {
  /**
   * Upload PDF files for indexing
   * @param files - List of PDF files to upload
   * @param parserType - "pymupdf" for fast extraction, "docling" for detailed (default: "docling")
   */
  async uploadFiles(files: File[], parserType: 'pymupdf' | 'docling' = 'docling'): Promise<UploadResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('parser_type', parserType);

    const response = await api.post<UploadResponse>('/api/chat/upload/json', formData);
    return response.data;
  },

  /**
   * List all indexed files in the RAG vector store
   * List all documents currently indexed in the RAG vector store
   */
  async listFiles(): Promise<IndexedFileInfo[]> {
    const response = await api.get('/api/chat/files/json');
    return response.data;
  },

  /**
   * Query the RAG system with optional source filtering
   * @param query - The user's question
   * @param selectedFiles - If provided, only search chunks from these filenames
   */
  async query(query: string, selectedFiles?: string[], chatHistory?: { role: string; content: string }[]): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/api/chat/query/json', {
      query,
      selected_files: selectedFiles,
      chat_history: chatHistory,
    });
    return response.data;
  },

  /**
   * Delete a source completely: removes chunks from ChromaDB and PDF from disk
   */
  async deleteFile(filename: string): Promise<{ status: string; message: string }> {
    const response = await api.delete(`/api/chat/files/${encodeURIComponent(filename)}`);
    return response.data;
  },

  /**
   * Permanently delete all chat history, indexed papers, and vector embeddings
   */
  async resetChat(): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/chat/reset');
    return response.data;
  },

  /**
   * Clean up all user data when they close the browser
   * Deletes ChromaDB, uploads, and all files for the current user
   */
  async cleanupUserData(): Promise<{ status: string; message: string }> {
    const response = await api.post('/api/chat/cleanup');
    return response.data;
  },
};

export const buildChatFileContentUrl = (filename: string): string => {
  return `/api/chat/files/${encodeURIComponent(filename)}/content`;
};

// Search Types API
export const searchTypesApi = {
  /**
   * Fetch available article types for a search source.
   * @param source - "europepmc" or "openalex"
   */
  async getTypes(source: string): Promise<{ types: { key: string; display_name: string; count: number | null }[] }> {
    const response = await api.get('/search/types', { params: { source } });
    return response.data;
  },
};

// Health API
export const healthApi = {
  async check(): Promise<{ status: string }> {
    const response = await api.get('/health');
    return response.data;
  },
};

// DOI Abstract Fallback API
export const doiApi = {
  /**
 * Fetch abstract for a DOI when Europe PMC doesn't have it.
 * Multi-source fallback: OpenAlex → Semantic Scholar
   */
  async getAbstract(doi: string): Promise<{
    doi: string;
    title: string;
    abstract: string;
    authors: string[];
    year: number | null;
    source: string;
    url: string;
  } | null> {
    try {
      console.log('[DOI API] Fetching:', `/doi/abstract?doi=${doi}`);
      const response = await api.get('/doi/abstract', {
        params: { doi },
        timeout: 60000,
      });
      console.log('[DOI API] Response:', response.status, response.data);
      return response.data;
    } catch (e: any) {
      console.error('[DOI API] Error:', e?.response?.status, e?.response?.data || e?.message);
      return null;
    }
  },
};

export default api;
