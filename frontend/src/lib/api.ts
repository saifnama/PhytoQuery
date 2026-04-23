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

const API_BASE = ''; // Uses Vite proxy in dev, same origin in production

const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000, 
});

// Generate or retrieve a unique user ID for this browser
export const getUserId = (): string => {
  const STORAGE_KEY = 'pq_user_id';
  let userId = localStorage.getItem(STORAGE_KEY);
  if (!userId) {
    // Generate a simple unique ID
    userId = 'user_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
    localStorage.setItem(STORAGE_KEY, userId);
  }
  return userId;
};

// Add user ID to all requests for multi-user isolation
api.interceptors.request.use((config) => {
  config.headers['X-User-ID'] = getUserId();
  return config;
});

// NER API
export const nerApi = {
  /**
   * Search papers by query
   */
  async search(query: string, filters: SearchFilters, cursorMark: string = '*'): Promise<{ results: SearchResult[]; pagination: { total: number; cursorMark: string; nextCursorMark: string; hasMore: boolean; pageSize: number } ; error?: string }> {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('open_access', String(filters.open_access));
    formData.append('has_full_text', String(filters.has_full_text));
    formData.append('article_type', filters.article_type);
    formData.append('sort', filters.sort);
    formData.append('page_size', String(filters.page_size));
    formData.append('cursor_mark', cursorMark);

    const response = await api.post('/search/json', formData);
    return response.data;
  },

  /**
   * Fetch paper data with optional NER extraction
   */
  async analyzePaper(doi: string, runNer: boolean = false): Promise<PaperApiResponse> {
    const formData = new FormData();
    formData.append('doi', doi);
    formData.append('run_ner', String(runNer));

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
  const params = new URLSearchParams({ user_id: getUserId() });
  return `/api/chat/files/${encodeURIComponent(filename)}/content?${params.toString()}`;
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
