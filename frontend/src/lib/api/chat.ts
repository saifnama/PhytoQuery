import { api } from './client';
import type {
  QueryResponse,
  UploadResponse,
  UploadJobStatus,
  IndexedFileInfo,
} from '../../types';

// RAG API
export const ragApi = {
  /**
   * Upload PDF files for indexing
   * @param files - List of PDF files to upload
   * @param parserType - "pymupdf" for fast extraction, "docling" for detailed (default: "pymupdf")
   */
  async uploadFiles(files: File[], parserType: 'pymupdf' | 'docling' = 'pymupdf'): Promise<UploadResponse> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('parser_type', parserType);

    const response = await api.post<UploadResponse>('/api/chat/upload/json', formData);
    return response.data;
  },

  /**
   * Upload many PDFs in fixed-size batches.
   *
   * Why: a single multipart POST of ~1000 PDFs (5 GB+) trips
   * reverse-proxy body-size limits (nginx, Cloudflare default 100 MB)
   * long before FastAPI sees it, and consumes peak memory both in the
   * browser and on the server. Slicing into batches of ~20 keeps each
   * request well under typical proxy caps and lets the user see
   * incremental progress as each batch is queued.
   *
   * Each batch becomes its own background indexing job on the server
   * (own ``job_id``); ``onBatch`` fires once per batch with the upload
   * response so the caller can chain its own polling per job.
   *
   * Returns the LAST batch's UploadResponse (so the existing single-
   * batch caller path keeps working unchanged when ``files.length``
   * is below the chunk threshold). For multi-batch uploads, prefer
   * the ``onBatch`` callback to track every job_id.
   */
  async uploadFilesChunked(
    files: File[],
    parserType: 'pymupdf' | 'docling' = 'pymupdf',
    chunkSize = 20,
    onBatch?: (
      batchIndex: number,
      totalBatches: number,
      batchResult: UploadResponse,
    ) => void,
  ): Promise<UploadResponse> {
    if (files.length === 0) {
      throw new Error('uploadFilesChunked: no files');
    }
    const batches: File[][] = [];
    for (let i = 0; i < files.length; i += chunkSize) {
      batches.push(files.slice(i, i + chunkSize));
    }
    let last: UploadResponse | undefined;
    for (let i = 0; i < batches.length; i += 1) {
      const result = await ragApi.uploadFiles(batches[i], parserType);
      last = result;
      if (onBatch) {
        onBatch(i, batches.length, result);
      }
    }
    if (!last) {
      throw new Error('uploadFilesChunked: no batches succeeded');
    }
    return last;
  },

  /**
   * Poll the status of an async upload job.
   */
  async getUploadStatus(jobId: string): Promise<UploadJobStatus> {
    const response = await api.get<UploadJobStatus>(`/api/chat/upload/status/${encodeURIComponent(jobId)}`);
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

