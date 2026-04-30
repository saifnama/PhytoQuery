// API Types based on FastAPI schemas

export interface TocItem {
  id: string;
  text: string;
  level?: number;
}

export interface Entity {
  text: string;
  label: string;
  score: number;
  canonical?: string;  // Normalized form for display (e.g., "seed" for "seeds")
  aliases?: string[];  // All variations for counting
  name_type?: 'scientific' | 'common' | null;
  linked_to?: string | null;
  scientific_name_verified?: string;
  accepted_scientific_name?: string;
  common_name?: string;
  source_db?: string;
  source_url?: string;
  taxon_id?: string;
  match_status?: string;
  review_required?: string;
  // Chemical entity fields
  preferred_name?: string;
  inchikey?: string;
  smiles?: string;
  molecular_formula?: string;
}

export interface NERRequest {
  doi: string;
}

export interface NERResponse {
  doi: string;
  mode: 'full_text' | 'abstract';
  text: string;
  entities: Entity[];
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface QueryRequest {
  query: string;
  selected_files?: string[];
  chat_history?: ChatMessage[];
}

export interface QueryResponse {
  answer: string;
  sources: {
    source: string;
    section: string;
    parser_type: string;
    score: number;
    chunk_text: string;
  }[];
}

export interface UploadResponse {
  status: string;
  message: string;
  files: string[];
  summaries?: Record<string, string>;
  job_id?: string;
}

export interface UploadJobStatus {
  job_id: string;
  status: string; // "processing" | "completed" | "failed"
  message: string;
  files: string[];
  parser_type: string;
  summaries?: Record<string, string>;
  error?: string;
  created_at: string;
  completed_at?: string;
}

export interface IndexedFileInfo {
  name: string;
  file_type: string;
  chunk_count: number;
  indexed_at: string;
  parser_type: string;
  authors?: string;
  doi?: string;
  journal?: string;
  summary?: string;
}

export interface Heading {
  text: string;
  id: string;
}

export interface Reference {
  id: string;
  authors: string[];
  title: string;
  journal: string;
  year: string;
  volume: string;
  issue: string;
  fpage: string;
  lpage: string;
  doi: string;
  pmid: string;
}

export interface Section {
  title: string;
  content: string;
  headings: Heading[];
}

export interface PaperData {
  doi?: string;
  // HTML blob of the paper content (new shape)
  html?: string;
  // Optional table-of-contents items derived from API
  toc?: TocItem[];
  // Fallbacks for compatibility with older API responses
  sections?: Section[];
  mode: 'full_text' | 'abstract';
  title: string;
  references: Record<string, Reference>;
  pmcid: string;
  // Summary of entities by label for sidebar counts
  summary?: Record<string, { text: string; count: number; avg_score: number }[]>;
  entities?: Entity[];
  is_extracted?: boolean;
  fallback_source?: string;
  fallback_url?: string;
  // Metadata from fallback sources (OpenAlex, Semantic Scholar, PMC)
  authors?: string[];
  year?: number | null;
  journal?: string;
  date?: string;
  // PDF links (from OpenAlex or Semantic Scholar)
  pdfUrl?: string;
  openAccessPdf?: { url: string } | null;
}

export interface SearchResult {
  id: string;
  pmcid?: string;
  doi?: string;
  pmid?: string;
  title: string;
  authors: string;
  journal: string;
  year: string;
  citationCount?: number;
  isOpenAccess?: boolean;
  hasTextMinedTerms?: boolean;
  hasFullText?: boolean;
  hasPdfUrl?: boolean;
  pdfUrl?: string;
  abstract?: string;
  source?: string;
}

export interface SearchFilters {
  open_access: boolean;
  has_full_text: boolean;
  article_type: string;
  sort: string;
  source: string;  // "europepmc" or "openalex"
}

export interface RAGResult {
  query: string;
  answer: string;
  sources: Record<string, unknown>[];
}
