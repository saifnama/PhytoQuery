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

export interface QueryRequest {
  query: string;
  selected_files?: string[];
}

export interface QueryResponse {
  answer: string;
  sources: Record<string, unknown>[];
}

export interface UploadResponse {
  status: string;
  message: string;
  files: string[];
}

export interface IndexedFileInfo {
  name: string;
  file_type: string;
  chunk_count: number;
  indexed_at: string;
  parser_type: string;
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
  // Metadata from fallback sources (OpenAlex, Semantic Scholar, PubMed)
  authors?: string[];
  year?: number | null;
  journal?: string;
  date?: string;
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
  abstract?: string;
}

export interface SearchFilters {
  open_access: boolean;
  has_full_text: boolean;
  article_type: string;
  sort: string;
}

export interface RAGResult {
  query: string;
  answer: string;
  sources: Record<string, unknown>[];
}
