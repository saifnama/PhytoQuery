from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class NERRequest(BaseModel):
    doi: str = Field(..., min_length=1, example="10.1016/j.phytochem.2021.112818")


class Entity(BaseModel):
    text: str
    label: str
    score: float
    canonical: Optional[str] = None  # Normalized form for display
    preferred_name: Optional[str] = None
    aliases: Optional[List[str]] = None  # All variations for counting
    name_type: Optional[str] = None
    linked_to: Optional[str] = None
    scientific_name_verified: Optional[str] = None
    accepted_scientific_name: Optional[str] = None
    common_name: Optional[str] = None
    inchikey: Optional[str] = None
    smiles: Optional[str] = None
    molecular_formula: Optional[str] = None
    source_db: Optional[str] = None
    source_url: Optional[str] = None
    taxon_id: Optional[str] = None
    match_status: Optional[str] = None
    review_required: Optional[str] = None


class NERResponse(BaseModel):
    doi: str
    mode: str  # "full_text" or "abstract"
    text: str
    entities: List[Entity]


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        example="What are the main bioactive compounds in Ocimum sanctum?",
    )
    selected_files: Optional[List[str]] = None  # Filter RAG to only these sources
    chat_history: Optional[List[ChatMessage]] = None  # Previous conversation turns


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


# --- Citation schemas (industry-standard structured-output pattern) ---
#
# The two-pass pipeline:
#   Pass 1 (streaming): the LLM emits the answer with [chunk_id] markers
#                        inline. Frontend renders them as [1] [2] etc.
#   Pass 2 (one-shot JSON): a fast follow-up call extracts verbatim
#                            quotes per cited chunk_id. The LLM is
#                            forced into JSON mode and its output is
#                            validated against the Citations schema
#                            below, then fuzzy-matched against the
#                            actual chunk text to drop hallucinations.


class Citation(BaseModel):
    """One verbatim quote backing one claim in the answer."""

    chunk_id: str = Field(
        ...,
        description="The chunk_id of the source chunk that supports this claim.",
    )
    quote: str = Field(
        ...,
        description="Exact verbatim sentence or phrase from the chunk used.",
    )


class Citations(BaseModel):
    """Wrapper schema the JSON-mode LLM call must conform to.

    A wrapper rather than a bare list because providers' JSON mode
    require an object at the top level — bare arrays are rejected
    by Groq, OpenRouter, and Ollama format='json'.
    """

    citations: List[Citation] = Field(default_factory=list)


class IndexedFileInfo(BaseModel):
    name: str
    file_type: str
    chunk_count: int
    indexed_at: str
    parser_type: str
    authors: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    summary: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    message: str
    files: List[str]
    summaries: Optional[Dict[str, str]] = None  # filename -> summary
    job_id: Optional[str] = None  # Set when processing is async


class UploadJobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    message: str
    files: List[str]
    parser_type: str
    summaries: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
