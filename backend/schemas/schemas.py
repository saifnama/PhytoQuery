from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class NERRequest(BaseModel):
    doi: str = Field(..., min_length=1, example="10.1016/j.phytochem.2021.112818")


class Entity(BaseModel):
    text: str
    label: str
    score: float
    canonical: Optional[str] = None  # Normalized form for display
    aliases: Optional[List[str]] = None  # All variations for counting


class NERResponse(BaseModel):
    doi: str
    mode: str  # "full_text" or "abstract"
    text: str
    entities: List[Entity]


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        example="What are the main bioactive compounds in Ocimum sanctum?",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    status: str
    message: str
    files: List[str]
