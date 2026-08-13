from pydantic import BaseModel, Field
from typing import List, Optional

class Source(BaseModel):
    id: str
    language: str
    chunk_strategy: str
    score: float
    text: str

class AnswerEnvelope(BaseModel):
    answer: str
    language: str
    grounded: bool
    refused: bool
    refusal_reason: Optional[str] = None
    sources: List[Source] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

class TextAsk(BaseModel):
    query: str
    language: Optional[str] = None
