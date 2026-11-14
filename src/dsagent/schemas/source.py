"""Source and provenance contracts (README sections 8, 10, 13)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

SourceKind = Literal["api", "file", "open_repo", "html_table", "jsonld", "web_page"]
# Decision D2: entity sources are stacked (union); context sources are joined on a key.
SourceRole = Literal["entity", "context"]
ExtractionMethod = Literal["api", "file", "html_table", "jsonld", "llm", "derived"]


class SourceCandidate(BaseModel):
    source_id: str
    url: str
    kind: SourceKind
    role: SourceRole
    join_key: Optional[str] = Field(
        default=None, description="Canonical column a context source joins on, e.g. locality"
    )
    covers_features: list[str]
    est_rows: Optional[int] = None
    last_updated: Optional[datetime] = None
    licence: Optional[str] = None
    robots_allowed: bool
    trust_score: float = Field(ge=0, le=1)
    notes: str = ""


class Provenance(BaseModel):
    source_id: str
    source_url: str
    retrieved_at: datetime
    method: ExtractionMethod
    llm_verified: Optional[bool] = None
    parser_id: Optional[str] = Field(default=None, description="Recipe-cache parser version")
