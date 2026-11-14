"""Source and provenance contracts (README sections 8, 10, 13)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

SourceKind = Literal["api", "file", "open_repo", "html_table", "jsonld", "web_page"]
# Decision D2: entity sources are stacked (union); context sources are joined on a key.
SourceRole = Literal["entity", "context"]
ExtractionMethod = Literal["api", "file", "html_table", "jsonld", "llm", "derived"]
AuthorityClass = Literal[
    "official", "academic_or_open_repo", "community", "commercial_scrape", "unknown"
]


class SourceDescriptor(BaseModel):
    """What a catalog knows about a source before it is scored for a problem."""

    source_id: str
    title: str
    url: str
    kind: SourceKind
    role: SourceRole
    authority_class: AuthorityClass = "unknown"
    licence: Optional[str] = None
    robots_allowed: bool = True
    last_updated: Optional[datetime] = None
    est_rows: Optional[int] = None
    region: Optional[str] = Field(default=None, description="None = global / any region")
    # Entity sources: real column names plus a small sample (never fabricated).
    columns: list[str] = Field(default_factory=list)
    sample_rows: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    # Context sources: canonical features they can supply or derive, and the join
    # keys they need on the entity table (any one alternative suffices).
    provides: list[str] = Field(default_factory=list)
    join_keys_any: list[list[str]] = Field(default_factory=list)
    local_path: Optional[str] = None
    catalog: str = ""
    notes: str = ""


class SourceCandidate(BaseModel):
    source_id: str
    url: str
    kind: SourceKind
    role: SourceRole
    title: str = ""
    authority_class: AuthorityClass = "unknown"
    join_key: Optional[str] = Field(
        default=None, description="Canonical column a context source joins on, e.g. locality"
    )
    covers_features: list[str]
    feature_map: dict[str, str] = Field(
        default_factory=dict, description="spec feature -> source column (entity sources)"
    )
    coverage_by_feature: dict[str, float] = Field(
        default_factory=dict, description="non-null share in the sampled rows"
    )
    est_rows: Optional[int] = None
    last_updated: Optional[datetime] = None
    licence: Optional[str] = None
    robots_allowed: bool
    trust_score: float = Field(ge=0, le=1)
    trust_components: dict[str, float] = Field(default_factory=dict)
    notes: str = ""


class Provenance(BaseModel):
    source_id: str
    source_url: str
    retrieved_at: datetime
    method: ExtractionMethod
    llm_verified: Optional[bool] = None
    parser_id: Optional[str] = Field(default=None, description="Recipe-cache parser version")
