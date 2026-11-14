"""Problem specification contracts (README section 8).

These models double as LLM structured-output schemas, so they avoid JSON-schema
constructs that structured outputs handle poorly (e.g. tuples -> use ValueRange).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

DType = Literal["int", "float", "category", "text", "date", "bool"]
TaskType = Literal["regression", "classification", "ranking"]


class ValueRange(BaseModel):
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> "ValueRange":
        if self.min > self.max:
            raise ValueError(f"range min {self.min} > max {self.max}")
        return self


class Feature(BaseModel):
    name: str = Field(description="snake_case column name")
    dtype: DType
    unit: Optional[str] = None
    description: str
    required: bool = Field(
        default=True, description="Critical features block delivery if missing"
    )
    expected_range: Optional[ValueRange] = None
    allowed_values: Optional[list[str]] = None

    @field_validator("name")
    @classmethod
    def _snake_case(cls, v: str) -> str:
        v = v.strip()
        if not v or not v.replace("_", "").isalnum() or v != v.lower():
            raise ValueError(f"feature name must be lower snake_case, got {v!r}")
        return v


class TemporalPolicy(BaseModel):
    mode: Literal["recent_weighted", "window", "uniform"] = "recent_weighted"
    half_life_years: float = Field(default=2.0, gt=0)
    window_start: Optional[int] = None
    window_end: Optional[int] = None

    @model_validator(mode="after")
    def _window_complete(self) -> "TemporalPolicy":
        if self.mode == "window":
            if self.window_start is None or self.window_end is None:
                raise ValueError("window mode needs window_start and window_end")
            if self.window_start > self.window_end:
                raise ValueError("window_start > window_end")
        return self


class DatasetSpec(BaseModel):
    problem_text: str
    task_type: TaskType
    target: Feature
    features: list[Feature] = Field(min_length=1)
    region: Optional[str] = None
    temporal: TemporalPolicy = Field(default_factory=TemporalPolicy)
    min_rows: int = Field(default=1000, ge=1)

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetSpec":
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("duplicate feature names")
        if self.target.name in names:
            raise ValueError("target must not also be listed as a feature")
        return self

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]
