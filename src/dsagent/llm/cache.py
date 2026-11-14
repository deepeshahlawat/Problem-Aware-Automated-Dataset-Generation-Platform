"""LLM call cache (README section 12.2, decision D8).

Temperature 0 is not deterministic on hosted models, so reproducibility comes from
replaying stored responses: an identical request returns the stored answer.
Key = sha256 of (model, prompt template version, system, user prompt, output schema).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CachedResponse:
    response_json: str
    model_served: str
    input_tokens: int
    output_tokens: int
    created_at: str


def make_key(
    *, model: str, template_version: str, system: str, user: str, output_schema: dict
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "template_version": template_version,
            "system": system,
            "user": user,
            "schema": output_schema,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCallCache:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS llm_cache (
                   key TEXT PRIMARY KEY,
                   model_requested TEXT NOT NULL,
                   model_served TEXT NOT NULL,
                   template_version TEXT NOT NULL,
                   response_json TEXT NOT NULL,
                   input_tokens INTEGER NOT NULL,
                   output_tokens INTEGER NOT NULL,
                   created_at TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    def get(self, key: str) -> CachedResponse | None:
        row = self._conn.execute(
            "SELECT response_json, model_served, input_tokens, output_tokens, created_at "
            "FROM llm_cache WHERE key = ?",
            (key,),
        ).fetchone()
        return CachedResponse(*row) if row else None

    def put(
        self,
        key: str,
        *,
        model_requested: str,
        model_served: str,
        template_version: str,
        response_json: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                model_requested,
                model_served,
                template_version,
                response_json,
                input_tokens,
                output_tokens,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
