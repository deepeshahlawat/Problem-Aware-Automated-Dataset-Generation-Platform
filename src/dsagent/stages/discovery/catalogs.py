"""Source catalogs for Stage 2 (design spec section 10, decision D4).

Curated sources first, live search second:
  * LocalManifestCatalog  - hand-downloaded open files listed in data/sources.json
  * CuratedContextCatalog - known context sources (OSM metro, circle rates, ...)
                            in configs/context_sources.json
  * HuggingFaceCatalog    - live search of the Hugging Face Hub (no key needed)

Catalogs only describe sources and read a small real sample; they never invent
columns or values. Every HTTP request counts against the run's page budget.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from dsagent.llm.budget import Budget
from dsagent.schemas.source import SourceDescriptor
from dsagent.schemas.spec import DatasetSpec
from dsagent.text import norm


class Catalog(Protocol):
    name: str

    def find(self, spec: DatasetSpec) -> list[SourceDescriptor]: ...


def read_csv_sample(path: Path, max_rows: int) -> tuple[list[str], list[dict], int]:
    """Header, first `max_rows` rows, and total row count of a CSV file."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with path.open(newline="", encoding=enc) as fh:
                reader = csv.DictReader(fh)
                columns = list(reader.fieldnames or [])
                rows, total = [], 0
                for row in reader:
                    total += 1
                    if len(rows) < max_rows:
                        rows.append(row)
                return columns, rows, total
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


class LocalManifestCatalog:
    """Hand-downloaded files. Manifest entries are SourceDescriptor fields plus `path`
    (relative to the manifest's folder). Missing files are reported, not guessed."""

    name = "local"

    def __init__(self, manifest_path: Path, sample_rows: int = 200) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_rows = sample_rows
        self.warnings: list[str] = []

    def find(self, spec: DatasetSpec) -> list[SourceDescriptor]:
        if not self.manifest_path.exists():
            self.warnings.append(f"no local manifest at {self.manifest_path}")
            return []
        entries = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        out = []
        for e in entries:
            e = dict(e)
            rel = e.pop("path", None)
            if rel:
                p = (self.manifest_path.parent / rel).resolve()
                if not p.exists():
                    self.warnings.append(f"{e.get('source_id')}: file not found ({rel})")
                    continue
                e["local_path"] = str(p)
                if p.suffix.lower() == ".csv":
                    cols, rows, total = read_csv_sample(p, self.sample_rows)
                    e.setdefault("columns", cols)
                    e["sample_rows"] = rows
                    e.setdefault("est_rows", total)
            e.setdefault("kind", "file")
            e.setdefault("url", f"file://{rel}" if rel else "")
            out.append(SourceDescriptor(**e, catalog=self.name))
        return out


class CuratedContextCatalog:
    """Known context sources. Kept only if their region is global or matches the spec."""

    name = "curated"

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)

    def find(self, spec: DatasetSpec) -> list[SourceDescriptor]:
        if not self.registry_path.exists():
            return []
        entries = json.loads(self.registry_path.read_text(encoding="utf-8"))
        region = (spec.region or "").lower()
        out = []
        for e in entries:
            src = SourceDescriptor(**e, catalog=self.name)
            if src.region and (not region or src.region.lower() not in region):  # e.g. Delhi in "delhi, india"
                continue
            out.append(src)
        return out


_STOP = {
    "i", "we", "want", "would", "like", "to", "predict", "predicting", "estimate", "forecast",
    "classify", "the", "a", "an", "of", "in", "for", "from", "its", "it", "will", "whether",
    "next", "year", "based", "on", "using", "and", "or", "with", "what", "which", "per",
}


def search_queries(spec: DatasetSpec, max_queries: int = 3) -> list[str]:
    """Hub search matches dataset names, so short noun phrases work best
    ('house price'); the region is checked afterwards, not searched."""
    region_words = set(norm(spec.region or "").split("_"))
    words = [
        w[:-1] if len(w) > 3 and w.endswith("s") else w
        for w in re.findall(r"[a-z0-9]+", spec.problem_text.lower())
        if w not in _STOP and w not in region_words
    ]
    target_words = [w for w in norm(spec.target.name).split("_") if w not in _STOP]
    queries: list[str] = []
    for i in range(len(words) - 1):
        queries.append(f"{words[i]} {words[i + 1]}")
    if words and target_words and target_words[0] not in words:
        queries.append(f"{words[0]} {target_words[0]}")
    queries += words[:1]
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_queries]


class HuggingFaceCatalog:
    """Live Hub search + datasets-server column sample (first rows only)."""

    name = "huggingface"
    API = "https://huggingface.co/api/datasets"
    SERVER = "https://datasets-server.huggingface.co"

    def __init__(self, budget: Budget, user_agent: str, limit: int = 8, http_client=None) -> None:
        if http_client is None:
            import httpx

            http_client = httpx.Client(timeout=30, headers={"User-Agent": user_agent})
        self.http = http_client
        self.budget = budget
        self.limit = limit
        self.warnings: list[str] = []

    def _get(self, url: str):
        self.budget.record_page()
        r = self.http.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        return r.json()

    def find(self, spec: DatasetSpec) -> list[SourceDescriptor]:
        seen: dict[str, dict] = {}
        for q in search_queries(spec):
            try:
                hits = self._get(f"{self.API}?search={quote(q)}&limit={self.limit}&full=true")
            except Exception as exc:  # network problems degrade to fewer sources
                self.warnings.append(f"search '{q}' failed: {exc}")
                continue
            for h in hits:
                seen.setdefault(h["id"], h)
        out = []
        for ds_id, h in list(seen.items())[: self.limit]:
            try:
                out.append(self._describe(ds_id, h))
            except Exception as exc:
                self.warnings.append(f"{ds_id}: no column sample ({exc})")
        return out

    def _describe(self, ds_id: str, hit: dict) -> SourceDescriptor:
        splits = self._get(f"{self.SERVER}/splits?dataset={quote(ds_id, safe='')}")["splits"]
        if not splits:
            raise RuntimeError("no splits")
        sp = next((s for s in splits if s["split"] == "train"), splits[0])
        fr = self._get(
            f"{self.SERVER}/first-rows?dataset={quote(ds_id, safe='')}"
            f"&config={quote(sp['config'])}&split={quote(sp['split'])}"
        )
        columns = [f["name"] for f in fr.get("features", [])]
        rows = [r["row"] for r in fr.get("rows", [])]
        card = hit.get("cardData") or {}
        lic = card.get("license")
        if isinstance(lic, list):
            lic = lic[0] if lic else None
        if not lic:
            lic = next((t.split(":", 1)[1] for t in hit.get("tags", []) if t.startswith("license:")), None)
        modified = hit.get("lastModified")
        return SourceDescriptor(
            source_id=f"hf:{ds_id}",
            title=ds_id,
            url=f"https://huggingface.co/datasets/{ds_id}",
            kind="open_repo",
            role="entity",
            authority_class="community",
            licence=lic,
            last_updated=datetime.fromisoformat(modified.replace("Z", "+00:00")) if modified else None,
            columns=columns,
            sample_rows=rows,
            catalog=self.name,
            notes=f"downloads={hit.get('downloads')}; sample of {len(rows)} rows from datasets-server",
        )
