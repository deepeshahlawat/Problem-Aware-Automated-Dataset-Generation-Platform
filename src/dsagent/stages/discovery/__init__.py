"""Stage 2: source discovery, trust ranking, complementary selection, feasibility report.

discover_sources(spec, catalogs, llm) -> DiscoveryResult

* Entity sources (row-level records) must contain the target and match the region.
* Context sources (joined on a key, decision D2) are chosen only for features the
  selected entity sources do not cover, and only if their join key is available.
* The feasibility report is the human approval point before any expensive acquisition
  (design spec section 15). Nothing is fabricated: unknown means unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from dsagent.config import Settings, get_settings
from dsagent.llm.budget import BudgetExceeded
from dsagent.llm.client import LLMClient
from dsagent.schemas.report import FeasibilityItem
from dsagent.schemas.source import SourceCandidate, SourceDescriptor
from dsagent.schemas.spec import DatasetSpec
from dsagent.stages.discovery.catalogs import Catalog
from dsagent.stages.discovery.matching import coverage, match_columns, name_score
from dsagent.trust.source_trust import SourceNotPermitted, trust_components, trust_score


@dataclass
class DiscoveryResult:
    candidates: list[SourceCandidate]
    selected: list[str]
    rejected: list[tuple[str, str]]
    feasibility: list[FeasibilityItem]
    deliverable: bool
    blocking: list[str]
    warnings: list[str] = field(default_factory=list)

    def selected_candidates(self) -> list[SourceCandidate]:
        by_id = {c.source_id: c for c in self.candidates}
        return [by_id[s] for s in self.selected]

    def feasibility_markdown(self) -> str:
        lines = [
            "| Feature | Status | Sources | Coverage | Newest data | Trust | Note |",
            "|---|---|---|---|---|---|---|",
        ]
        for it in self.feasibility:
            cov = "n/a" if it.coverage is None else f"{it.coverage:.0%}"
            newest = it.newest_date.date().isoformat() if it.newest_date else "unknown"
            trust = "n/a" if it.trust is None else f"{it.trust:.2f}"
            lines.append(
                f"| {it.feature} | {it.status} | {', '.join(it.sources) or 'none'} | {cov} "
                f"| {newest} | {trust} | {it.note} |"
            )
        return "\n".join(lines)


def primary_place(region: str) -> str:
    """'Delhi, India' -> 'delhi'; 'New York City, USA' -> 'new york city'."""
    return region.split(",")[0].strip().lower()


def _region_ok(src: SourceDescriptor, region: str | None) -> bool:
    if not region:
        return True
    r = primary_place(region)
    if src.region:
        s = src.region.lower()
        return r in s or s in region.lower()
    hay = " ".join([src.title, src.source_id, src.notes]).lower().replace("_", " ").replace("-", " ")
    if r in hay:
        return True
    return any(isinstance(v, str) and r in v.lower() for row in src.sample_rows[:50] for v in row.values())


def _required(spec: DatasetSpec) -> list[str]:
    return [spec.target.name] + [f.name for f in spec.features if f.required]


def _score_entity(src, spec, llm, settings, now, rejected, warnings):
    if not _region_ok(src, spec.region):
        rejected.append((src.source_id, f"region mismatch (wanted {spec.region})"))
        return None
    fmap, method, notes = match_columns(spec, src.columns, src.sample_rows, llm)
    warnings += [f"{src.source_id}: {n}" for n in notes]
    if spec.target.name not in fmap:
        rejected.append((src.source_id, f"target '{spec.target.name}' not found in columns"))
        return None
    # Topic relevance: a 'price' column alone does not make a crude-oil table a laptop
    # source. Require a minimum share of the non-target features as well.
    n_feat = len(spec.features)
    overlap = sum(1 for f in spec.features if f.name in fmap)
    need = max(settings.discovery.min_feature_overlap_abs,
               round(settings.discovery.min_feature_overlap_frac * n_feat))
    if overlap < min(need, n_feat):
        rejected.append((src.source_id, f"off-topic: only {overlap}/{n_feat} spec features besides the target"))
        return None
    cov = {f: coverage(src.sample_rows, c) for f, c in fmap.items()}
    req = _required(spec)
    cov_term = sum(1 for f in req if cov.get(f, 0) > 0) / len(req)
    try:
        comps = trust_components(src, cov_term, spec.temporal.half_life_years, settings, now)
    except SourceNotPermitted as exc:
        rejected.append((src.source_id, str(exc)))
        return None
    return SourceCandidate(
        source_id=src.source_id, url=src.url, kind=src.kind, role="entity", title=src.title,
        authority_class=src.authority_class, covers_features=sorted(fmap),
        feature_map=fmap, coverage_by_feature=cov, est_rows=src.est_rows or len(src.sample_rows) or None,
        last_updated=src.last_updated, licence=src.licence, robots_allowed=src.robots_allowed,
        trust_score=trust_score(comps, settings), trust_components=comps,
        notes="; ".join(filter(None, [src.notes, _llm_note(method)])),
    )


def _llm_note(method: dict[str, str]) -> str:
    llm_matched = [f for f, m in method.items() if m == "llm"]
    return f"llm-matched: {', '.join(llm_matched)}" if llm_matched else ""


def _context_covers(src: SourceDescriptor, wanted: list[str]) -> list[str]:
    return [w for w in wanted if any(name_score(w, p) >= 0.75 for p in src.provides)]


def _score_context(src, spec, settings, now, rejected):
    wanted = [f.name for f in spec.features]
    covers = _context_covers(src, wanted)
    provides_keys = sorted({p for p in src.provides})
    if not covers and not provides_keys:
        return None
    try:
        frac = len(covers) / len(src.provides) if src.provides else 0.0
        comps = trust_components(src, frac, spec.temporal.half_life_years, settings, now)
    except SourceNotPermitted as exc:
        rejected.append((src.source_id, str(exc)))
        return None
    return SourceCandidate(
        source_id=src.source_id, url=src.url, kind=src.kind, role="context", title=src.title,
        authority_class=src.authority_class,
        join_key=" | ".join("+".join(alt) for alt in src.join_keys_any) or None,
        covers_features=covers, last_updated=src.last_updated, licence=src.licence,
        robots_allowed=src.robots_allowed, trust_score=trust_score(comps, settings),
        trust_components=comps, notes=src.notes,
    ), src


def discover_sources(
    spec: DatasetSpec,
    catalogs: list[Catalog],
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> DiscoveryResult:
    settings = settings or get_settings()
    d = settings.discovery
    warnings: list[str] = []
    rejected: list[tuple[str, str]] = []

    descriptors: list[SourceDescriptor] = []
    for cat in catalogs:
        try:
            descriptors += cat.find(spec)
        except BudgetExceeded:
            raise
        except Exception as exc:
            warnings.append(f"catalog {cat.name} failed: {exc}")
        warnings += getattr(cat, "warnings", [])

    entities: list[SourceCandidate] = []
    contexts: list[tuple[SourceCandidate, SourceDescriptor]] = []
    for src in descriptors:
        if src.role == "entity":
            c = _score_entity(src, spec, llm, settings, now, rejected, warnings)
            if c:
                entities.append(c)
        else:
            r = _score_context(src, spec, settings, now, rejected)
            if r:
                contexts.append(r)

    entities.sort(key=lambda c: -c.trust_score)
    contexts.sort(key=lambda x: -x[0].trust_score)

    # --- complementary selection -------------------------------------------------
    selected: list[SourceCandidate] = []
    for c in entities:
        if c.trust_score < d.min_trust:
            rejected.append((c.source_id, f"trust {c.trust_score:.2f} below minimum {d.min_trust}"))
        elif len(selected) < d.max_entity_sources:
            selected.append(c)
    have = {f for c in selected for f in c.feature_map}

    def key_available(src: SourceDescriptor, extra: set[str]) -> list[str] | None:
        for alt in src.join_keys_any or [[]]:
            if all(k in have or k in extra for k in alt):
                return alt
        return None

    chosen_ctx: list[SourceCandidate] = []
    provided: set[str] = set()
    for f in [x.name for x in spec.features]:
        if f in have or f in provided or not selected:
            continue
        for cand, src in contexts:
            if f not in cand.covers_features or cand.trust_score < d.min_trust:
                continue
            if key_available(src, provided) is not None:
                if cand not in chosen_ctx:
                    chosen_ctx.append(cand)
                    provided |= set(cand.covers_features)
                break
            # one level of dependency: another context source supplies the join key
            for alt in src.join_keys_any:
                helpers = [(hc, hs) for hc, hs in contexts if hs is not src and set(alt) <= set(hs.provides)]
                if helpers and key_available(helpers[0][1], provided) is not None:
                    hc = helpers[0][0]
                    for x in (hc, cand):
                        if x not in chosen_ctx:
                            chosen_ctx.append(x)
                    provided |= set(cand.covers_features) | set(alt)
                    break
            if f in provided:
                break

    sel_ids = [c.source_id for c in selected + chosen_ctx][: settings.selection.max_candidate_sources]

    # --- feasibility report ---------------------------------------------------------
    feas: list[FeasibilityItem] = []
    total_rows = sum(c.est_rows or 0 for c in selected)
    for feat in [spec.target, *spec.features]:
        f = feat.name
        ent = [c for c in selected if f in c.feature_map]
        ctx = [c for c in chosen_ctx if f in c.covers_features]
        if ent:
            if total_rows:
                cov = sum(c.coverage_by_feature.get(f, 0) * (c.est_rows or 0) for c in ent) / total_rows
            else:
                cov = max(c.coverage_by_feature.get(f, 0) for c in ent)
            status = "found" if cov >= d.found_coverage else "partial"
            srcs, note = ent, ""
        elif ctx:
            status, cov, srcs = "derivable", None, ctx
            note = f"joined via {ctx[0].join_key}" + (" (derived)" if ctx[0].kind == "api" else "")
        else:
            status, cov, srcs, note = "not_found", 0.0, [], ""
        if f == spec.target.name:
            note = ("target; " + note).strip("; ")
        elif not feat.required and status == "not_found":
            note = "optional"
        dates = [c.last_updated for c in srcs if c.last_updated]
        feas.append(FeasibilityItem(
            feature=f, status=status, sources=[c.source_id for c in srcs],
            coverage=None if cov is None else round(cov, 4),
            newest_date=max(dates) if dates else None,
            trust=max((c.trust_score for c in srcs), default=None), note=note,
        ))

    blocking: list[str] = []
    by_f = {it.feature: it for it in feas}
    if by_f[spec.target.name].status not in ("found", "partial"):
        blocking.append(f"target '{spec.target.name}' not available from any selected entity source")
    for feat in spec.features:
        if feat.required and by_f[feat.name].status == "not_found":
            blocking.append(f"required feature '{feat.name}' not found")
    if selected and total_rows and total_rows < spec.min_rows:
        blocking.append(f"only ~{total_rows} rows available, spec asks for at least {spec.min_rows}")

    all_cands = entities + [c for c, _ in contexts]
    return DiscoveryResult(
        candidates=all_cands, selected=sel_ids, rejected=rejected, feasibility=feas,
        deliverable=not blocking, blocking=blocking, warnings=warnings,
    )
