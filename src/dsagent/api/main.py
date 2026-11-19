"""Web API + demo UI (design spec section 23, MVP form).

    .venv/Scripts/python -m uvicorn dsagent.api.main:app --port 8000

Endpoints
  GET  /                                  demo UI (static/index.html)
  GET  /api/sources                       registered sources (data/sources.json)
  POST /api/jobs                          {problem}            -> Stage 1, spec for review
  GET  /api/jobs                          recent runs
  GET  /api/jobs/{id}                     job status and results
  POST /api/jobs/{id}/spec/approve        {drop_features, overrides} -> Stage 2, feasibility for review
  POST /api/jobs/{id}/feasibility/approve {continue_on_review} -> Stages 4-7 in the background
  GET  /api/jobs/{id}/preview             first rows of the generated dataset
  GET  /api/jobs/{id}/download/{name}     dataset / reports

Jobs live in memory (single-user demo; the spec's Celery/Redis queue is deferred).
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dsagent import pipeline
from dsagent.config import get_settings
from dsagent.llm import BudgetExceeded
from dsagent.llm.client import LLMError
from dsagent.stages.requirement import NeedsClarification, RequirementError

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="dsagent", version="0.1.0")
JOBS: dict[str, pipeline.Job] = {}


class NewJob(BaseModel):
    problem: str = Field(min_length=3, max_length=500)


class ApproveSpec(BaseModel):
    drop_features: list[str] = Field(default_factory=list)
    overrides: dict[str, dict] = Field(default_factory=dict)


class ApproveFeasibility(BaseModel):
    continue_on_review: bool = False


def _job(job_id: str) -> pipeline.Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def _llm_errors(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NeedsClarification as exc:
        raise HTTPException(422, {"kind": "clarification", "message": exc.question})
    except (RequirementError, ValueError) as exc:
        raise HTTPException(422, {"kind": "invalid", "message": str(exc)})
    except BudgetExceeded as exc:
        raise HTTPException(402, {"kind": "budget", "message": str(exc)})
    except LLMError as exc:
        raise HTTPException(502, {"kind": "llm", "message": str(exc)})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/sources")
def sources() -> list[dict]:
    s = get_settings()
    path = pipeline.ROOT / s.source_manifest_path
    if not path.exists():
        return []
    out = []
    for e in json.loads(path.read_text(encoding="utf-8")):
        local = (path.parent / e.get("path", "")).exists() if e.get("path") else False
        out.append({k: e.get(k) for k in ("source_id", "title", "url", "licence", "region", "role")} | {"available": local})
    return out


@app.post("/api/jobs")
def create_job(body: NewJob) -> dict:
    job = _llm_errors(pipeline.start_job, body.problem)
    JOBS[job.id] = job
    return pipeline.job_view(job)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    """Recent runs in this server session, newest first."""
    jobs = sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)[:20]
    return [{"id": j.id, "problem": j.problem, "status": j.status, "created_at": j.created_at} for j in jobs]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return pipeline.job_view(_job(job_id))


@app.post("/api/jobs/{job_id}/spec/approve")
def approve_spec(job_id: str, body: ApproveSpec) -> dict:
    job = _job(job_id)
    if job.status not in ("spec_ready", "feasibility_ready"):
        raise HTTPException(409, f"cannot approve the spec while the job is '{job.status}'")
    _llm_errors(pipeline.approve_spec, job, body.drop_features, None, body.overrides)
    return pipeline.job_view(job)


@app.post("/api/jobs/{job_id}/feasibility/approve")
def approve_feasibility(job_id: str, body: ApproveFeasibility, tasks: BackgroundTasks) -> dict:
    job = _job(job_id)
    if job.status not in ("feasibility_ready", "needs_review", "failed", "done"):
        raise HTTPException(409, f"cannot build while the job is '{job.status}'")
    if job.discovery is None or not job.discovery.deliverable:
        raise HTTPException(422, {"kind": "not_deliverable",
                                  "message": "; ".join(job.discovery.blocking if job.discovery else ["no discovery"])})
    job.status = "building"
    tasks.add_task(pipeline.build, job, body.continue_on_review)
    return pipeline.job_view(job)


@app.get("/api/jobs/{job_id}/preview")
def preview(job_id: str, rows: int = 20) -> dict:
    job = _job(job_id)
    if job.dataset is None:
        raise HTTPException(409, "dataset not built yet")
    df = job.dataset.head(max(1, min(rows, 200)))
    return {"columns": df.columns, "rows": json.loads(df.write_json())}


@app.get("/api/jobs/{job_id}/download/{name}")
def download(job_id: str, name: str) -> FileResponse:
    job = _job(job_id)
    if name not in pipeline.DOWNLOADS or job.out_dir is None or not (job.out_dir / name).exists():
        raise HTTPException(404, "file not available")
    return FileResponse(job.out_dir / name, filename=f"{job.id}_{name}")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
