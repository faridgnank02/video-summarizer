# src/api/main.py
"""FastAPI adapter over the video_intelligence pipeline."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.video_intelligence.pipeline import PipelineError, build_pipeline
from src.video_intelligence.schemas import (
    JobOptions, QualityPreference, SourceKind, StageEvent, VideoSource,
)
from src.video_intelligence.tracing import TraceStore

from .jobs import JobStore


class CreateJobRequest(BaseModel):
    url: str
    options: JobOptions = JobOptions()


def create_app(pipeline_factory=build_pipeline,
               db_path: str | Path = "data/app.db",
               trace_db: str | Path = "data/traces.db",
               upload_dir: str | Path = "data/uploads") -> FastAPI:
    app = FastAPI(title="Video Intelligence")
    store = JobStore(db_path)
    trace_store = TraceStore(trace_db)
    upload_dir = Path(upload_dir)
    queues: dict[str, asyncio.Queue] = {}

    async def _run_job(job_id: str, source: VideoSource, options: JobOptions) -> None:
        queue = queues[job_id]

        async def on_event(ev: StageEvent) -> None:
            await queue.put(ev)

        pipeline = pipeline_factory(on_event=on_event)
        store.update(job_id, status="running")
        try:
            report = await pipeline.run(source, options)
        except PipelineError as e:
            store.update(job_id, status="failed", error=str(e))
            await queue.put(StageEvent(stage=e.stage, type="failed", message=e.reason))
        except Exception as e:  # any unexpected failure must not wedge the job
            store.update(job_id, status="failed", error=f"unexpected error: {e}")
            await queue.put(StageEvent(stage="pipeline", type="failed", message=str(e)))
        else:
            store.update(job_id, status="completed", report=report, trace_id=report.trace_id)
            await queue.put(StageEvent(stage="pipeline", type="completed"))
        finally:
            await queue.put(None)          # always close the SSE stream
            # Prune unconditionally once the job is done. A live subscriber captured
            # the queue OBJECT in gen() (queues.get) before this pop, so removing the
            # dict entry does not break its drain -- it reads the object to the None
            # sentinel and returns. A subscriber that attaches after completion finds
            # queue is None and gets the terminal status replayed from the store.
            # Popping here (rather than in gen()) guarantees cleanup even when no
            # client ever calls /events, bounding the queues dict.
            queues.pop(job_id, None)

    def _start_job(background: BackgroundTasks, source: VideoSource,
                   options: JobOptions) -> dict:
        job_id = uuid.uuid4().hex
        store.create(job_id, source, options)
        queues[job_id] = asyncio.Queue()
        background.add_task(_run_job, job_id, source, options)
        return {"job_id": job_id}

    @app.post("/api/jobs")
    async def create_job(req: CreateJobRequest, background: BackgroundTasks) -> dict:
        source = VideoSource(kind=SourceKind.YOUTUBE, url=req.url)
        return _start_job(background, source, req.options)

    @app.post("/api/jobs/upload")
    async def upload_job(background: BackgroundTasks,
                         file: UploadFile = File(...),
                         language: str = Form("en"),
                         quality: str = Form("balanced"),
                         force_whisper: bool = Form(False)) -> dict:
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "upload").name or "upload"
        dest = upload_dir / f"{uuid.uuid4().hex}-{safe_name}"
        dest.write_bytes(await file.read())
        source = VideoSource(kind=SourceKind.LOCAL_FILE, path=str(dest), title=file.filename)
        options = JobOptions(language=language, quality=QualityPreference(quality),
                             force_whisper=force_whisper)
        return _start_job(background, source, options)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/api/jobs/{job_id}/trace")
    async def get_trace(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        trace_id = job.get("trace_id")
        spans = trace_store.spans(trace_id) if trace_id else []
        return {"spans": [s.model_dump() for s in spans],
                "total_cost_usd": trace_store.total_cost(trace_id) if trace_id else 0.0}

    @app.get("/api/jobs/{job_id}/events")
    async def stream_events(job_id: str) -> StreamingResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def gen():
            queue = queues.get(job_id)
            if queue is None:
                # No queue for this job: it predates this process (restart) or its
                # backlog was already drained and pruned by an earlier subscriber.
                # Re-fetch the job fresh rather than trusting the outer `job` snapshot
                # taken before this generator ran -- that snapshot can be stale (e.g.
                # still "running") if the background task finished in between, which
                # would otherwise replay the wrong status to a late subscriber.
                fresh = store.get(job_id)
                status = fresh["status"] if fresh is not None else job["status"]
                error = fresh.get("error") if fresh is not None else job.get("error")
                ev = StageEvent(stage="pipeline", type=status, message=error)
                yield f"data: {ev.model_dump_json()}\n\n"
                return
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield f"data: {ev.model_dump_json()}\n\n"
            # Cleanup is owned by _run_job's finally (which pops the dict entry once
            # the job ends). This subscriber still holds the queue object reference,
            # so its drain above is unaffected by that pop.

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = create_app()
