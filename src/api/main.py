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
        else:
            store.update(job_id, status="completed", report=report, trace_id=report.trace_id)
            await queue.put(StageEvent(stage="pipeline", type="completed"))
        await queue.put(None)  # sentinel: closes the SSE stream

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
        dest = upload_dir / f"{uuid.uuid4().hex}-{file.filename}"
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
            if queue is None or job["status"] in ("completed", "failed"):
                # job already finished (or process restarted): replay terminal status
                ev = StageEvent(stage="pipeline", type=job["status"], message=job.get("error"))
                yield f"data: {ev.model_dump_json()}\n\n"
                return
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield f"data: {ev.model_dump_json()}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = create_app()
