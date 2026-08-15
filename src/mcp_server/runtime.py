"""Execution runtime for the MCP adapter — framework-free, testable without MCP."""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from src.api.jobs import JobStore
from src.video_intelligence.pipeline import PipelineError, build_pipeline
from src.video_intelligence.schemas import (
    JobOptions, QualityPreference, SourceKind, StageEvent, TraceSpan, VideoSource,
)
from src.video_intelligence.tracing import TraceStore

EventCallback = Callable[[StageEvent], Awaitable[None]]


async def _noop(_ev: StageEvent) -> None:
    return None


_background_tasks: set[asyncio.Task] = set()


class Runtime:
    def __init__(self, pipeline_factory=build_pipeline,
                 db_path: str | Path = "data/app.db",
                 trace_db: str | Path = "data/traces.db"):
        self.factory = pipeline_factory
        self.jobs = JobStore(db_path)
        self.traces = TraceStore(trace_db)

    async def _execute(self, job_id: str, source: VideoSource,
                       options: JobOptions, on_event: EventCallback) -> dict:
        pipeline = self.factory(on_event=on_event)
        self.jobs.update(job_id, status="running")
        try:
            report = await pipeline.run(source, options)
        except PipelineError as e:
            self.jobs.update(job_id, status="failed", error=str(e))
            return {"status": "failed", "job_id": job_id,
                    "stage": e.stage, "reason": e.reason}
        except Exception as e:  # never let a crash escape the tool boundary
            self.jobs.update(job_id, status="failed", error=f"unexpected error: {e}")
            return {"status": "failed", "job_id": job_id, "reason": str(e)}
        self.jobs.update(job_id, status="completed", report=report,
                         trace_id=report.trace_id)
        return {"status": "completed", "job_id": job_id,
                "trace_id": report.trace_id, "report": report.model_dump(),
                "trace": self._trace_footer(report.trace_id)}

    async def analyze(self, url: str, quality: str = "balanced",
                      language: str = "en", force_whisper: bool = False,
                      async_: bool = False,
                      on_event: EventCallback | None = None) -> dict:
        try:
            quality_pref = QualityPreference(quality)
        except ValueError:
            return {"status": "failed", "reason": f"invalid quality: {quality!r}"}
        source = VideoSource(kind=SourceKind.YOUTUBE, url=url)
        options = JobOptions(language=language,
                             quality=quality_pref,
                             force_whisper=force_whisper)
        job_id = uuid.uuid4().hex
        self.jobs.create(job_id, source, options)
        if async_:
            task = asyncio.create_task(
                self._execute(job_id, source, options, _noop))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            return {"status": "running", "job_id": job_id}
        return await self._execute(job_id, source, options, on_event or _noop)

    def _trace_footer(self, trace_id: str) -> dict:
        spans = self.traces.spans(trace_id)
        return {
            "total_cost_usd": self.traces.total_cost(trace_id),
            "stages": [{"stage": s.stage, "model_used": s.model_used,
                        "latency_ms": s.latency_ms} for s in spans],
        }

    def get_trace(self, trace_id: str) -> dict:
        spans = self.traces.spans(trace_id)
        return {"spans": [s.model_dump() for s in spans],
                "total_cost_usd": self.traces.total_cost(trace_id)}

    async def extract_chapters(self, url: str, quality: str = "balanced",
                               language: str = "en",
                               on_event: EventCallback | None = None):
        result = await self.analyze(url=url, quality=quality, language=language,
                                    async_=False, on_event=on_event)
        if result["status"] != "completed":
            return result
        return result["report"]["chapters"]

    def job_status(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            return {"status": "not_found"}
        report = job["report"] or {}
        return {"status": job["status"],
                "degraded_stages": report.get("degraded_stages", []),
                "error": job["error"]}

    def get_report(self, job_id: str) -> dict:
        job = self.jobs.get(job_id)
        if job is None:
            return {"status": "not_found"}
        if job["report"] is None:
            return {"status": job["status"], "error": job["error"]}
        return job["report"]
