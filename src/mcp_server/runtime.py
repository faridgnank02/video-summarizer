"""Execution runtime for the MCP adapter — framework-free, testable without MCP."""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from src.api.jobs import JobStore
from src.video_intelligence.agents.factchecker import build_factchecker
from src.video_intelligence.pipeline import PipelineError, build_pipeline
from src.video_intelligence.schemas import (
    AnalysisReport, JobOptions, QualityPreference, SourceKind, StageEvent, TraceSpan,
    VideoSource,
)
from src.video_intelligence.tracing import TraceStore

EventCallback = Callable[[StageEvent], Awaitable[None]]


async def _noop(_ev: StageEvent) -> None:
    return None


_background_tasks: set[asyncio.Task] = set()


class Runtime:
    def __init__(self, pipeline_factory=build_pipeline,
                 checker_factory=build_factchecker,
                 db_path: str | Path = "data/app.db",
                 trace_db: str | Path = "data/traces.db"):
        self.factory = pipeline_factory
        self.checker_factory = checker_factory
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
                      analyze_visuals: bool = False,
                      fact_check: bool = False, async_: bool = False,
                      on_event: EventCallback | None = None) -> dict:
        try:
            quality_pref = QualityPreference(quality)
        except ValueError:
            return {"status": "failed", "reason": f"invalid quality: {quality!r}"}
        source = VideoSource(kind=SourceKind.YOUTUBE, url=url)
        options = JobOptions(language=language,
                             quality=quality_pref,
                             force_whisper=force_whisper,
                             analyze_visuals=analyze_visuals,
                             fact_check=fact_check)
        job_id = uuid.uuid4().hex
        self.jobs.create(job_id, source, options)
        if async_:
            task = asyncio.create_task(
                self._execute(job_id, source, options, _noop))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            return {"status": "running", "job_id": job_id}
        return await self._execute(job_id, source, options, on_event or _noop)

    async def fact_check(self, job_id: str | None = None, url: str | None = None,
                         claims: list[str] | None = None, quality: str = "balanced",
                         language: str = "en",
                         on_event: EventCallback | None = None) -> dict:
        provided = [x is not None for x in (job_id, url, claims)]
        if sum(provided) != 1:
            return {"status": "failed",
                    "reason": "provide exactly one of job_id, url, or claims"}
        try:
            quality_pref = QualityPreference(quality)
        except ValueError:
            return {"status": "failed", "reason": f"invalid quality: {quality!r}"}

        checker = self.checker_factory()

        if url is not None:
            result = await self.analyze(url=url, quality=quality, language=language,
                                        fact_check=True, async_=False,
                                        on_event=on_event)
            if result.get("status") != "completed":
                return result
            report = AnalysisReport.model_validate(result["report"])
            report.fact_checks = await checker.run(report, None, quality_pref,
                                                    report.trace_id)
            self.jobs.update(result["job_id"], report=report)
            return report.model_dump()
        if claims is not None:
            results = await checker.check(claims, quality_pref, uuid.uuid4().hex)
            return {"fact_checks": [fc.model_dump() for fc in results]}

        job = self.jobs.get(job_id)
        if job is None or job["report"] is None:
            return {"status": "failed", "reason": "no completed report for job_id"}
        report = AnalysisReport.model_validate(job["report"])
        report.fact_checks = await checker.run(report, None, quality_pref, report.trace_id)
        self.jobs.update(job_id, report=report)
        return {"fact_checks": [fc.model_dump() for fc in report.fact_checks]}

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
                               analyze_visuals: bool = False,
                               on_event: EventCallback | None = None):
        result = await self.analyze(url=url, quality=quality, language=language,
                                    analyze_visuals=analyze_visuals,
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
