import asyncio

import pytest

from src.mcp_server.runtime import Runtime
from src.video_intelligence.pipeline import PipelineError
from src.video_intelligence.schemas import AnalysisReport, Chapter, StageEvent, TraceSpan


class FakePipeline:
    def __init__(self, on_event, report=None, error: PipelineError | None = None):
        self._on_event = on_event
        self._report = report
        self._error = error

    async def run(self, source, options):
        if self._on_event is not None:
            await self._on_event(StageEvent(stage="ingest", type="started"))
        if self._error:
            raise self._error
        return self._report


def make_runtime(tmp_path, report=None, error=None):
    def factory(on_event=None):
        return FakePipeline(on_event, report=report, error=error)

    return Runtime(pipeline_factory=factory,
                   db_path=tmp_path / "app.db",
                   trace_db=tmp_path / "traces.db")


SAMPLE_REPORT = AnalysisReport(summary="Great talk.", language="en", trace_id="tr1")


@pytest.mark.asyncio
async def test_blocking_analyze_returns_completed_report(tmp_path):
    rt = make_runtime(tmp_path, report=SAMPLE_REPORT)
    result = await rt.analyze(url="https://youtu.be/x")
    assert result["status"] == "completed"
    assert result["trace_id"] == "tr1"
    assert result["report"]["summary"] == "Great talk."
    # persisted to the job store as completed
    job = rt.jobs.get(result["job_id"])
    assert job["status"] == "completed"


@pytest.mark.asyncio
async def test_analyze_invalid_quality_returns_structured_failure(tmp_path):
    rt = make_runtime(tmp_path, report=SAMPLE_REPORT)
    result = await rt.analyze(url="https://youtu.be/x", quality="high")
    assert result["status"] == "failed"
    assert "high" in result["reason"]


@pytest.mark.asyncio
async def test_extract_chapters_invalid_quality_forwards_failure(tmp_path):
    rt = make_runtime(tmp_path, report=SAMPLE_REPORT)
    result = await rt.extract_chapters(url="https://youtu.be/x", quality="high")
    assert result["status"] == "failed"
    assert "high" in result["reason"]


DEGRADED_REPORT = AnalysisReport(summary="Partial.", language="en",
                                 trace_id="tr2", degraded_stages=["chaptering"])


@pytest.mark.asyncio
async def test_pipeline_error_maps_to_failed_result(tmp_path):
    rt = make_runtime(tmp_path, error=PipelineError("transcribe", "no audio"))
    result = await rt.analyze(url="https://youtu.be/x")
    assert result["status"] == "failed"
    assert result["stage"] == "transcribe"
    assert result["reason"] == "no audio"
    assert rt.jobs.get(result["job_id"])["status"] == "failed"


@pytest.mark.asyncio
async def test_unexpected_error_maps_to_generic_failed_result(tmp_path):
    rt = make_runtime(tmp_path, error=RuntimeError("boom"))
    result = await rt.analyze(url="https://youtu.be/x")
    assert result["status"] == "failed"
    assert "stage" not in result
    assert "boom" in result["reason"]


@pytest.mark.asyncio
async def test_degraded_report_completes_with_degraded_stages(tmp_path):
    rt = make_runtime(tmp_path, report=DEGRADED_REPORT)
    result = await rt.analyze(url="https://youtu.be/x")
    assert result["status"] == "completed"
    assert result["report"]["degraded_stages"] == ["chaptering"]


@pytest.mark.asyncio
async def test_async_analyze_returns_job_id_then_report(tmp_path):
    rt = make_runtime(tmp_path, report=SAMPLE_REPORT)
    result = await rt.analyze(url="https://youtu.be/x", async_=True)
    assert result["status"] == "running"
    job_id = result["job_id"]
    # let the background task finish
    for _ in range(50):
        if rt.job_status(job_id)["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert rt.job_status(job_id)["status"] == "completed"
    assert rt.get_report(job_id)["summary"] == "Great talk."


def test_job_status_and_report_not_found(tmp_path):
    rt = make_runtime(tmp_path)
    assert rt.job_status("nope") == {"status": "not_found"}
    assert rt.get_report("nope") == {"status": "not_found"}


REPORT_WITH_CHAPTERS = AnalysisReport(
    summary="s", language="en", trace_id="tr3",
    chapters=[Chapter(start_s=0, end_s=10, title="Intro", synopsis="hello")])


@pytest.mark.asyncio
async def test_extract_chapters_projects_only_chapters(tmp_path):
    rt = make_runtime(tmp_path, report=REPORT_WITH_CHAPTERS)
    chapters = await rt.extract_chapters(url="https://youtu.be/x")
    assert isinstance(chapters, list)
    assert chapters[0]["title"] == "Intro"


@pytest.mark.asyncio
async def test_extract_chapters_forwards_failure(tmp_path):
    rt = make_runtime(tmp_path, error=PipelineError("ingest", "bad url"))
    result = await rt.extract_chapters(url="https://youtu.be/x")
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_blocking_analyze_attaches_trace_footer(tmp_path):
    rt = make_runtime(tmp_path, report=SAMPLE_REPORT)
    rt.traces.add_span("tr1", TraceSpan(stage="synthesize",
                                        model_used="fake/big",
                                        cost_usd=0.05, latency_ms=1200))
    result = await rt.analyze(url="https://youtu.be/x")
    assert result["trace"]["total_cost_usd"] == 0.05
    assert result["trace"]["stages"][0] == {
        "stage": "synthesize", "model_used": "fake/big", "latency_ms": 1200}


def test_get_trace_returns_spans_and_cost(tmp_path):
    rt = make_runtime(tmp_path)
    rt.traces.add_span("trX", TraceSpan(stage="synthesize",
                                        model_used="fake/big", cost_usd=0.05))
    trace = rt.get_trace("trX")
    assert trace["total_cost_usd"] == 0.05
    assert trace["spans"][0]["stage"] == "synthesize"
