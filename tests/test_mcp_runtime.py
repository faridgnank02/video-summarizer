import pytest

from src.mcp_server.runtime import Runtime
from src.video_intelligence.pipeline import PipelineError
from src.video_intelligence.schemas import AnalysisReport, StageEvent


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
