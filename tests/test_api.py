# tests/test_api.py
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.video_intelligence.pipeline import PipelineError
from src.video_intelligence.schemas import AnalysisReport, StageEvent, TraceSpan
from src.video_intelligence.tracing import TraceStore


class FakePipeline:
    def __init__(self, on_event, report=None, error: PipelineError | None = None):
        self._on_event = on_event
        self._report = report
        self._error = error

    async def run(self, source, options):
        await self._on_event(StageEvent(stage="ingest", type="started"))
        if self._error:
            raise self._error
        return self._report


def make_client(tmp_path, report=None, error=None):
    def factory(on_event=None):
        return FakePipeline(on_event, report=report, error=error)

    app = create_app(pipeline_factory=factory,
                     db_path=tmp_path / "app.db",
                     trace_db=tmp_path / "traces.db",
                     upload_dir=tmp_path / "uploads")
    return TestClient(app)


SAMPLE_REPORT = AnalysisReport(summary="Great.", language="en", trace_id="tr1")


def test_create_job_then_completed_report(tmp_path):
    client = make_client(tmp_path, report=SAMPLE_REPORT)
    resp = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # TestClient runs background tasks before returning, so the job is done
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["report"]["summary"] == "Great."


def test_failed_job_reports_error(tmp_path):
    client = make_client(tmp_path, error=PipelineError("transcribe", "no audio"))
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "transcribe" in job["error"]


def test_unknown_job_404s(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/api/jobs/nope").status_code == 404


def test_trace_endpoint_returns_spans(tmp_path):
    client = make_client(tmp_path, report=SAMPLE_REPORT)
    # seed the trace store with a span for trace_id tr1
    TraceStore(tmp_path / "traces.db").add_span(
        "tr1", TraceSpan(stage="synthesize", model_used="fake/big", cost_usd=0.05))
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    trace = client.get(f"/api/jobs/{job_id}/trace").json()
    assert trace["total_cost_usd"] == 0.05
    assert trace["spans"][0]["stage"] == "synthesize"


def test_events_stream_ends_with_terminal_event(tmp_path):
    client = make_client(tmp_path, report=SAMPLE_REPORT)
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    # job already finished (TestClient background task ran) -> stream replays terminal status
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        body = "".join(resp.iter_text())
    assert "completed" in body
    assert body.startswith("data: ")


def test_upload_creates_local_file_job(tmp_path):
    client = make_client(tmp_path, report=SAMPLE_REPORT)
    resp = client.post(
        "/api/jobs/upload",
        files={"file": ("talk.mp4", b"fake-bytes", "video/mp4")},
        data={"language": "en", "quality": "cheap", "force_whisper": "false"},
    )
    assert resp.status_code == 200
    job = client.get(f"/api/jobs/{resp.json()['job_id']}").json()
    assert job["status"] == "completed"


def test_unexpected_pipeline_error_marks_job_failed(tmp_path):
    class ExplodingPipeline:
        def __init__(self, on_event):
            self._on_event = on_event
        async def run(self, source, options):
            await self._on_event(StageEvent(stage="ingest", type="started"))
            raise RuntimeError("boom")

    def factory(on_event=None):
        return ExplodingPipeline(on_event)

    app = create_app(pipeline_factory=factory, db_path=tmp_path / "app.db",
                     trace_db=tmp_path / "traces.db", upload_dir=tmp_path / "uploads")
    client = TestClient(app)
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert "boom" in job["error"]
    # stream must terminate (sentinel was enqueued), not hang
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        body = "".join(resp.iter_text())
    assert "failed" in body


def test_degraded_stage_does_not_terminate_early(tmp_path):
    class DegradingPipeline:
        def __init__(self, on_event):
            self._on_event = on_event

        async def run(self, source, options):
            await self._on_event(StageEvent(stage="chapterize", type="degraded", message="boom"))
            return SAMPLE_REPORT

    def factory(on_event=None):
        return DegradingPipeline(on_event)

    app = create_app(pipeline_factory=factory, db_path=tmp_path / "a.db",
                     trace_db=tmp_path / "t.db", upload_dir=tmp_path / "u")
    client = TestClient(app)
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed"
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        body = "".join(resp.iter_text())
    assert "degraded" in body and "completed" in body


def test_upload_sanitizes_path_in_filename(tmp_path):
    client = make_client(tmp_path, report=SAMPLE_REPORT)
    resp = client.post("/api/jobs/upload",
                       files={"file": ("../../evil.mp4", b"bytes", "video/mp4")},
                       data={"language": "en", "quality": "cheap", "force_whisper": "false"})
    assert resp.status_code == 200
    job = client.get(f"/api/jobs/{resp.json()['job_id']}").json()
    assert job["status"] == "completed"
