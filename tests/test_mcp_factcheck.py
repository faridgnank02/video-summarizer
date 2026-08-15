import pytest

from src.mcp_server.runtime import Runtime
from src.mcp_server.server import build_server
from src.video_intelligence.schemas import AnalysisReport, FactCheck, ClaimVerdict, StageEvent


class FakeChecker:
    async def run(self, report, transcript, quality, trace_id):
        return [FactCheck(claim="c", verdict=ClaimVerdict.SUPPORTED, rationale="ok")]

    async def check(self, claims, quality, trace_id):
        return [FactCheck(claim=c, verdict=ClaimVerdict.REFUTED, rationale="no") for c in claims]


class FakePipeline:
    def __init__(self, on_event, report):
        self._on_event = on_event
        self._report = report

    async def run(self, source, options):
        return self._report


def make_runtime(tmp_path):
    report = AnalysisReport(summary="s", language="en", trace_id="tr1")

    def pipeline_factory(on_event=None):
        return FakePipeline(on_event, report)

    return Runtime(pipeline_factory=pipeline_factory,
                   checker_factory=lambda **_: FakeChecker(),
                   db_path=tmp_path / "app.db", trace_db=tmp_path / "traces.db")


@pytest.mark.asyncio
async def test_fact_check_requires_exactly_one_input(tmp_path):
    rt = make_runtime(tmp_path)
    assert (await rt.fact_check())["status"] == "failed"
    assert (await rt.fact_check(url="u", claims=["a"]))["status"] == "failed"


@pytest.mark.asyncio
async def test_fact_check_raw_claims(tmp_path):
    rt = make_runtime(tmp_path)
    out = await rt.fact_check(claims=["the moon is cheese"])
    assert out["fact_checks"][0]["verdict"] == "refuted"


@pytest.mark.asyncio
async def test_fact_check_by_job_id_persists(tmp_path):
    rt = make_runtime(tmp_path)
    result = await rt.analyze(url="https://youtu.be/x", async_=False)
    job_id = result["job_id"]
    out = await rt.fact_check(job_id=job_id)
    assert out["fact_checks"][0]["verdict"] == "supported"
    stored = rt.get_report(job_id)
    assert stored["fact_checks"][0]["verdict"] == "supported"


@pytest.mark.asyncio
async def test_fact_check_by_url_returns_report(tmp_path):
    rt = make_runtime(tmp_path)
    out = await rt.fact_check(url="https://youtu.be/x")
    assert "summary" in out
    assert out["fact_checks"][0]["verdict"] == "supported"


@pytest.mark.asyncio
async def test_server_registers_six_tools(tmp_path):
    report = AnalysisReport(summary="s", language="en", trace_id="tr1")
    rt = Runtime(pipeline_factory=lambda on_event=None: FakePipeline(on_event, report),
                 checker_factory=lambda **_: FakeChecker(),
                 db_path=tmp_path / "a.db", trace_db=tmp_path / "t.db")
    server = build_server(rt)
    names = {t.name for t in await server.list_tools()}
    assert "fact_check_claims" in names and len(names) == 6
