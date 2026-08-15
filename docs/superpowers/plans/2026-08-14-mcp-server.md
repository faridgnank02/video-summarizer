# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing `video_intelligence` pipeline to MCP clients as five tools (`analyze_video`, `get_job_status`, `get_report`, `extract_chapters`, `get_trace`) via a third adapter beside FastAPI, with zero core changes.

**Architecture:** A new `src/mcp_server/` package. A framework-free `Runtime` class holds a `pipeline_factory`, a `JobStore`, and a `TraceStore` and carries all execution logic (blocking + async, error mapping, trace footer). `server.py` registers thin `FastMCP` tools that delegate to `Runtime`, so all logic is testable without MCP. `__main__.py` selects stdio or Streamable HTTP at launch.

**Tech Stack:** Python 3.12, official MCP SDK (`mcp` / `FastMCP`), Pydantic v2, `asyncio`, SQLite (existing stores), pytest + pytest-asyncio.

## Global Constraints

- Python 3.12; async tools use `pytest-asyncio` (already a dependency).
- New dependency: `mcp>=1.2` (official Python MCP SDK). No other new deps.
- **No changes** to `src/video_intelligence/` (core) or `src/api/` (FastAPI) or `frontend/`.
- Reuse the existing `JobStore` (`src/api/jobs.py`) and `TraceStore` (`src/video_intelligence/tracing.py`) — do not create new stores.
- Argument/return types come from `src/video_intelligence/schemas.py` (`VideoSource`, `JobOptions`, `QualityPreference`, `AnalysisReport`, `Chapter`, `StageEvent`, `TraceSpan`, `SourceKind`).
- `quality` values are exactly `cheap | balanced | best` (the `QualityPreference` enum).
- All tool results are plain JSON-serializable dicts/lists (never raw Pydantic models, never tracebacks).
- Follow phase-1 test style: inject a fake pipeline through the `pipeline_factory` seam; no network, no real Whisper in the default suite; the one real-wire test is `@pytest.mark.slow`.

## File Structure

- `src/mcp_server/__init__.py` — package marker, exports `Runtime`, `build_server`.
- `src/mcp_server/runtime.py` — `Runtime` class: execution, stores, error mapping, trace footer. MCP-free.
- `src/mcp_server/server.py` — `build_server(runtime) -> FastMCP`: registers the five tools, bridges MCP `Context` progress to the runtime.
- `src/mcp_server/__main__.py` — CLI entrypoint: argparse `--transport stdio|http`, `--host`, `--port`; builds a default `Runtime` + server and runs the chosen transport.
- `tests/test_mcp_runtime.py` — runtime logic tests (fake pipeline).
- `tests/test_mcp_server.py` — tool-surface tests (in-process `FastMCP`).
- `tests/test_mcp_stdio_smoke.py` — `@pytest.mark.slow` stdio handshake.

---

### Task 1: Package scaffold, dependency, and blocking `analyze` happy path

**Files:**
- Create: `src/mcp_server/__init__.py`
- Create: `src/mcp_server/runtime.py`
- Create: `tests/test_mcp_runtime.py`
- Modify: `requirements.txt` (add `mcp>=1.2`)

**Interfaces:**
- Consumes: `JobStore` from `src/api/jobs.py` (`create(job_id, source, options)`, `update(job_id, status=, report=, error=, trace_id=)`, `get(job_id) -> dict | None`); `PipelineError` from `src/video_intelligence/pipeline.py` (attrs `.stage`, `.reason`); schemas `VideoSource`, `JobOptions`, `QualityPreference`, `SourceKind`, `StageEvent`, `AnalysisReport`.
- Produces: `Runtime(pipeline_factory=build_pipeline, db_path="data/app.db", trace_db="data/traces.db")`; `async Runtime.analyze(url, quality="balanced", language="en", force_whisper=False, async_=False, on_event=None) -> dict`; internal `async Runtime._execute(job_id, source, options, on_event) -> dict`. A completed blocking result is `{"status": "completed", "job_id": str, "trace_id": str, "report": dict}`.

- [ ] **Step 1: Add the dependency**

Add under a new `# MCP` section at the end of `requirements.txt`:

```
# MCP
mcp>=1.2
```

Then install:

```bash
pip install "mcp>=1.2"
```

- [ ] **Step 2: Create the package marker**

Create `src/mcp_server/__init__.py`:

```python
"""MCP adapter over the video_intelligence pipeline."""
from .runtime import Runtime

__all__ = ["Runtime"]
```

- [ ] **Step 3: Write the failing test for blocking analyze**

Create `tests/test_mcp_runtime.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_mcp_runtime.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `Runtime`.

- [ ] **Step 5: Implement the minimal Runtime**

Create `src/mcp_server/runtime.py`:

```python
"""Execution runtime for the MCP adapter — framework-free, testable without MCP."""
from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from src.api.jobs import JobStore
from src.video_intelligence.pipeline import PipelineError, build_pipeline
from src.video_intelligence.schemas import (
    JobOptions, QualityPreference, SourceKind, StageEvent, VideoSource,
)
from src.video_intelligence.tracing import TraceStore

EventCallback = Callable[[StageEvent], Awaitable[None]]


async def _noop(_ev: StageEvent) -> None:
    return None


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
        report = await pipeline.run(source, options)
        self.jobs.update(job_id, status="completed", report=report,
                         trace_id=report.trace_id)
        return {"status": "completed", "job_id": job_id,
                "trace_id": report.trace_id, "report": report.model_dump()}

    async def analyze(self, url: str, quality: str = "balanced",
                      language: str = "en", force_whisper: bool = False,
                      async_: bool = False,
                      on_event: EventCallback | None = None) -> dict:
        source = VideoSource(kind=SourceKind.YOUTUBE, url=url)
        options = JobOptions(language=language,
                             quality=QualityPreference(quality),
                             force_whisper=force_whisper)
        job_id = uuid.uuid4().hex
        self.jobs.create(job_id, source, options)
        return await self._execute(job_id, source, options, on_event or _noop)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_mcp_runtime.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt src/mcp_server/__init__.py src/mcp_server/runtime.py tests/test_mcp_runtime.py
git commit -m "feat: add MCP runtime with blocking analyze happy path"
```

---

### Task 2: Error mapping, degraded reports, async path, `get_job_status`, `get_report`

**Files:**
- Modify: `src/mcp_server/runtime.py`
- Modify: `tests/test_mcp_runtime.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: `_execute` now returns `{"status": "failed", "job_id", "stage", "reason"}` on `PipelineError` and `{"status": "failed", "job_id", "reason"}` on any other exception (never raises). `async Runtime.analyze(..., async_=True)` returns `{"status": "running", "job_id"}` immediately. `Runtime.job_status(job_id) -> dict` = `{"status", "degraded_stages", "error"}` (or `{"status": "not_found"}`). `Runtime.get_report(job_id) -> dict` = the report dict, or `{"status", "error"}` when unavailable, or `{"status": "not_found"}`.

- [ ] **Step 1: Write failing tests for errors, degraded, async, and reads**

Append to `tests/test_mcp_runtime.py`:

```python
import asyncio


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcp_runtime.py -v`
Expected: FAIL — `PipelineError` propagates (no mapping), `analyze` has no `async_` background branch, `job_status`/`get_report` undefined.

- [ ] **Step 3: Implement error mapping, async path, and read methods**

In `src/mcp_server/runtime.py`, add `import asyncio` at the top, add a module-level task set after the imports:

```python
_background_tasks: set[asyncio.Task] = set()
```

Replace `_execute` with the error-mapping version:

```python
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
                "trace_id": report.trace_id, "report": report.model_dump()}
```

Replace the end of `analyze` (the `return await self._execute(...)` line) with the async branch:

```python
        self.jobs.create(job_id, source, options)
        if async_:
            task = asyncio.create_task(
                self._execute(job_id, source, options, _noop))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            return {"status": "running", "job_id": job_id}
        return await self._execute(job_id, source, options, on_event or _noop)
```

Add the two read methods to the class:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp_runtime.py -v`
Expected: PASS (all tests, including Task 1's).

- [ ] **Step 5: Commit**

```bash
git add src/mcp_server/runtime.py tests/test_mcp_runtime.py
git commit -m "feat: add MCP runtime error mapping, async jobs, and read methods"
```

---

### Task 3: `extract_chapters`, `get_trace`, and the blocking trace footer

**Files:**
- Modify: `src/mcp_server/runtime.py`
- Modify: `tests/test_mcp_runtime.py`

**Interfaces:**
- Consumes: everything from Tasks 1–2; `TraceStore.add_span(trace_id, span)`, `TraceStore.spans(trace_id) -> list[TraceSpan]`, `TraceStore.total_cost(trace_id) -> float`; schema `TraceSpan`.
- Produces: `async Runtime.extract_chapters(url, quality="balanced", language="en", on_event=None) -> list[dict] | dict` (chapter dicts on success, the failed-result dict otherwise). `Runtime.get_trace(trace_id) -> {"spans": list[dict], "total_cost_usd": float}`. Blocking `analyze` completed result now also carries `"trace": {"total_cost_usd": float, "stages": [{"stage", "model_used", "latency_ms"}]}`.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_mcp_runtime.py`:

```python
from src.video_intelligence.schemas import Chapter, TraceSpan


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mcp_runtime.py -v`
Expected: FAIL — `extract_chapters`/`get_trace` undefined, no `trace` key in the completed result.

- [ ] **Step 3: Implement the trace footer, `extract_chapters`, and `get_trace`**

In `src/mcp_server/runtime.py`, add a `TraceSpan` import (extend the schemas import line to include `TraceSpan`), then add a private footer helper and attach it in `_execute`'s completed return. Change the completed `return` in `_execute` to:

```python
        self.jobs.update(job_id, status="completed", report=report,
                         trace_id=report.trace_id)
        return {"status": "completed", "job_id": job_id,
                "trace_id": report.trace_id, "report": report.model_dump(),
                "trace": self._trace_footer(report.trace_id)}
```

Add the helper and the two public methods to the class:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp_runtime.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mcp_server/runtime.py tests/test_mcp_runtime.py
git commit -m "feat: add extract_chapters, get_trace, and blocking trace footer"
```

---

### Task 4: FastMCP server with the five tools and progress bridging

**Files:**
- Create: `src/mcp_server/server.py`
- Modify: `src/mcp_server/__init__.py`
- Create: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `Runtime` and all its methods from Tasks 1–3; `FastMCP` and `Context` from `mcp.server.fastmcp`; `StageEvent`.
- Produces: `build_server(runtime: Runtime) -> FastMCP` registering tools named exactly `analyze_video`, `get_job_status`, `get_report`, `extract_chapters`, `get_trace`. `__init__.py` also exports `build_server`.

- [ ] **Step 1: Write the failing tool-surface test**

Create `tests/test_mcp_server.py`:

```python
import pytest

from src.mcp_server.runtime import Runtime
from src.mcp_server.server import build_server
from src.video_intelligence.schemas import AnalysisReport, StageEvent


class FakePipeline:
    def __init__(self, on_event, report):
        self._on_event = on_event
        self._report = report

    async def run(self, source, options):
        if self._on_event is not None:
            await self._on_event(StageEvent(stage="ingest", type="started"))
        return self._report


def make_server(tmp_path):
    report = AnalysisReport(summary="ok", language="en", trace_id="tr1")

    def factory(on_event=None):
        return FakePipeline(on_event, report)

    rt = Runtime(pipeline_factory=factory,
                 db_path=tmp_path / "app.db",
                 trace_db=tmp_path / "traces.db")
    return build_server(rt)


@pytest.mark.asyncio
async def test_server_registers_all_five_tools(tmp_path):
    server = make_server(tmp_path)
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert names == {"analyze_video", "get_job_status", "get_report",
                     "extract_chapters", "get_trace"}


@pytest.mark.asyncio
async def test_analyze_video_tool_input_schema_defaults_quality(tmp_path):
    server = make_server(tmp_path)
    tools = {t.name: t for t in await server.list_tools()}
    schema = tools["analyze_video"].inputSchema
    assert schema["properties"]["quality"]["default"] == "balanced"
    assert "url" in schema["required"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_server.py -v`
Expected: FAIL — `src.mcp_server.server` / `build_server` does not exist.

- [ ] **Step 3: Implement the server**

Create `src/mcp_server/server.py`:

```python
"""FastMCP tool surface bridging MCP calls to the Runtime."""
from __future__ import annotations

from mcp.server.fastmcp import Context, FastMCP

from src.video_intelligence.schemas import StageEvent

from .runtime import Runtime


def build_server(runtime: Runtime) -> FastMCP:
    mcp = FastMCP("Video Intelligence")

    def _progress(ctx: Context):
        async def on_event(ev: StageEvent) -> None:
            await ctx.info(f"{ev.stage}: {ev.type}"
                           + (f" - {ev.message}" if ev.message else ""))
        return on_event

    @mcp.tool()
    async def analyze_video(url: str, ctx: Context, quality: str = "balanced",
                            language: str = "en", force_whisper: bool = False,
                            async_: bool = False) -> dict:
        """Analyze a YouTube video into summary, chapters, quotes, and action items."""
        return await runtime.analyze(
            url=url, quality=quality, language=language,
            force_whisper=force_whisper, async_=async_,
            on_event=_progress(ctx))

    @mcp.tool()
    async def get_job_status(job_id: str) -> dict:
        """Status of an async analyze_video job."""
        return runtime.job_status(job_id)

    @mcp.tool()
    async def get_report(job_id: str) -> dict:
        """The analysis report for a completed job."""
        return runtime.get_report(job_id)

    @mcp.tool()
    async def extract_chapters(url: str, ctx: Context, quality: str = "balanced",
                               language: str = "en") -> list | dict:
        """Timestamped chapters for a YouTube video."""
        return await runtime.extract_chapters(
            url=url, quality=quality, language=language, on_event=_progress(ctx))

    @mcp.tool()
    async def get_trace(trace_id: str) -> dict:
        """Cost and latency spans for a completed analysis."""
        return runtime.get_trace(trace_id)

    return mcp
```

Update `src/mcp_server/__init__.py`:

```python
"""MCP adapter over the video_intelligence pipeline."""
from .runtime import Runtime
from .server import build_server

__all__ = ["Runtime", "build_server"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_server.py -v`
Expected: PASS. (If `list_tools()` return shape differs by SDK version, the tools are still named as asserted; adjust only the accessor, never the names.)

- [ ] **Step 5: Commit**

```bash
git add src/mcp_server/server.py src/mcp_server/__init__.py tests/test_mcp_server.py
git commit -m "feat: add FastMCP server exposing the five video-intelligence tools"
```

---

### Task 5: CLI entrypoint (stdio/HTTP) and stdio handshake smoke test

**Files:**
- Create: `src/mcp_server/__main__.py`
- Create: `tests/test_mcp_stdio_smoke.py`

**Interfaces:**
- Consumes: `Runtime`, `build_server`; `mcp` client helpers `StdioServerParameters` and `stdio_client`, `ClientSession`.
- Produces: `python -m src.mcp_server --transport {stdio,http} [--host H] [--port P]`. `build_server` runs with `transport="stdio"` or `transport="streamable-http"`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_mcp_stdio_smoke.py`:

```python
import pytest

mcp_client = pytest.importorskip("mcp.client.stdio")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


@pytest.mark.slow
@pytest.mark.asyncio
async def test_stdio_handshake_lists_all_tools(tmp_path):
    params = StdioServerParameters(
        command="python",
        args=["-m", "src.mcp_server", "--transport", "stdio"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
    names = {t.name for t in listed.tools}
    assert names == {"analyze_video", "get_job_status", "get_report",
                     "extract_chapters", "get_trace"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_stdio_smoke.py -v -m slow`
Expected: FAIL — `python -m src.mcp_server` exits nonzero (`__main__.py` missing), so the handshake never completes.

- [ ] **Step 3: Implement the entrypoint**

Create `src/mcp_server/__main__.py`:

```python
"""CLI entrypoint: launch the MCP server over stdio or Streamable HTTP."""
from __future__ import annotations

import argparse

from .runtime import Runtime
from .server import build_server


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.mcp_server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = build_server(Runtime())
    if args.transport == "http":
        server.settings.host = args.host
        server.settings.port = args.port
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_stdio_smoke.py -v -m slow`
Expected: PASS (spawns the server, completes the MCP handshake, lists five tools).

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: all phase-1 tests plus the new MCP tests PASS; no core/API tests changed.

- [ ] **Step 6: Commit**

```bash
git add src/mcp_server/__main__.py tests/test_mcp_stdio_smoke.py
git commit -m "feat: add MCP CLI entrypoint with stdio/HTTP transports and smoke test"
```

---

## Notes for the implementer

- **Do not touch** `src/video_intelligence/` or `src/api/`. If a test seems to need a core change, stop — it almost certainly means the tool result should adapt, not the core.
- The `pytest-asyncio` mode is already configured in the phase-1 suite; the `@pytest.mark.asyncio` markers match existing async tests (see `tests/test_pipeline.py`). If markers are ignored, check `pyproject.toml`/`pytest.ini` for `asyncio_mode` and follow whatever the existing async tests do.
- `JobStore.get(job_id)["report"]` is already a plain dict (the store `json.loads`es it), which is why `get_report` and `job_status` read it directly without re-parsing.
- The `slow` marker is used by phase-1's `test_whisper_smoke.py`; reuse it, don't invent a new one.
