# MCP Server — Design

**Date:** 2026-08-14
**Status:** Approved (design), pending spec review
**Phase:** 2 of the "Agentic Multimodal Video Intelligence" transformation
**Branch:** `claude/mcp-server-phase-2` (off `claude/agentic-video-intelligence-d59cc7`)

## Goal

Expose the existing `video_intelligence` pipeline to MCP clients (Claude
Desktop, Claude Code, any MCP host) as a small set of tools, so an agent can
analyze a video and read back its chapters, report, and cost/latency trace.
This is the highest-visibility "2026" feature and the cheapest remaining phase:
a **third adapter beside FastAPI over the same core**, with zero changes to the
core library.

## Scope

**In:** a new `src/mcp_server/` package built on the official Python MCP SDK
(`mcp` / `FastMCP`); five tools (`analyze_video`, `get_job_status`,
`get_report`, `extract_chapters`, `get_trace`); blocking execution with MCP
progress notifications plus an async job-handle fallback; both stdio and
Streamable HTTP transports selectable at launch; reuse of the existing
`JobStore` and `TraceStore`; tests with fakes and one slow transport smoke test.

**Out (deferred, unchanged from phase-1 roadmap):**

- `fact_check_claims` — needs the Phase-4 Fact-Checker agent + web search, which
  does not exist yet. Not stubbed; simply absent this phase.
- Local-file upload over MCP — MCP clients pass URLs, not multipart uploads;
  local files stay a FastAPI concern.
- Any change to the `video_intelligence` core, the FastAPI adapter, or the
  frontend.
- Visual agent, live-stream mode, deployment re-fit, evaluation re-wire (their
  own later phases).

## Architecture

A new package `src/mcp_server/`, a third adapter sitting beside `src/api/`
(FastAPI) over the identical `video_intelligence` core. Built on the official
Python MCP SDK, whose `FastMCP` generates JSON-Schema tool definitions from
typed Python signatures and serves both stdio and Streamable HTTP from one tool
module.

```
src/
  video_intelligence/   # core library — UNCHANGED this phase
  api/                  # FastAPI adapter — UNCHANGED this phase
  mcp_server/
    __init__.py
    server.py           # FastMCP instance + the five tool registrations
    runtime.py          # pipeline execution, JobStore + TraceStore wiring,
                        #   blocking + async job runner, error mapping
    __main__.py         # entrypoint: python -m src.mcp_server
                        #   --transport stdio|http [--host --port]
```

Key decisions:

- **Third sibling adapter, core untouched.** The MCP server imports
  `build_pipeline`, `Pipeline.run`, the Pydantic schemas, `JobStore`
  (`src/api/jobs.py`), and `TraceStore` (`src/video_intelligence/tracing.py`).
  It adds no intelligence of its own.
- **Reuse the existing stores.** MCP jobs are written to the same `JobStore`
  SQLite rows and the same `TraceStore` as FastAPI. A job started over MCP is
  therefore the same row the web UI can render — cross-adapter visibility for
  free.
- **Both transports from one tool module.** A launch flag picks stdio (local,
  "add to Claude Desktop") or Streamable HTTP (networked, fits the Phase-2e AWS
  deployment). Tool definitions are identical across transports.
- **Official MCP SDK, not hand-rolled JSON-RPC.** `FastMCP` derives tool
  schemas from typed signatures, matching phase-1's "types are the contract"
  approach.

## Tool Surface

Argument and return types are the existing `video_intelligence.schemas`
Pydantic models wherever possible, so tool schemas are generated, not
hand-written.

### 1. `analyze_video` — primary tool

- **Args:** `url: str` (YouTube), `quality: "cheap" | "balanced" | "best" =
  "balanced"`, `language: str = "en"`, `force_whisper: bool = False`,
  `async_: bool = False`.
- **Blocking (default, `async_=False`):** runs the pipeline to completion.
  Each core `StageEvent` from the `on_event` channel is translated into an MCP
  progress notification (`Context.report_progress` / `Context.info`). Returns
  the full `AnalysisReport` (summary, chapters, key_quotes, action_items,
  degraded_stages) **plus a compact trace footer**: `total_cost_usd`, and
  per-stage `{model_used, latency_ms}`. The report is also persisted to
  `JobStore` so it is retrievable later via `get_report`.
- **Async (`async_=True`):** creates a job row, launches the pipeline on a
  background task, and returns `{job_id, status: "running"}` immediately.
  Timeout-proof for long videos.

### 2. `get_job_status`

- **Args:** `job_id: str`.
- **Returns:** `{status, degraded_stages, error?}` where `status ∈ queued |
  running | completed | failed`. For polling async jobs.

### 3. `get_report`

- **Args:** `job_id: str`.
- **Returns:** the `AnalysisReport` for a completed job, or a structured
  `{status: "running" | "failed", ...}` result when the report is not (yet)
  available. Completes the async flow.

### 4. `extract_chapters`

- **Args:** `url: str`, `quality`, `language` (same defaults as
  `analyze_video`).
- **Returns:** `list[Chapter]` only.
- **Honesty note:** chapters require a transcript, so this runs the **same full
  pipeline** and projects out the chapters — it is a narrower *result shape*,
  not a cheaper code path. A chapters-only fast path was considered and
  rejected as premature (YAGNI; it would fork the pipeline wiring).

### 5. `get_trace`

- **Args:** `trace_id: str`.
- **Returns:** `list[TraceSpan]` + `total_cost_usd`, read directly from
  `TraceStore`. The observability tool; surfaces the "$2 vs $20" story.

### Deliberately excluded

- `fact_check_claims` (Phase 4 — no backing agent yet).
- File upload (URLs only over MCP).

## Async Job Runner

`runtime.py` owns execution. The FastAPI adapter's `_run_job` is coupled to a
per-subscriber `asyncio.Queue` for HTTP-SSE; MCP has no such subscriber model
(progress goes through the SDK's per-call `Context`, async results are polled
from the store). So the runner is an adapter-appropriate rewrite, not a copy,
built around one shared execution core:

- **`_execute(job_id, source, options, on_event)`** — the single code path both
  modes call, so blocking and async cannot drift. Persists status transitions
  (`running → completed | failed`) and the final report to `JobStore`.
- **Blocking path:** `on_event` closes over the call's MCP `Context` and emits
  `report_progress` / `info`. The pipeline runs inline; the report is persisted
  and returned.
- **Async path:** `on_event` is a no-op sink (nobody is listening on a poll
  model). The pipeline runs via `asyncio.create_task`. A module-level task set
  holds a strong reference to each task until it completes, so fire-and-forget
  tasks are not garbage-collected mid-flight.

## Error Handling

Pipeline failures become structured MCP tool results, never raw tracebacks:

- **`PipelineError(stage, reason)`** (essential-stage failure) → tool result
  `{status: "failed", stage, reason}`; for async jobs the same reason is
  written to the job row's `error`.
- **Degraded** (non-essential stage, e.g. chapterizer): the run **succeeds** —
  the report ships with `degraded_stages` populated and a note in the result,
  mirroring the core's existing degraded-flow contract.
- **Unexpected exceptions** are caught at the tool boundary and mapped to a
  generic `{status: "failed", reason}` so a crash cannot wedge the MCP session.

## Testing

Mirrors phase-1's `FakeProvider` / fake-pipeline approach — no network and no
real Whisper in the default suite. The pipeline is injected through the same
`pipeline_factory` seam FastAPI uses.

- **Runtime unit tests** (fake pipeline injected):
  - blocking returns the full report + trace footer;
  - async returns a `job_id`, then `get_report` yields the report;
  - `PipelineError` → `{status: "failed", stage, reason}`;
  - degraded stage → completed result with `degraded_stages` populated;
  - unexpected exception → generic failed result;
  - async task strong-reference retention (task not GC'd before completion).
- **Tool-surface tests:** invoke the registered tools in-process through the
  `FastMCP` instance — assert generated schemas, argument defaults, and that
  `extract_chapters` projects only chapters. No subprocess/transport needed for
  logic.
- **Transport smoke test** (`@pytest.mark.slow`): launch the stdio transport,
  run the MCP `initialize` + `list_tools` handshake, assert the five tools
  appear. Keeps CI fast while proving the wire works.
- **Frontend:** untouched this phase.

## Dependencies

Add `mcp>=1.2` (official Python MCP SDK) to `requirements.txt`. No other new
dependencies.

## Roadmap (remaining phases, unchanged)

- **Phase 3 — Visual Agent:** slide/code/chart detection + OCR between
  Transcriber and Synthesizer.
- **Phase 4 — Fact-Checker:** claims vs. web search; unlocks
  `fact_check_claims` as a sixth MCP tool.
- **Phase 5 — Live streams:** chunked rolling summaries on the existing event
  channel.
- **Phase 2e — Deployment re-fit:** adapt Docker/compose/nginx/AWS tooling from
  `ollama-integration` to FastAPI + SPA (+ the MCP HTTP transport).
- **Evaluation re-wire:** point `src/evaluation/` (BERTScore/quality metrics) at
  the new pipeline's reports.
