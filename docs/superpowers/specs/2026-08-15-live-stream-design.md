# Live-Stream Rolling Summaries — Design

**Date:** 2026-08-15
**Status:** Approved (design), pending spec review
**Phase:** 5 of the "Agentic Multimodal Video Intelligence" transformation
**Branch:** `claude/live-stream-phase-5` (off `claude/mcp-server-phase-2`)

## Goal

Make the pipeline emit **rolling summaries** as content arrives, instead of only
one report at the end. A `RollingSummarizer` consumes a time-ordered
`SegmentFeed`, buckets it into windows, and per window emits a short *delta*
plus an updated cumulative *running digest* over the existing SSE event channel;
at feed end it produces a final consolidated `AnalysisReport`. This delivers the
live rolling-summary UX that phase 1 reserved ("live-stream mode will ride this
same event channel") without the flaky live-download infrastructure.

## Scope

**In:**

- A `SegmentFeed` abstraction and one concrete `WindowedTranscriptFeed` that
  replays a finite `Transcript`'s segments in order.
- A `RollingSummarizer` service: windowing, per-window delta, cumulative digest
  fold, rolling-event emission, final consolidation.
- A non-optional `RollingSummarizerAgent` that drives it in a live pipeline
  variant and produces the final report.
- `build_live_pipeline(on_event=…)` = `[Ingestor, Transcriber,
  RollingSummarizerAgent]`.
- Schema changes: `StageEvent.data` (structured payload), `JobOptions.live`, a
  `RollingSummary` payload model.
- FastAPI: a `live` flag on the existing `POST /api/jobs` that selects the live
  pipeline; rolling events flow over the existing `GET /api/jobs/{id}/events`
  SSE. No new endpoint.
- Config: a `live:` block (`window_s`) and a `rolling` model task.
- Tests with `FakeProvider` + a fake/finite feed; no network.

**Out (deferred):**

- **True live ingestion** (yt-dlp HLS / a live YouTube URL). The `SegmentFeed`
  seam is exactly where a live transcribing source slots in later; this phase
  ships only the finite-transcript feed.
- **MCP.** Request/response tools do not fit a streaming interaction; live mode
  is FastAPI-SSE only this phase.
- **Frontend.** The backend emits the rolling events; a live SPA view is a later
  slice (consistent with phase 4).
- **Chapters in live mode.** Chapterization wants a whole-transcript view; the
  live pipeline omits the Chapterizer. Batch mode is unchanged and still
  chapterizes.
- Any change to the batch pipeline, the MCP adapter, or existing schemas beyond
  the two additive fields above.

## Architecture

```
src/video_intelligence/
  live/                        # NEW
    __init__.py
    feed.py                    # SegmentFeed (ABC) + WindowedTranscriptFeed
    summarizer.py              # RollingSummarizer service
  agents/
    rolling.py                 # RollingSummarizerAgent (drives the summarizer)
  pipeline.py                  # + build_live_pipeline(on_event=...)
  schemas.py                   # + StageEvent.data, JobOptions.live, RollingSummary
src/api/
  main.py                      # POST /api/jobs honors options.live -> live pipeline
```

Key decisions:

- **Feed abstraction, finite feed now.** `SegmentFeed.segments()` is an
  `AsyncIterator[TranscriptSegment]` — the unit a live transcriber would yield
  incrementally. `WindowedTranscriptFeed` yields a pre-computed transcript's
  segments in order, so the rolling machinery is exercised end-to-end offline
  and a genuinely-live feed is a drop-in later.
- **Rides the existing event channel.** `StageEvent` gains an optional
  `data: dict | None = None`; existing events omit it, so the change is
  backward compatible and needs no SSE-endpoint change (`ev.model_dump_json()`
  already serializes the whole event). Rolling updates use a new
  `type="summary"`.
- **The rolling agent replaces the Synthesizer in live mode.** In the live
  pipeline the `RollingSummarizerAgent` is the essential final agent: it streams
  per-window summaries AND produces the final `AnalysisReport`. The batch
  Synthesizer and Chapterizer are not in the live pipeline.
- **Digest fold is bounded.** Each window is summarized once (a cheap-tier
  delta), and the running digest is updated by summarizing `digest + delta`
  (small), never the whole transcript — so per-window cost is roughly constant
  regardless of stream length.

## The Rolling Loop

`RollingSummarizer.__init__(router, window_s=60, quality=BALANCED,
on_event=None)`.

`async def run(self, feed: SegmentFeed, trace_id: str) -> AnalysisReport`:

1. Pull `TranscriptSegment`s from `feed.segments()`, accumulating them into the
   current window and into a full-transcript buffer.
2. When the accumulated window span reaches `window_s` (i.e. a segment's
   `end_s` minus the window's first `start_s` ≥ `window_s`), **close the
   window**:
   - **delta**: one `router.complete(task="rolling", ...)` call over the
     window's `[M:SS] text` lines → a 1–2 sentence "what's new".
   - **digest fold**: one `router.complete(task="rolling", ...)` call over
     `running_digest + delta` → the updated cumulative digest (empty digest on
     the first window just adopts the delta's content).
   - **emit**: `await on_event(StageEvent(stage="live", type="summary",
     message=running_digest, data=RollingSummary(...).model_dump()))`.
   - reset the window buffer; keep the full-transcript buffer growing.
3. After the feed is exhausted, flush any non-empty partial window through the
   same close-window path (so the tail is summarized and emitted).
4. **Final consolidation**: one `router.complete(task="rolling", ...)` over the
   final running digest → the report `summary`. Return
   `AnalysisReport(summary=<consolidated>, chapters=[], key_quotes=[],
   action_items=[], language=<feed language>, trace_id=trace_id)`.

Windowing is by **content time** (segment timestamps), not wall-clock, so it is
deterministic and testable. `window_s` comes from config.

Trace stages: `rolling.delta`, `rolling.fold`, `rolling.consolidate` — every
model call is traced through the existing `Router`/`TraceStore` like all others.

## Schemas (`schemas.py`)

```python
class RollingSummary(BaseModel):
    window_index: int
    window_start_s: float
    window_end_s: float
    delta: str
    running_summary: str

# StageEvent gains:
    data: dict | None = None          # structured payload for e.g. type="summary"

# JobOptions gains:
    live: bool = False
```

`StageEvent.type` stays a free-form string (already `started | progress |
completed | failed | degraded`); live mode adds `summary`. The `data` field is a
plain `dict` (the `RollingSummary.model_dump()`), keeping `StageEvent` decoupled
from live-specific types.

## The Feed (`live/feed.py`)

```python
class SegmentFeed(ABC):
    language: str
    @abstractmethod
    def segments(self) -> AsyncIterator[TranscriptSegment]: ...

class WindowedTranscriptFeed(SegmentFeed):
    def __init__(self, transcript: Transcript): ...
    # yields transcript.segments in order; language = transcript.language
```

`WindowedTranscriptFeed` does not itself window — it just yields segments; the
`RollingSummarizer` owns windowing. This keeps the feed dumb and the windowing
policy in one place, and lets a future live feed yield segments as they are
transcribed without knowing the window size.

## The Agent (`agents/rolling.py`)

```python
class RollingSummarizerAgent(Agent):
    name = "rolling_summarize"
    essential = True
    def __init__(self, router: Router, on_event=None, window_s: int = 60): ...
    async def run(self, ctx: PipelineContext) -> PipelineContext:
        # requires ctx.transcript (Ingestor+Transcriber ran)
        feed = WindowedTranscriptFeed(ctx.transcript)
        summarizer = RollingSummarizer(self._router, window_s=self._window_s,
                                       quality=ctx.options.quality,
                                       on_event=self._on_event)
        ctx.report = await summarizer.run(feed, ctx.trace_id)
        return ctx
```

The agent is essential (its report is the live job's report). It receives the
same `on_event` the adapter passes to `build_live_pipeline`, so its rolling
events land on the identical channel as the pipeline's own stage events.

## Pipeline Wiring (`pipeline.py`)

```python
def build_live_pipeline(config_path="config/models.yaml",
                        db_path="data/traces.db", workdir="data/work",
                        on_event=None) -> Pipeline:
    # Ingestor + Transcriber as in build_pipeline, then:
    #   RollingSummarizerAgent(router, on_event=on_event, window_s=<config live.window_s>)
    # No Chapterizer, no Synthesizer.
```

`build_pipeline` (batch) is unchanged.

## FastAPI Adapter (`api/main.py`)

`create_app` gains a `live_pipeline_factory=build_live_pipeline` parameter
(mirroring the existing `pipeline_factory` seam for tests). `_start_job` /
`_run_job` select the factory by `options.live`:

```python
factory = live_pipeline_factory if options.live else pipeline_factory
pipeline = factory(on_event=on_event)
```

`JobOptions.live` flows in through the existing `CreateJobRequest.options`, so
`POST /api/jobs` needs no new field of its own. The client subscribes to the
existing `GET /api/jobs/{id}/events` SSE, renders `type="summary"` events as they
arrive, and reads the final report from `GET /api/jobs/{id}` at the end. The
upload route gets a `live: bool = Form(False)` for parity.

## Config

`config/models.yaml` additions:

```yaml
tasks:
  rolling:
    cheap:    ["ollama/llama3.1:8b", "openai/gpt-4o-mini"]
    balanced: ["anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"]
    best:     ["anthropic/claude-sonnet-5"]

live:
  window_s: 60          # content-seconds per rolling window
```

## Error Handling

- **A window's model call fails** (`RouterError` after retry/fallback): that
  window's delta/fold is skipped, a `type="summary"` event is still emitted with
  the unchanged prior digest and a `delta` noting the gap, and rolling
  continues. One bad window never aborts the stream.
- **Final consolidation fails**: the report `summary` falls back to the last
  running digest (never empty when at least one window closed).
- **Empty transcript** (no segments): no windows close; the report ships with an
  empty summary and the job completes (not failed).
- **Essential upstream failure** (ingest/transcribe): same as batch — the job
  fails with the stage name via the existing `PipelineError` path.

## Testing

Network-free via `FakeProvider` + an in-memory feed; the summarizer takes its
router and `on_event` by injection.

- **Feed tests:** `WindowedTranscriptFeed` yields all segments in order and
  exposes the transcript language.
- **Windowing tests:** segments spanning `> window_s` close a window at the right
  boundary; a short tail flushes as a final partial window.
- **Rolling tests** (capturing emitted events via a list-appending `on_event`):
  - N full windows produce N `type="summary"` events in order with correct
    `window_index` / `window_start_s` / `window_end_s`;
  - the running digest folds cumulatively (each event's `running_summary`
    reflects prior windows);
  - a window's `RouterError` emits a gap event and continues;
  - final report `summary` is the consolidated digest;
  - empty transcript → zero events, completed job, empty summary.
- **Agent test:** `RollingSummarizerAgent` requires a transcript, sets
  `ctx.report`, and forwards `on_event`.
- **Pipeline/adapter test:** `build_live_pipeline` lists `[Ingestor,
  Transcriber, RollingSummarizerAgent]`; the FastAPI job selects the live
  factory when `options.live` is true (asserted with an injected fake live
  factory, no network).

## Dependencies

None. Live mode reuses the existing providers, router, tracing, and SSE
infrastructure.

## Roadmap (remaining phases)

- **True live ingestion:** a `LiveSourceFeed` implementing `SegmentFeed` by
  transcribing a yt-dlp HLS pull incrementally — slots into this phase's seam.
- **Live SPA view:** render `type="summary"` events as a rolling timeline.
- **Phase 2e — Deployment re-fit** and the **evaluation re-wire** remain, along
  with merging the phase-3 (visual) and phase-4 (fact-checker) sibling branches.
