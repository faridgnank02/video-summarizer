# Live-Stream Rolling Summaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit rolling per-window summaries (delta + cumulative digest) over the existing SSE event channel as a transcript is replayed through a feed, then a final consolidated report — a live-mode pipeline variant selected by a `live` flag.

**Architecture:** A `SegmentFeed` abstraction (finite `WindowedTranscriptFeed` now) drives a `RollingSummarizer` service that windows by content-time, summarizes each window into a delta, folds it into a running digest, emits a `type="summary"` `StageEvent`, and consolidates a final `AnalysisReport`. A `RollingSummarizerAgent` runs it as the final agent of `build_live_pipeline` (Ingestor → Transcriber → RollingSummarizerAgent); FastAPI selects the live factory when `JobOptions.live` is set. No new endpoint, no new dependency.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest + pytest-asyncio (`asyncio_mode = auto`), FastAPI/SSE.

## Global Constraints

- Model IDs live ONLY in `config/models.yaml`, never in code.
- Default test suite runs with NO network and NO Whisper; use `FakeProvider` + in-memory feeds/fakes. Slow tests marked `@pytest.mark.slow`.
- Backward compatibility: `StageEvent.data` and `JobOptions.live` are additive with defaults; existing events/options must be unaffected.
- Every model call goes through `Router.complete(task="rolling", ...)` so it is traced; stages are `rolling.delta`, `rolling.fold`, `rolling.consolidate`.
- Windowing is by CONTENT time (segment `start_s`/`end_s`), never wall-clock. Frozen default: `window_s=60`.
- The live pipeline is `[Ingestor, Transcriber, RollingSummarizerAgent]` — NO Chapterizer, NO Synthesizer. `build_pipeline` (batch) is unchanged.
- Rolling event shape: `StageEvent(stage="live", type="summary", message=<running_summary>, data=RollingSummary(...).model_dump())`.
- One bad window (`RouterError`) never aborts the stream; it emits a gap event with the unchanged prior digest and continues.

---

### Task 1: Schemas — StageEvent.data, JobOptions.live, RollingSummary

**Files:**
- Modify: `src/video_intelligence/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: existing `StageEvent`, `JobOptions`, `BaseModel`.
- Produces:
  - `StageEvent.data: dict | None = None`
  - `JobOptions.live: bool = False`
  - `RollingSummary(window_index: int, window_start_s: float, window_end_s: float, delta: str, running_summary: str)`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas.py`:

```python
from src.video_intelligence.schemas import JobOptions, RollingSummary, StageEvent


def test_stage_event_data_defaults_none_and_roundtrips():
    assert StageEvent(stage="live", type="summary").data is None
    ev = StageEvent(stage="live", type="summary", message="digest",
                    data={"window_index": 0})
    assert ev.model_dump()["data"] == {"window_index": 0}


def test_job_options_live_defaults_false():
    assert JobOptions().live is False


def test_rolling_summary_shape():
    rs = RollingSummary(window_index=1, window_start_s=0.0, window_end_s=60.0,
                        delta="new bit", running_summary="so far")
    assert rs.window_index == 1 and rs.running_summary == "so far"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -k "stage_event_data or live_defaults or rolling_summary" -v`
Expected: FAIL with `ImportError: cannot import name 'RollingSummary'`.

- [ ] **Step 3: Write minimal implementation**

In `src/video_intelligence/schemas.py`, add `data` to `StageEvent`:

```python
class StageEvent(BaseModel):
    stage: str
    type: str  # started | progress | completed | failed | degraded | summary
    message: str | None = None
    data: dict | None = None
```

Add `live` to `JobOptions`:

```python
    live: bool = False
```

Add the payload model (near `StageEvent`):

```python
class RollingSummary(BaseModel):
    window_index: int
    window_start_s: float
    window_end_s: float
    delta: str
    running_summary: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/schemas.py tests/test_schemas.py
git commit -m "feat: add StageEvent.data, JobOptions.live, RollingSummary schema"
```

---

### Task 2: SegmentFeed abstraction + WindowedTranscriptFeed

**Files:**
- Create: `src/video_intelligence/live/__init__.py` (empty)
- Create: `src/video_intelligence/live/feed.py`
- Test: `tests/test_live_feed.py`

**Interfaces:**
- Consumes: `Transcript`, `TranscriptSegment` from schemas.
- Produces:
  - `class SegmentFeed(ABC)`: attribute `language: str`; `def segments(self) -> AsyncIterator[TranscriptSegment]` (abstract).
  - `class WindowedTranscriptFeed(SegmentFeed)`: `__init__(self, transcript: Transcript)`; async-generator `segments()` yielding the transcript's segments in order; `.language` = transcript language.

- [ ] **Step 1: Write the failing test**

Create `tests/test_live_feed.py`:

```python
from src.video_intelligence.live.feed import WindowedTranscriptFeed
from src.video_intelligence.schemas import Transcript, TranscriptOrigin, TranscriptSegment


def make_transcript(n):
    segs = [TranscriptSegment(start_s=i * 5.0, end_s=(i + 1) * 5.0, text=f"seg{i}")
            for i in range(n)]
    return Transcript(segments=segs, language="en", origin=TranscriptOrigin.CAPTIONS)


async def test_feed_yields_all_segments_in_order():
    feed = WindowedTranscriptFeed(make_transcript(3))
    out = [seg async for seg in feed.segments()]
    assert [s.text for s in out] == ["seg0", "seg1", "seg2"]


def test_feed_exposes_language():
    assert WindowedTranscriptFeed(make_transcript(1)).language == "en"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live_feed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.video_intelligence.live'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/video_intelligence/live/__init__.py` (empty file).

Create `src/video_intelligence/live/feed.py`:

```python
"""Time-ordered segment feeds for live rolling summarization."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..schemas import Transcript, TranscriptSegment


class SegmentFeed(ABC):
    language: str

    @abstractmethod
    def segments(self) -> AsyncIterator[TranscriptSegment]:
        """Yield transcript segments in chronological order."""
        ...


class WindowedTranscriptFeed(SegmentFeed):
    def __init__(self, transcript: Transcript):
        self.language = transcript.language
        self._transcript = transcript

    async def segments(self) -> AsyncIterator[TranscriptSegment]:
        for seg in self._transcript.segments:
            yield seg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_live_feed.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/live/__init__.py src/video_intelligence/live/feed.py tests/test_live_feed.py
git commit -m "feat: add SegmentFeed abstraction and WindowedTranscriptFeed"
```

---

### Task 3: RollingSummarizer service

**Files:**
- Create: `src/video_intelligence/live/summarizer.py`
- Modify: `config/models.yaml` (add `rolling` task + `live` block)
- Test: `tests/test_rolling_summarizer.py`

**Interfaces:**
- Consumes: `Router` + `RouterError` (`models/router.py`), schemas `AnalysisReport`, `RollingSummary`, `StageEvent`, `QualityPreference`, `Transcript`, `TranscriptOrigin`; `SegmentFeed` (Task 2); `transcript_lines` (`agents/prompting.py`).
- Produces:
  - `class RollingResult(BaseModel)`: `summary: str`
  - `class RollingSummarizer.__init__(self, router: Router, window_s: int = 60, quality: QualityPreference = QualityPreference.BALANCED, on_event=None)`
  - `async def run(self, feed: SegmentFeed, trace_id: str) -> AnalysisReport`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rolling_summarizer.py`:

```python
import pytest

from src.video_intelligence.live.feed import WindowedTranscriptFeed
from src.video_intelligence.live.summarizer import RollingResult, RollingSummarizer
from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    QualityPreference, StageEvent, Transcript, TranscriptOrigin, TranscriptSegment,
)
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"rolling": {"balanced": ["fake/m"]}}}
BAL = QualityPreference.BALANCED


def make(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    events: list[StageEvent] = []

    async def on_event(ev):
        events.append(ev)

    summ = RollingSummarizer(router, window_s=10, quality=BAL, on_event=on_event)
    return summ, fake, events


def feed_of(spans):
    # spans: list of (start_s, end_s)
    segs = [TranscriptSegment(start_s=s, end_s=e, text=f"t{i}")
            for i, (s, e) in enumerate(spans)]
    return WindowedTranscriptFeed(
        Transcript(segments=segs, language="en", origin=TranscriptOrigin.CAPTIONS))


async def test_two_windows_emit_ordered_summary_events(tmp_path):
    summ, fake, events = make(tmp_path)
    # window 1 closes at seg spanning >=10s; window 2 is the flushed tail.
    # per window: delta call (+ fold call only when prior digest non-empty)
    fake.enqueue(RollingResult(summary="delta1"))          # window0 delta (digest empty -> adopts delta1)
    fake.enqueue(RollingResult(summary="delta2"))          # window1 delta
    fake.enqueue(RollingResult(summary="digest2"))         # window1 fold
    fake.enqueue(RollingResult(summary="final"))           # consolidate
    report = await summ.run(feed_of([(0, 5), (5, 12), (12, 15)]), "tr1")
    summaries = [e for e in events if e.type == "summary"]
    assert [e.data["window_index"] for e in summaries] == [0, 1]
    assert summaries[0].data["running_summary"] == "delta1"
    assert summaries[1].data["running_summary"] == "digest2"
    assert summaries[1].data["window_start_s"] == 12.0
    assert report.summary == "final"


async def test_window_bounds_reflect_segments(tmp_path):
    summ, fake, events = make(tmp_path)
    fake.enqueue(RollingResult(summary="d0"))
    fake.enqueue(RollingResult(summary="final"))
    await summ.run(feed_of([(0, 4), (4, 11)]), "tr1")
    ev = [e for e in events if e.type == "summary"][0]
    assert ev.data["window_start_s"] == 0.0 and ev.data["window_end_s"] == 11.0


async def test_router_error_emits_gap_and_continues(tmp_path):
    summ, fake, events = make(tmp_path)
    fake.enqueue(ProviderError("boom"))    # window0 delta fails (initial)
    fake.enqueue(ProviderError("boom"))    # window0 delta retry -> RouterError
    report = await summ.run(feed_of([(0, 11)]), "tr1")
    summaries = [e for e in events if e.type == "summary"]
    assert len(summaries) == 1
    assert summaries[0].data["delta"] == "(summary unavailable for this window)"
    assert summaries[0].data["running_summary"] == ""   # digest unchanged (was empty)
    # digest stayed empty, so consolidation is skipped and summary is empty
    assert report.summary == ""
    assert len(fake.calls) == 2   # only the two failed delta attempts


async def test_empty_transcript_no_events_empty_summary(tmp_path):
    summ, fake, events = make(tmp_path)
    report = await summ.run(feed_of([]), "tr1")
    assert [e for e in events if e.type == "summary"] == []
    assert report.summary == ""
    assert fake.calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rolling_summarizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.video_intelligence.live.summarizer'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/video_intelligence/live/summarizer.py`:

```python
"""Rolling window summarization: per-window delta + cumulative digest + final report."""
from __future__ import annotations

from pydantic import BaseModel

from ..models.router import Router, RouterError
from ..schemas import (
    AnalysisReport, QualityPreference, RollingSummary, StageEvent, Transcript,
    TranscriptOrigin, TranscriptSegment,
)
from ..agents.prompting import transcript_lines
from .feed import SegmentFeed


class RollingResult(BaseModel):
    summary: str


DELTA_PROMPT = """Summarize what is NEW in this portion of a live video transcript, in 1-2 sentences.
Return ONLY a JSON object: {"summary": "<string>"}
Write in language code: <<LANGUAGE>>.

TRANSCRIPT PORTION:
<<TEXT>>"""


FOLD_PROMPT = """You maintain a running digest of a live video. Merge the NEW UPDATE into the
CURRENT DIGEST, keeping it concise and in chronological order.
Return ONLY a JSON object: {"summary": "<updated running digest>"}
Write in language code: <<LANGUAGE>>.

CURRENT DIGEST:
<<DIGEST>>

NEW UPDATE:
<<DELTA>>"""


CONSOLIDATE_PROMPT = """Produce a final, well-structured summary of a video from its running digest.
Return ONLY a JSON object: {"summary": "<markdown, 120-300 words>"}
Write in language code: <<LANGUAGE>>.

RUNNING DIGEST:
<<DIGEST>>"""


class RollingSummarizer:
    def __init__(self, router: Router, window_s: int = 60,
                 quality: QualityPreference = QualityPreference.BALANCED,
                 on_event=None):
        self._router = router
        self._window_s = window_s
        self._quality = quality
        self._on_event = on_event
        self._language = "en"

    async def run(self, feed: SegmentFeed, trace_id: str) -> AnalysisReport:
        self._language = feed.language
        window: list[TranscriptSegment] = []
        digest = ""
        window_index = 0
        async for seg in feed.segments():
            window.append(seg)
            if window[-1].end_s - window[0].start_s >= self._window_s:
                digest = await self._close_window(window, window_index, digest, trace_id)
                window_index += 1
                window = []
        if window:
            digest = await self._close_window(window, window_index, digest, trace_id)
        summary = await self._consolidate(digest, trace_id) if digest else ""
        return AnalysisReport(summary=summary, language=self._language, trace_id=trace_id)

    async def _close_window(self, segs: list[TranscriptSegment], index: int,
                            prev_digest: str, trace_id: str) -> str:
        text = transcript_lines(Transcript(segments=segs, language=self._language,
                                           origin=TranscriptOrigin.CAPTIONS))
        try:
            delta = await self._complete(DELTA_PROMPT.replace("<<TEXT>>", text),
                                         "rolling.delta", trace_id)
            digest = delta if not prev_digest else await self._complete(
                FOLD_PROMPT.replace("<<DIGEST>>", prev_digest).replace("<<DELTA>>", delta),
                "rolling.fold", trace_id)
        except RouterError:
            delta = "(summary unavailable for this window)"
            digest = prev_digest
        await self._emit(RollingSummary(
            window_index=index, window_start_s=segs[0].start_s,
            window_end_s=segs[-1].end_s, delta=delta, running_summary=digest))
        return digest

    async def _consolidate(self, digest: str, trace_id: str) -> str:
        try:
            return await self._complete(CONSOLIDATE_PROMPT.replace("<<DIGEST>>", digest),
                                        "rolling.consolidate", trace_id)
        except RouterError:
            return digest

    async def _complete(self, prompt: str, stage: str, trace_id: str) -> str:
        result = await self._router.complete(
            task="rolling", quality=self._quality,
            prompt=prompt.replace("<<LANGUAGE>>", self._language),
            schema=RollingResult, trace_id=trace_id, stage=stage)
        return result.summary

    async def _emit(self, rs: RollingSummary) -> None:
        if self._on_event is not None:
            await self._on_event(StageEvent(stage="live", type="summary",
                                            message=rs.running_summary,
                                            data=rs.model_dump()))
```

Add to `config/models.yaml` under `tasks:`:

```yaml
  rolling:
    cheap:    ["ollama/llama3.1:8b", "openai/gpt-4o-mini"]
    balanced: ["anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"]
    best:     ["anthropic/claude-sonnet-5"]
```

Add to `config/models.yaml` (top level):

```yaml
live:
  window_s: 60          # content-seconds per rolling window
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rolling_summarizer.py tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/live/summarizer.py config/models.yaml tests/test_rolling_summarizer.py
git commit -m "feat: add RollingSummarizer with delta/fold/consolidate loop"
```

---

### Task 4: RollingSummarizerAgent + build_live_pipeline

**Files:**
- Create: `src/video_intelligence/agents/rolling.py`
- Modify: `src/video_intelligence/pipeline.py` (add `build_live_pipeline`, extract shared router builder)
- Test: `tests/test_rolling_agent.py`
- Test: `tests/test_pipeline.py` (extend — live pipeline shape)

**Interfaces:**
- Consumes: `Agent`, `Router`, `RollingSummarizer`, `WindowedTranscriptFeed`, `PipelineContext`, config loaders + provider classes (as in `build_pipeline`).
- Produces:
  - `class RollingSummarizerAgent(Agent)`: `name = "rolling_summarize"`, `essential = True`, `__init__(self, router: Router, on_event=None, window_s: int = 60)`, `async def run(self, ctx) -> ctx`.
  - `def build_live_pipeline(config_path=..., db_path=..., workdir=..., on_event=None) -> Pipeline`
  - `build_pipeline` and `build_live_pipeline` share a `_build_router(config, store)` helper.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rolling_agent.py`:

```python
import pytest

from src.video_intelligence.agents.rolling import RollingSummarizerAgent
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, Transcript, TranscriptOrigin,
    TranscriptSegment, VideoSource,
)
from src.video_intelligence.live.summarizer import RollingResult
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"rolling": {"balanced": ["fake/m"]}}}


def ctx_with_transcript():
    segs = [TranscriptSegment(start_s=i * 5.0, end_s=(i + 1) * 5.0, text=f"s{i}")
            for i in range(3)]
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/x"),
        options=JobOptions(live=True),
        transcript=Transcript(segments=segs, language="en", origin=TranscriptOrigin.CAPTIONS),
        trace_id="tr1")


def test_agent_is_essential():
    router = Router(CONFIG, {}, None)
    agent = RollingSummarizerAgent(router)
    assert agent.essential is True and agent.name == "rolling_summarize"


async def test_agent_sets_report_and_emits(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    events = []

    async def on_event(ev):
        events.append(ev)

    fake.enqueue(RollingResult(summary="d0"))     # single window delta
    fake.enqueue(RollingResult(summary="final"))  # consolidate
    agent = RollingSummarizerAgent(router, on_event=on_event, window_s=10)
    ctx = await agent.run(ctx_with_transcript())
    assert ctx.report.summary == "final"
    assert any(e.type == "summary" for e in events)


async def test_agent_requires_transcript(tmp_path):
    router = Router(CONFIG, {"fake": FakeProvider()}, TraceStore(tmp_path / "t.db"))
    ctx = ctx_with_transcript()
    ctx.transcript = None
    with pytest.raises(ValueError, match="requires a transcript"):
        await RollingSummarizerAgent(router).run(ctx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rolling_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.video_intelligence.agents.rolling'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/video_intelligence/agents/rolling.py`:

```python
"""Live rolling-summary agent: streams per-window summaries and produces the final report."""
from __future__ import annotations

from ..live.feed import WindowedTranscriptFeed
from ..live.summarizer import RollingSummarizer
from ..models.router import Router
from ..schemas import PipelineContext
from .base import Agent


class RollingSummarizerAgent(Agent):
    name = "rolling_summarize"
    essential = True

    def __init__(self, router: Router, on_event=None, window_s: int = 60):
        self._router = router
        self._on_event = on_event
        self._window_s = window_s

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.transcript is None:
            raise ValueError("rolling_summarize requires a transcript")
        feed = WindowedTranscriptFeed(ctx.transcript)
        summarizer = RollingSummarizer(self._router, window_s=self._window_s,
                                       quality=ctx.options.quality,
                                       on_event=self._on_event)
        ctx.report = await summarizer.run(feed, ctx.trace_id)
        return ctx
```

In `src/video_intelligence/pipeline.py`, extract a shared router builder and add `build_live_pipeline`. Replace the body of `build_pipeline` from the imports down to the `router = Router(...)` line with a call to a new helper, and add the live builder below it:

```python
def _build_router(config: dict, store: "TraceStore") -> "Router":
    from .models.providers.anthropic import AnthropicProvider
    from .models.providers.ollama import OllamaProvider
    from .models.providers.openai import OpenAIProvider
    from .models.router import Router
    providers = {
        "ollama": OllamaProvider(),
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
    }
    return Router(config, providers, store)


def build_live_pipeline(config_path: str = "config/models.yaml",
                        db_path: str = "data/traces.db",
                        workdir: str = "data/work",
                        on_event: EventCallback | None = None) -> Pipeline:
    """Wire the live pipeline: Ingestor -> Transcriber -> RollingSummarizerAgent."""
    from .agents.ingestor import Ingestor
    from .agents.rolling import RollingSummarizerAgent
    from .agents.transcriber import Transcriber
    from .models.router import load_model_config
    from .tracing import TraceStore

    config = load_model_config(config_path)
    store = TraceStore(db_path)
    router = _build_router(config, store)
    whisper_model = config.get("transcription", {}).get("whisper_model", "base")
    window_s = config.get("live", {}).get("window_s", 60)
    return Pipeline(
        [
            Ingestor(workdir=workdir),
            Transcriber(model_name=whisper_model),
            RollingSummarizerAgent(router, on_event=on_event, window_s=window_s),
        ],
        on_event=on_event,
    )
```

Then update `build_pipeline` to use `_build_router` instead of its inline provider/`Router` construction (replace the `providers = {...}` dict and `router = Router(config, providers, store)` lines with `router = _build_router(config, store)`; keep the rest — `load_model_config`, `TraceStore`, `whisper_model`, and the agent list — unchanged). Leave the `from .models.router import Router, load_model_config` import in `build_pipeline` as `load_model_config` only if `Router` is no longer referenced there.

- [ ] **Step 4: Extend the pipeline shape test**

Add to `tests/test_pipeline.py`:

```python
def test_build_live_pipeline_shape(tmp_path):
    from src.video_intelligence.pipeline import build_live_pipeline
    pipeline = build_live_pipeline(config_path="config/models.yaml",
                                   db_path=str(tmp_path / "t.db"),
                                   workdir=str(tmp_path / "work"))
    names = [type(a).__name__ for a in pipeline._agents]
    assert names == ["Ingestor", "Transcriber", "RollingSummarizerAgent"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rolling_agent.py tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/video_intelligence/agents/rolling.py src/video_intelligence/pipeline.py tests/test_rolling_agent.py tests/test_pipeline.py
git commit -m "feat: add RollingSummarizerAgent and build_live_pipeline"
```

---

### Task 5: FastAPI live-pipeline selection

**Files:**
- Modify: `src/api/main.py`
- Test: `tests/test_api.py` (extend)

**Interfaces:**
- Consumes: `build_live_pipeline`, existing `create_app` / `_run_job` / `_start_job`.
- Produces:
  - `create_app(pipeline_factory=build_pipeline, live_pipeline_factory=build_live_pipeline, ...)`.
  - `_run_job` selects `live_pipeline_factory` when `options.live` is true.
  - `upload_job` gains `live: bool = Form(False)` into `JobOptions`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
from src.video_intelligence.schemas import RollingSummary


class FakeLivePipeline:
    def __init__(self, on_event, report):
        self._on_event = on_event
        self._report = report

    async def run(self, source, options):
        await self._on_event(StageEvent(
            stage="live", type="summary", message="rolling",
            data=RollingSummary(window_index=0, window_start_s=0.0, window_end_s=10.0,
                                delta="d", running_summary="rolling").model_dump()))
        return self._report


def test_live_flag_selects_live_factory(tmp_path):
    live_report = AnalysisReport(summary="LIVE", language="en", trace_id="trL")

    def batch_factory(on_event=None):
        return FakePipeline(on_event, report=AnalysisReport(
            summary="BATCH", language="en", trace_id="trB"))

    def live_factory(on_event=None):
        return FakeLivePipeline(on_event, report=live_report)

    app = create_app(pipeline_factory=batch_factory,
                     live_pipeline_factory=live_factory,
                     db_path=tmp_path / "app.db",
                     trace_db=tmp_path / "traces.db",
                     upload_dir=tmp_path / "uploads")
    client = TestClient(app)
    job_id = client.post("/api/jobs", json={
        "url": "https://youtu.be/x", "options": {"live": True}}).json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["report"]["summary"] == "LIVE"


def test_default_flag_uses_batch_factory(tmp_path):
    def batch_factory(on_event=None):
        return FakePipeline(on_event, report=AnalysisReport(
            summary="BATCH", language="en", trace_id="trB"))

    def live_factory(on_event=None):
        raise AssertionError("live factory must not be used for a batch job")

    app = create_app(pipeline_factory=batch_factory, live_pipeline_factory=live_factory,
                     db_path=tmp_path / "app.db", trace_db=tmp_path / "traces.db",
                     upload_dir=tmp_path / "uploads")
    client = TestClient(app)
    job_id = client.post("/api/jobs", json={"url": "https://youtu.be/x"}).json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["report"]["summary"] == "BATCH"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -k "live_flag or batch_factory" -v`
Expected: FAIL — `create_app() got an unexpected keyword argument 'live_pipeline_factory'`.

- [ ] **Step 3: Write minimal implementation**

In `src/api/main.py`:

Add the import:

```python
from src.video_intelligence.pipeline import PipelineError, build_live_pipeline, build_pipeline
```

Change the `create_app` signature to accept the live factory:

```python
def create_app(pipeline_factory=build_pipeline,
               live_pipeline_factory=build_live_pipeline,
               db_path: str | Path = "data/app.db",
               trace_db: str | Path = "data/traces.db",
               upload_dir: str | Path = "data/uploads") -> FastAPI:
```

In `_run_job`, select the factory by `options.live` (replace the `pipeline = pipeline_factory(on_event=on_event)` line):

```python
        factory = live_pipeline_factory if options.live else pipeline_factory
        pipeline = factory(on_event=on_event)
```

In `upload_job`, add the form field and thread it into `JobOptions`:

```python
    async def upload_job(background: BackgroundTasks,
                         file: UploadFile = File(...),
                         language: str = Form("en"),
                         quality: str = Form("balanced"),
                         force_whisper: bool = Form(False),
                         live: bool = Form(False)) -> dict:
        ...
        options = JobOptions(language=language, quality=QualityPreference(quality),
                             force_whisper=force_whisper, live=live)
```

(The JSON `POST /api/jobs` route already carries `live` via `options: JobOptions` — no change there.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/main.py tests/test_api.py
git commit -m "feat: select live pipeline when JobOptions.live is set"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full default suite**

Run: `pytest`
Expected: PASS — the prior suite plus the new feed/summarizer/agent/api tests, all network-free.

- [ ] **Step 2: Confirm no batch-mode regression**

Run: `pytest tests/test_pipeline.py tests/test_api.py tests/test_synthesizer.py tests/test_chapterizer.py -v`
Expected: PASS — batch pipeline, synthesizer, and chapterizer behavior unchanged by the `_build_router` extraction and the live additions.

---

## Self-Review

**Spec coverage:**
- `SegmentFeed` + `WindowedTranscriptFeed` → Task 2. ✓
- `RollingSummarizer` (windowing, delta, fold, emit, consolidate) → Task 3. ✓
- Rolling event shape `StageEvent(stage="live", type="summary", data=RollingSummary(...))` → Task 1 (schema) + Task 3 (`_emit`). ✓
- `RollingSummarizerAgent` (essential, replaces Synthesizer; requires transcript) → Task 4. ✓
- `build_live_pipeline` = `[Ingestor, Transcriber, RollingSummarizerAgent]`, no Chapterizer/Synthesizer → Task 4. ✓
- FastAPI `live` selection (JSON via options, upload via form), no new endpoint → Task 5. ✓
- Schemas `StageEvent.data`, `JobOptions.live`, `RollingSummary` → Task 1. ✓
- Config `rolling` task + `live.window_s` → Task 3. ✓
- Windowing by content-time; frozen `window_s=60` default → Task 3 (`window[-1].end_s - window[0].start_s >= window_s`). ✓
- Error handling: per-window `RouterError` → gap event + continue; consolidation fallback to digest; empty transcript → no events, empty summary → Task 3 tests. ✓
- Trace stages `rolling.delta/fold/consolidate` → Task 3 (`_complete` stage arg). ✓
- No new dependency → confirmed (no requirements.txt change). ✓
- MCP/frontend/true-live/live-chapters deferred → not in plan (matches spec Out). ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete code. ✓

**Type consistency:** `RollingSummarizer(router, window_s, quality, on_event).run(feed, trace_id)`; `RollingResult(summary)`; `RollingSummary(window_index, window_start_s, window_end_s, delta, running_summary)`; `RollingSummarizerAgent(router, on_event, window_s)`; `build_live_pipeline(config_path, db_path, workdir, on_event)`; `create_app(pipeline_factory, live_pipeline_factory, ...)`; `StageEvent(stage, type, message, data)` — consistent across Tasks 1, 3, 4, 5. The digest-fold contract (empty prior digest → adopt delta; non-empty → fold call) is used identically in the summarizer and asserted in the Task 3 tests. ✓
