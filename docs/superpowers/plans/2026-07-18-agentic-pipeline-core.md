# Agentic Pipeline Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LED/Streamlit summarizer with a multi-agent pipeline (Ingestor → Transcriber → Chapterizer → Synthesizer) producing structured video reports, behind a FastAPI backend and React SPA, with cost-aware model routing and per-stage tracing.

**Architecture:** A standalone core library `src/video_intelligence/` (agents, router, providers, tracing, schemas — zero web dependencies), a thin FastAPI adapter in `src/api/`, and a React/Vite/TS/Tailwind SPA in `frontend/`. Hand-rolled asyncio orchestration; jobs run via FastAPI BackgroundTasks with SSE progress.

**Tech Stack:** Python 3.12, Pydantic v2, faster-whisper, yt-dlp, youtube-transcript-api ≥1.0, httpx (Ollama), openai ≥1.x, anthropic SDK, FastAPI + uvicorn, SQLite, React 18 + Vite + TypeScript + Tailwind v4, pytest + pytest-asyncio, vitest.

**Spec:** `docs/superpowers/specs/2026-07-18-agentic-pipeline-core-design.md`

## Global Constraints

- Python 3.12; all core-library I/O is `async` (blocking work wrapped in `asyncio.to_thread`).
- `src/video_intelligence/` MUST NOT import FastAPI or anything web-related.
- Model IDs appear ONLY in `config/models.yaml`, never in code.
- Pydantic v2 APIs only (`model_validate_json`, `model_dump_json`, `model_copy`).
- `ffmpeg` must be on PATH (document in README; already required by old project via yt-dlp).
- Run tests from the repo root with plain `pytest` (pytest.ini sets `pythonpath = .`, `asyncio_mode = auto`, and excludes `slow` by default).
- Commit after every task with a conventional-commit message.
- Prompts that embed JSON examples use `<<PLACEHOLDER>>` + `.replace()`, never `str.format` (braces in JSON break format strings).

---

### Task 1: Retire legacy code + new dependency baseline

**Files:**
- Delete: `src/ui/`, `src/models/`, `src/data/`, `src/monitoring/`, `src/training/`, `src/api/summarization.py`, `tests/test_api.py`, `tests/test_architecture.py`, `tests/test_functionality.py`, `tests/test_integration.py`, `tests/test_monitoring.py`, `scripts/setup_led.py`, `scripts/launch.py`, `scripts/launch_api.py`, `scripts/test_models.py`, `config/model_config.yaml`, `metrics.db`
- Keep: `src/evaluation/` (unwired, per spec), `src/__init__.py`, `src/api/__init__.py`, `config/app_config.yaml`, `scripts/install.py`, `scripts/setup.sh`
- Create: `requirements.txt` (rewrite), `requirements-eval.txt`, `pytest.ini`, `src/video_intelligence/__init__.py`, `src/video_intelligence/agents/__init__.py`, `src/video_intelligence/models/__init__.py`, `src/video_intelligence/models/providers/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable dependency baseline and empty package skeleton every later task builds inside.

- [ ] **Step 1: Delete legacy modules and tests**

Rationale: `src/ui` (Streamlit), `src/models` (LED + old OpenAI + manager) and `src/api/summarization.py` are retired by the spec. `src/data`, `src/monitoring`, `src/training` are only consumed by the deleted UI/manager code (verify with the grep below — expect matches only in deleted files). The five old test files all import deleted modules.

```bash
grep -rl "src.data\|src.monitoring\|src.training" src/ scripts/ tests/ | grep -v "src/data\|src/monitoring\|src/training"
# review output — should list only files already on the delete list above
git rm -r src/ui src/models src/data src/monitoring src/training
git rm src/api/summarization.py config/model_config.yaml metrics.db
git rm tests/test_api.py tests/test_architecture.py tests/test_functionality.py tests/test_integration.py tests/test_monitoring.py
git rm scripts/setup_led.py scripts/launch.py scripts/launch_api.py scripts/test_models.py
```

- [ ] **Step 2: Rewrite `requirements.txt`**

```
# Core
pydantic>=2.7
pyyaml>=6.0
python-dotenv>=1.0.0
httpx>=0.27

# Ingestion / transcription
yt-dlp>=2025.1.1
youtube-transcript-api>=1.0
faster-whisper>=1.0

# Cloud providers
openai>=1.40
anthropic>=0.34

# API
fastapi>=0.111
uvicorn[standard]>=0.30
python-multipart>=0.0.9

# Tests
pytest>=8.0
pytest-asyncio>=0.23
```

- [ ] **Step 3: Create `requirements-eval.txt`** (parked deps for the unwired `src/evaluation/`)

```
# Only needed to run src/evaluation (currently unwired — see spec roadmap)
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
spacy>=3.6.0
rouge-score>=0.1.2
nltk>=3.8.1
numpy>=1.24.0
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
asyncio_mode = auto
markers =
    slow: long-running tests (real whisper); run with -m slow
addopts = -m "not slow"
```

- [ ] **Step 5: Create the package skeleton**

```bash
mkdir -p src/video_intelligence/agents src/video_intelligence/models/providers
touch src/video_intelligence/__init__.py src/video_intelligence/agents/__init__.py \
      src/video_intelligence/models/__init__.py src/video_intelligence/models/providers/__init__.py \
      tests/__init__.py
```

- [ ] **Step 6: Install and verify**

```bash
pip install -r requirements.txt
pytest
```
Expected: `no tests ran` (exit code 5 is fine at this point).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: retire LED/Streamlit stack, add agentic-pipeline dependency baseline"
```

---

### Task 2: Core schemas

**Files:**
- Create: `src/video_intelligence/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by every later task): `SourceKind`, `TranscriptOrigin`, `QualityPreference` (StrEnums); `VideoSource(kind, url?, path?, title?, duration_s?, channel?)`; `TranscriptSegment(start_s, end_s, text)`; `Transcript(segments, language, origin)` with `.full_text` property; `Chapter(start_s, end_s, title, synopsis)`; `KeyQuote(timestamp_s, speaker?, text)`; `AnalysisReport(summary, chapters, key_quotes, action_items, language, trace_id, degraded_stages)`; `TraceSpan(stage, model_used, tokens_in, tokens_out, cost_usd, latency_ms, status, fallback_from?)`; `JobOptions(language="en", quality=BALANCED, force_whisper=False)`; `StageEvent(stage, type, message?)`; `PipelineContext(source, options, trace_id, audio_path?, transcript?, chapters?, report?, degraded_stages)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
from src.video_intelligence.schemas import (
    AnalysisReport, JobOptions, PipelineContext, QualityPreference,
    SourceKind, Transcript, TranscriptOrigin, TranscriptSegment, VideoSource,
)


def test_transcript_full_text_joins_segments():
    t = Transcript(
        segments=[
            TranscriptSegment(start_s=0.0, end_s=2.0, text="Hello"),
            TranscriptSegment(start_s=2.0, end_s=4.0, text="world"),
        ],
        language="en",
        origin=TranscriptOrigin.CAPTIONS,
    )
    assert t.full_text == "Hello world"


def test_job_options_defaults():
    opts = JobOptions()
    assert opts.language == "en"
    assert opts.quality == QualityPreference.BALANCED
    assert opts.force_whisper is False


def test_pipeline_context_generates_trace_id():
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/x"),
        options=JobOptions(),
    )
    assert len(ctx.trace_id) == 32
    assert ctx.transcript is None
    assert ctx.degraded_stages == []


def test_analysis_report_round_trips_json():
    report = AnalysisReport(summary="s", language="en", trace_id="abc")
    parsed = AnalysisReport.model_validate_json(report.model_dump_json())
    assert parsed.summary == "s"
    assert parsed.chapters == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.video_intelligence.schemas'`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/schemas.py
"""Shared data shapes for the video-intelligence pipeline."""
from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class SourceKind(StrEnum):
    YOUTUBE = "youtube"
    LOCAL_FILE = "local_file"


class TranscriptOrigin(StrEnum):
    CAPTIONS = "captions"
    WHISPER = "whisper"


class QualityPreference(StrEnum):
    CHEAP = "cheap"
    BALANCED = "balanced"
    BEST = "best"


class VideoSource(BaseModel):
    kind: SourceKind
    url: str | None = None
    path: str | None = None
    title: str | None = None
    duration_s: float | None = None
    channel: str | None = None


class TranscriptSegment(BaseModel):
    start_s: float
    end_s: float
    text: str


class Transcript(BaseModel):
    segments: list[TranscriptSegment]
    language: str
    origin: TranscriptOrigin

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


class Chapter(BaseModel):
    start_s: float
    end_s: float
    title: str
    synopsis: str


class KeyQuote(BaseModel):
    timestamp_s: float
    speaker: str | None = None
    text: str


class AnalysisReport(BaseModel):
    summary: str
    chapters: list[Chapter] = Field(default_factory=list)
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    language: str
    trace_id: str
    degraded_stages: list[str] = Field(default_factory=list)


class TraceSpan(BaseModel):
    stage: str
    model_used: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    status: str = "ok"  # ok | error
    fallback_from: str | None = None


class JobOptions(BaseModel):
    language: str = "en"
    quality: QualityPreference = QualityPreference.BALANCED
    force_whisper: bool = False


class StageEvent(BaseModel):
    stage: str
    type: str  # started | progress | completed | failed
    message: str | None = None


class PipelineContext(BaseModel):
    source: VideoSource
    options: JobOptions
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    audio_path: str | None = None
    transcript: Transcript | None = None
    chapters: list[Chapter] | None = None
    report: AnalysisReport | None = None
    degraded_stages: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/schemas.py tests/test_schemas.py
git commit -m "feat: add core pipeline schemas"
```

---

### Task 3: Trace store

**Files:**
- Create: `src/video_intelligence/tracing.py`
- Test: `tests/test_tracing.py`

**Interfaces:**
- Consumes: `TraceSpan` from `schemas.py`.
- Produces: `TraceStore(db_path)` with `.add_span(trace_id: str, span: TraceSpan) -> None`, `.spans(trace_id: str) -> list[TraceSpan]` (insertion order), `.total_cost(trace_id: str) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracing.py
from src.video_intelligence.schemas import TraceSpan
from src.video_intelligence.tracing import TraceStore


def test_add_and_read_spans_in_order(tmp_path):
    store = TraceStore(tmp_path / "traces.db")
    store.add_span("t1", TraceSpan(stage="transcribe", model_used="whisper-base"))
    store.add_span("t1", TraceSpan(stage="synthesize", model_used="anthropic/claude-sonnet", cost_usd=0.02))
    store.add_span("other", TraceSpan(stage="ingest", model_used="none"))

    spans = store.spans("t1")
    assert [s.stage for s in spans] == ["transcribe", "synthesize"]
    assert spans[1].cost_usd == 0.02


def test_total_cost_sums_only_that_trace(tmp_path):
    store = TraceStore(tmp_path / "traces.db")
    store.add_span("t1", TraceSpan(stage="a", model_used="m", cost_usd=0.01))
    store.add_span("t1", TraceSpan(stage="b", model_used="m", cost_usd=0.02))
    store.add_span("t2", TraceSpan(stage="a", model_used="m", cost_usd=5.0))
    assert store.total_cost("t1") == 0.03


def test_unknown_trace_is_empty(tmp_path):
    store = TraceStore(tmp_path / "traces.db")
    assert store.spans("nope") == []
    assert store.total_cost("nope") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.video_intelligence.tracing'`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/tracing.py
"""SQLite-backed per-stage trace spans (cost, latency, model choice)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .schemas import TraceSpan


class TraceStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    span_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id)")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def add_span(self, trace_id: str, span: TraceSpan) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO spans (trace_id, span_json) VALUES (?, ?)",
                (trace_id, span.model_dump_json()),
            )

    def spans(self, trace_id: str) -> list[TraceSpan]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT span_json FROM spans WHERE trace_id = ? ORDER BY id", (trace_id,)
            ).fetchall()
        return [TraceSpan.model_validate_json(row[0]) for row in rows]

    def total_cost(self, trace_id: str) -> float:
        return round(sum(s.cost_usd for s in self.spans(trace_id)), 6)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tracing.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/tracing.py tests/test_tracing.py
git commit -m "feat: add SQLite trace store for per-stage spans"
```

---

### Task 4: Provider interface + FakeProvider

**Files:**
- Create: `src/video_intelligence/models/providers/base.py`, `tests/fakes.py`
- Test: `tests/test_provider_base.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Usage(tokens_in, tokens_out)`; `ProviderError(Exception)`; abstract `Provider` with `name: str`, `async is_available() -> bool`, `async complete(model: str, prompt: str, schema: type[T]) -> tuple[T, Usage]` (T bound to `pydantic.BaseModel`); helper `parse_json_response(text: str, schema: type[T]) -> T`; test double `FakeProvider(name="fake", available=True)` with `.enqueue(item: BaseModel | Exception)` and `.calls: list[dict]` recording `{"model", "prompt", "schema"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_base.py
import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.base import ProviderError, Usage, parse_json_response
from tests.fakes import FakeProvider


class Answer(BaseModel):
    value: int


def test_parse_json_response_extracts_embedded_object():
    text = 'Sure! Here is the JSON:\n{"value": 42}\nHope that helps.'
    assert parse_json_response(text, Answer).value == 42


def test_parse_json_response_rejects_missing_json():
    with pytest.raises(ProviderError):
        parse_json_response("no json here", Answer)


def test_parse_json_response_rejects_wrong_shape():
    with pytest.raises(ProviderError):
        parse_json_response('{"other": 1}', Answer)


async def test_fake_provider_returns_queued_items_and_records_calls():
    fake = FakeProvider()
    fake.enqueue(Answer(value=1))
    parsed, usage = await fake.complete("some-model", "prompt text", Answer)
    assert parsed.value == 1
    assert isinstance(usage, Usage)
    assert fake.calls[0]["model"] == "some-model"


async def test_fake_provider_raises_queued_exception():
    fake = FakeProvider()
    fake.enqueue(ProviderError("boom"))
    with pytest.raises(ProviderError):
        await fake.complete("m", "p", Answer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider_base.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/models/providers/base.py
"""Provider interface: one thin async client per model vendor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Usage(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0


class ProviderError(Exception):
    """Any provider failure: network, API error, unparseable output."""


class Provider(ABC):
    name: str

    @abstractmethod
    async def is_available(self) -> bool: ...

    @abstractmethod
    async def complete(self, model: str, prompt: str, schema: type[T]) -> tuple[T, Usage]: ...


def parse_json_response(text: str, schema: type[T]) -> T:
    """Extract the first JSON object from a model response and validate it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ProviderError(f"no JSON object in response: {text[:200]!r}")
    try:
        return schema.model_validate_json(text[start : end + 1])
    except ValueError as e:
        raise ProviderError(f"schema validation failed: {e}") from e
```

```python
# tests/fakes.py
"""Test doubles shared across the test suite."""
from __future__ import annotations

from pydantic import BaseModel

from src.video_intelligence.models.providers.base import Provider, Usage


class FakeProvider(Provider):
    def __init__(self, name: str = "fake", available: bool = True):
        self.name = name
        self._available = available
        self._queue: list[BaseModel | Exception] = []
        self.calls: list[dict] = []

    def enqueue(self, item: BaseModel | Exception) -> None:
        self._queue.append(item)

    async def is_available(self) -> bool:
        return self._available

    async def complete(self, model, prompt, schema):
        self.calls.append({"model": model, "prompt": prompt, "schema": schema})
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, Usage(tokens_in=100, tokens_out=50)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider_base.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/models/providers/base.py tests/fakes.py tests/test_provider_base.py
git commit -m "feat: add provider interface, JSON parsing helper, and FakeProvider"
```

---

### Task 5: Ollama provider

**Files:**
- Create: `src/video_intelligence/models/providers/ollama.py`
- Test: `tests/test_ollama_provider.py`

**Interfaces:**
- Consumes: `Provider`, `Usage`, `ProviderError`, `parse_json_response` from Task 4.
- Produces: `OllamaProvider(base_url="http://localhost:11434", transport=None)` — `transport` is an optional `httpx.AsyncBaseTransport` for tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ollama_provider.py
import httpx
import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.providers.ollama import OllamaProvider


class Answer(BaseModel):
    value: int


def make_transport(handler):
    return httpx.MockTransport(handler)


async def test_complete_parses_chat_response():
    def handler(request):
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": '{"value": 7}'},
            "prompt_eval_count": 12,
            "eval_count": 5,
        })

    provider = OllamaProvider(transport=make_transport(handler))
    parsed, usage = await provider.complete("llama3.1:8b", "hi", Answer)
    assert parsed.value == 7
    assert usage.tokens_in == 12
    assert usage.tokens_out == 5


async def test_http_error_becomes_provider_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    provider = OllamaProvider(transport=make_transport(handler))
    with pytest.raises(ProviderError):
        await provider.complete("llama3.1:8b", "hi", Answer)


async def test_is_available_true_when_tags_responds():
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    assert await OllamaProvider(transport=make_transport(handler)).is_available() is True


async def test_is_available_false_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert await OllamaProvider(transport=make_transport(handler)).is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ollama_provider.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/models/providers/ollama.py
from __future__ import annotations

import httpx

from .base import Provider, ProviderError, Usage, parse_json_response


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434",
                 transport: httpx.AsyncBaseTransport | None = None):
        self._base_url = base_url
        self._transport = transport

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def is_available(self) -> bool:
        try:
            async with self._client(timeout=2.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def complete(self, model, prompt, schema):
        try:
            async with self._client(timeout=300.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                })
                resp.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"ollama request failed: {e}") from e
        data = resp.json()
        usage = Usage(tokens_in=data.get("prompt_eval_count", 0),
                      tokens_out=data.get("eval_count", 0))
        return parse_json_response(data["message"]["content"], schema), usage
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ollama_provider.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/models/providers/ollama.py tests/test_ollama_provider.py
git commit -m "feat: add Ollama provider"
```

---

### Task 6: OpenAI + Anthropic providers

**Files:**
- Create: `src/video_intelligence/models/providers/openai.py`, `src/video_intelligence/models/providers/anthropic.py`
- Test: `tests/test_cloud_providers.py`

**Interfaces:**
- Consumes: Task 4 base.
- Produces: `OpenAIProvider(client=None)` (injectable `AsyncOpenAI`-shaped client; `is_available()` = `OPENAI_API_KEY` env set); `AnthropicProvider(client=None)` (injectable `AsyncAnthropic`-shaped client; `is_available()` = `ANTHROPIC_API_KEY` env set).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cloud_providers.py
import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.anthropic import AnthropicProvider
from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.providers.openai import OpenAIProvider


class Answer(BaseModel):
    value: int


# --- minimal stubs shaped like each vendor SDK's response objects ---

class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class StubOpenAIClient:
    def __init__(self, content='{"value": 3}', error: Exception | None = None):
        outer = self
        class _Completions:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                if error:
                    raise error
                return _Obj(
                    choices=[_Obj(message=_Obj(content=content))],
                    usage=_Obj(prompt_tokens=10, completion_tokens=4),
                )
        self.chat = _Obj(completions=_Completions())


class StubAnthropicClient:
    def __init__(self, content='{"value": 9}', error: Exception | None = None):
        outer = self
        class _Messages:
            async def create(self, **kwargs):
                outer.kwargs = kwargs
                if error:
                    raise error
                return _Obj(
                    content=[_Obj(text=content)],
                    usage=_Obj(input_tokens=20, output_tokens=6),
                )
        self.messages = _Messages()


async def test_openai_complete_parses_and_reports_usage():
    stub = StubOpenAIClient()
    parsed, usage = await OpenAIProvider(client=stub).complete("gpt-4o-mini", "p", Answer)
    assert parsed.value == 3
    assert (usage.tokens_in, usage.tokens_out) == (10, 4)
    assert stub.kwargs["model"] == "gpt-4o-mini"


async def test_openai_error_becomes_provider_error():
    stub = StubOpenAIClient(error=RuntimeError("api down"))
    with pytest.raises(ProviderError):
        await OpenAIProvider(client=stub).complete("gpt-4o-mini", "p", Answer)


async def test_anthropic_complete_parses_and_reports_usage():
    stub = StubAnthropicClient()
    parsed, usage = await AnthropicProvider(client=stub).complete("claude-sonnet", "p", Answer)
    assert parsed.value == 9
    assert (usage.tokens_in, usage.tokens_out) == (20, 6)


async def test_availability_follows_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert await OpenAIProvider().is_available() is False
    assert await AnthropicProvider().is_available() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    assert await OpenAIProvider().is_available() is True
    assert await AnthropicProvider().is_available() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cloud_providers.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementations**

```python
# src/video_intelligence/models/providers/openai.py
from __future__ import annotations

import os

from .base import Provider, ProviderError, Usage, parse_json_response


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        return self._client

    async def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    async def complete(self, model, prompt, schema):
        try:
            resp = await self._get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise ProviderError(f"openai request failed: {e}") from e
        usage = Usage(tokens_in=resp.usage.prompt_tokens, tokens_out=resp.usage.completion_tokens)
        return parse_json_response(resp.choices[0].message.content, schema), usage
```

```python
# src/video_intelligence/models/providers/anthropic.py
from __future__ import annotations

import os

from .base import Provider, ProviderError, Usage, parse_json_response


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, client=None):
        self._client = client

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        return self._client

    async def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    async def complete(self, model, prompt, schema):
        try:
            resp = await self._get_client().messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            raise ProviderError(f"anthropic request failed: {e}") from e
        usage = Usage(tokens_in=resp.usage.input_tokens, tokens_out=resp.usage.output_tokens)
        return parse_json_response(resp.content[0].text, schema), usage
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cloud_providers.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/models/providers/openai.py src/video_intelligence/models/providers/anthropic.py tests/test_cloud_providers.py
git commit -m "feat: add OpenAI and Anthropic providers"
```

---

### Task 7: Model router + `config/models.yaml`

**Files:**
- Create: `src/video_intelligence/models/router.py`, `config/models.yaml`
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: providers (Tasks 4–6), `TraceStore` (Task 3), `QualityPreference`, `TraceSpan` (Task 2).
- Produces: `RouterError(Exception)`; `load_model_config(path) -> dict`; `Router(config: dict, providers: dict[str, Provider], store: TraceStore)` with `async complete(*, task: str, quality: QualityPreference, prompt: str, schema: type[T], trace_id: str, stage: str) -> T`. Candidate strings are `"<provider>/<model>"` split on the FIRST slash (Ollama models contain colons, e.g. `ollama/llama3.1:8b`). Config keys: `tasks.<task>.<quality>` → ordered candidate list; `pricing.<candidate>.{input_per_mtok, output_per_mtok}`; `transcription.whisper_model`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_router.py
import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.router import Router, RouterError
from src.video_intelligence.schemas import QualityPreference
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider


class Answer(BaseModel):
    value: int


CONFIG = {
    "tasks": {
        "chaptering": {
            "cheap": ["ollama/llama3.1:8b", "openai/gpt-4o-mini"],
        }
    },
    "pricing": {
        "openai/gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
    },
}


def make(store, ollama=None, openai=None):
    providers = {"ollama": ollama or FakeProvider("ollama"),
                 "openai": openai or FakeProvider("openai")}
    return Router(CONFIG, providers, store), providers


async def test_uses_first_available_candidate_and_records_span(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store)
    providers["ollama"].enqueue(Answer(value=1))

    result = await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                                   prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    assert result.value == 1
    span = store.spans("tr")[0]
    assert span.model_used == "ollama/llama3.1:8b"
    assert span.fallback_from is None
    assert span.cost_usd == 0.0  # no pricing entry for local model
    # model passed to the provider keeps its colon
    assert providers["ollama"].calls[0]["model"] == "llama3.1:8b"


async def test_falls_back_when_first_unavailable_and_prices_usage(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store, ollama=FakeProvider("ollama", available=False))
    providers["openai"].enqueue(Answer(value=2))

    result = await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                                   prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    assert result.value == 2
    span = store.spans("tr")[0]
    assert span.model_used == "openai/gpt-4o-mini"
    assert span.fallback_from == "ollama/llama3.1:8b"
    # FakeProvider reports 100 in / 50 out tokens
    assert span.cost_usd == pytest.approx(100 / 1e6 * 0.15 + 50 / 1e6 * 0.60)


async def test_retries_once_then_falls_back_on_errors(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store)
    providers["ollama"].enqueue(ProviderError("flaky"))
    providers["ollama"].enqueue(ProviderError("flaky again"))
    providers["openai"].enqueue(Answer(value=3))

    result = await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                                   prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    assert result.value == 3
    assert len(providers["ollama"].calls) == 2  # initial + one retry


async def test_all_candidates_failing_raises_and_records_error_span(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    router, providers = make(store, ollama=FakeProvider("ollama", available=False),
                             openai=FakeProvider("openai", available=False))
    with pytest.raises(RouterError):
        await router.complete(task="chaptering", quality=QualityPreference.CHEAP,
                              prompt="p", schema=Answer, trace_id="tr", stage="chapterize")
    span = store.spans("tr")[0]
    assert span.status == "error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/models/router.py
"""Task → model routing with availability checks, retry, fallback, tracing."""
from __future__ import annotations

import time
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from ..schemas import QualityPreference, TraceSpan
from ..tracing import TraceStore
from .providers.base import Provider, ProviderError, Usage

T = TypeVar("T", bound=BaseModel)


class RouterError(Exception):
    pass


def load_model_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Router:
    def __init__(self, config: dict, providers: dict[str, Provider], store: TraceStore):
        self._config = config
        self._providers = providers
        self._store = store

    def candidates(self, task: str, quality: QualityPreference) -> list[str]:
        try:
            return self._config["tasks"][task][quality.value]
        except KeyError as e:
            raise RouterError(f"no candidates configured for task={task} quality={quality}") from e

    def _cost(self, candidate: str, usage: Usage) -> float:
        pricing = self._config.get("pricing", {}).get(candidate)
        if not pricing:
            return 0.0
        return (usage.tokens_in / 1e6 * pricing["input_per_mtok"]
                + usage.tokens_out / 1e6 * pricing["output_per_mtok"])

    async def complete(self, *, task: str, quality: QualityPreference, prompt: str,
                       schema: type[T], trace_id: str, stage: str) -> T:
        fallback_from: str | None = None
        errors: list[str] = []
        for candidate in self.candidates(task, quality):
            provider_name, model = candidate.split("/", 1)
            provider = self._providers.get(provider_name)
            if provider is None or not await provider.is_available():
                errors.append(f"{candidate}: unavailable")
                fallback_from = candidate
                continue
            for _attempt in range(2):  # initial call + one retry
                start = time.monotonic()
                try:
                    parsed, usage = await provider.complete(model, prompt, schema)
                except ProviderError as e:
                    errors.append(f"{candidate}: {e}")
                    continue
                self._store.add_span(trace_id, TraceSpan(
                    stage=stage,
                    model_used=candidate,
                    tokens_in=usage.tokens_in,
                    tokens_out=usage.tokens_out,
                    cost_usd=self._cost(candidate, usage),
                    latency_ms=int((time.monotonic() - start) * 1000),
                    status="ok",
                    fallback_from=fallback_from,
                ))
                return parsed
            fallback_from = candidate
        self._store.add_span(trace_id, TraceSpan(stage=stage, model_used="none", status="error"))
        raise RouterError(f"all candidates failed for task={task}: {'; '.join(errors)}")
```

- [ ] **Step 4: Create `config/models.yaml`**

```yaml
# Model routing config. Model IDs live ONLY here — never in code.
# Candidate format: "<provider>/<model>", tried in order per quality tier.

transcription:
  whisper_model: base        # faster-whisper model name

tasks:
  chaptering:
    cheap:    ["ollama/llama3.1:8b", "openai/gpt-4o-mini"]
    balanced: ["ollama/llama3.1:8b", "anthropic/claude-haiku-4-5-20251001"]
    best:     ["anthropic/claude-sonnet-5"]
  synthesis:
    cheap:    ["openai/gpt-4o-mini"]
    balanced: ["anthropic/claude-sonnet-5", "openai/gpt-4o"]
    best:     ["anthropic/claude-opus-4-8", "openai/gpt-4o"]

pricing:  # USD per million tokens; local models omitted (cost 0)
  openai/gpt-4o-mini: {input_per_mtok: 0.15, output_per_mtok: 0.60}
  openai/gpt-4o: {input_per_mtok: 2.50, output_per_mtok: 10.00}
  anthropic/claude-haiku-4-5-20251001: {input_per_mtok: 1.00, output_per_mtok: 5.00}
  anthropic/claude-sonnet-5: {input_per_mtok: 3.00, output_per_mtok: 15.00}
  anthropic/claude-opus-4-8: {input_per_mtok: 5.00, output_per_mtok: 25.00}
```

Note: verify current model IDs and prices against provider docs at implementation time; the yaml is the only place they live.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_router.py -v`
Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/video_intelligence/models/router.py config/models.yaml tests/test_router.py
git commit -m "feat: add cost-aware model router with fallback and tracing"
```

---

### Task 8: Agent base + prompt helpers

**Files:**
- Create: `src/video_intelligence/agents/base.py`, `src/video_intelligence/agents/prompting.py`
- Test: `tests/test_prompting.py`

**Interfaces:**
- Consumes: `PipelineContext`, `Transcript` (Task 2).
- Produces: abstract `Agent` with `name: str`, `essential: bool = True`, `async run(ctx: PipelineContext) -> PipelineContext`; `transcript_lines(transcript: Transcript, block_s: float = 15.0) -> str` (lines like `[M:SS] text`, segments merged into ~15s blocks); `chunk_text(text: str, max_chars: int) -> list[str]` (splits on line boundaries, every chunk ≤ max_chars, order preserved).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompting.py
from src.video_intelligence.agents.prompting import chunk_text, transcript_lines
from src.video_intelligence.schemas import Transcript, TranscriptOrigin, TranscriptSegment


def make_transcript(n_segments: int, seg_len_s: float = 5.0) -> Transcript:
    segs = [
        TranscriptSegment(start_s=i * seg_len_s, end_s=(i + 1) * seg_len_s, text=f"seg{i}")
        for i in range(n_segments)
    ]
    return Transcript(segments=segs, language="en", origin=TranscriptOrigin.WHISPER)


def test_transcript_lines_merges_into_blocks():
    text = transcript_lines(make_transcript(6))  # 6 x 5s = 30s -> 2 blocks of ~15s
    lines = text.splitlines()
    assert lines[0].startswith("[0:00] ")
    assert "seg0" in lines[0] and "seg2" in lines[0]
    assert lines[1].startswith("[0:15] ")
    assert len(lines) == 2


def test_transcript_lines_formats_hours():
    t = Transcript(
        segments=[TranscriptSegment(start_s=3661.0, end_s=3665.0, text="late")],
        language="en", origin=TranscriptOrigin.WHISPER,
    )
    assert transcript_lines(t).startswith("[1:01:01] ")


def test_chunk_text_respects_line_boundaries():
    text = "\n".join(f"line {i} " + "x" * 50 for i in range(10))
    chunks = chunk_text(text, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)
    assert "\n".join(chunks).replace("\n\n", "\n") == text  # nothing lost
    assert len(chunks) > 1


def test_chunk_text_single_chunk_when_small():
    assert chunk_text("short", max_chars=100) == ["short"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompting.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/agents/base.py
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import PipelineContext


class Agent(ABC):
    name: str
    essential: bool = True

    @abstractmethod
    async def run(self, ctx: PipelineContext) -> PipelineContext: ...
```

```python
# src/video_intelligence/agents/prompting.py
"""Prompt-building helpers shared by LLM-calling agents."""
from __future__ import annotations

from ..schemas import Transcript


def _ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def transcript_lines(transcript: Transcript, block_s: float = 15.0) -> str:
    """Render a transcript as '[M:SS] text' lines, merging segments into ~block_s blocks."""
    lines: list[str] = []
    block_start: float | None = None
    buf: list[str] = []
    for seg in transcript.segments:
        if block_start is None:
            block_start = seg.start_s
        buf.append(seg.text)
        if seg.end_s - block_start >= block_s:
            lines.append(f"[{_ts(block_start)}] {' '.join(buf)}")
            block_start, buf = None, []
    if buf:
        lines.append(f"[{_ts(block_start)}] {' '.join(buf)}")
    return "\n".join(lines)


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most max_chars, breaking on line boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        if size + len(line) + 1 > max_chars and current:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompting.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/base.py src/video_intelligence/agents/prompting.py tests/test_prompting.py
git commit -m "feat: add agent base class and prompt helpers"
```

---

### Task 9: Ingestor agent

**Files:**
- Create: `src/video_intelligence/agents/ingestor.py`
- Test: `tests/test_ingestor.py`

**Interfaces:**
- Consumes: `Agent` (Task 8), schemas (Task 2).
- Produces: `IngestError(Exception)`; `extract_video_id(url: str) -> str`; `Ingestor(workdir="data/work", metadata_fetcher=..., caption_fetcher=..., audio_downloader=..., audio_extractor=...)`. Injectable callables (sync; run via `asyncio.to_thread`): `metadata_fetcher(url) -> dict` (keys `title`, `duration_s`, `channel`), `caption_fetcher(video_id, language) -> list[dict] | None` (raw items `{"text","start","duration"}`), `audio_downloader(url, workdir: Path) -> Path`, `audio_extractor(path: str, workdir: Path) -> Path`. After `run`: for YouTube-with-captions `ctx.transcript` is set (origin `CAPTIONS`) and `ctx.audio_path` stays None; otherwise `ctx.audio_path` is set.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingestor.py
from pathlib import Path

import pytest

from src.video_intelligence.agents.ingestor import Ingestor, IngestError, extract_video_id
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, TranscriptOrigin, VideoSource,
)

RAW_CAPTIONS = [
    {"text": "hello", "start": 0.0, "duration": 2.0},
    {"text": "world", "start": 2.0, "duration": 2.0},
]


def ctx_for(source: VideoSource, **opts) -> PipelineContext:
    return PipelineContext(source=source, options=JobOptions(**opts))


def make_ingestor(tmp_path, captions=RAW_CAPTIONS, downloads=None):
    calls = {"downloaded": False}

    def metadata_fetcher(url):
        return {"title": "T", "duration_s": 120.0, "channel": "C"}

    def caption_fetcher(video_id, language):
        return captions

    def audio_downloader(url, workdir: Path):
        calls["downloaded"] = True
        out = workdir / "a.m4a"
        out.write_bytes(b"fake")
        return out

    def audio_extractor(path, workdir: Path):
        out = workdir / "a.wav"
        out.write_bytes(b"fake")
        return out

    ing = Ingestor(workdir=tmp_path, metadata_fetcher=metadata_fetcher,
                   caption_fetcher=caption_fetcher, audio_downloader=audio_downloader,
                   audio_extractor=audio_extractor)
    return ing, calls


def test_extract_video_id_handles_common_forms():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    with pytest.raises(IngestError):
        extract_video_id("https://example.com/nope")


async def test_youtube_with_captions_skips_download(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"))
    ctx = await ing.run(ctx)
    assert ctx.transcript is not None
    assert ctx.transcript.origin == TranscriptOrigin.CAPTIONS
    assert ctx.transcript.segments[1].text == "world"
    assert ctx.transcript.segments[1].end_s == 4.0
    assert ctx.audio_path is None
    assert calls["downloaded"] is False
    assert ctx.source.title == "T"  # metadata resolved


async def test_youtube_without_captions_downloads_audio(tmp_path):
    ing, calls = make_ingestor(tmp_path, captions=None)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"))
    ctx = await ing.run(ctx)
    assert ctx.transcript is None
    assert ctx.audio_path is not None and calls["downloaded"] is True


async def test_force_whisper_ignores_captions(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
                  force_whisper=True)
    ctx = await ing.run(ctx)
    assert ctx.transcript is None and calls["downloaded"] is True


async def test_local_file_extracts_audio(tmp_path):
    ing, _ = make_ingestor(tmp_path)
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"fake video")
    ctx = ctx_for(VideoSource(kind=SourceKind.LOCAL_FILE, path=str(video)))
    ctx = await ing.run(ctx)
    assert ctx.audio_path.endswith(".wav")


async def test_missing_local_file_raises(tmp_path):
    ing, _ = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.LOCAL_FILE, path=str(tmp_path / "gone.mp4")))
    with pytest.raises(IngestError):
        await ing.run(ctx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/agents/ingestor.py
"""Resolve a video source into either captions or an audio file to transcribe."""
from __future__ import annotations

import asyncio
import re
import subprocess
import uuid
from pathlib import Path

from ..schemas import (
    PipelineContext, SourceKind, Transcript, TranscriptOrigin, TranscriptSegment,
)
from .base import Agent


class IngestError(Exception):
    pass


_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    m = _YOUTUBE_ID_RE.search(url)
    if not m:
        raise IngestError(f"could not extract YouTube video id from {url!r}")
    return m.group(1)


def default_metadata_fetcher(url: str) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {"title": info.get("title"), "duration_s": info.get("duration"),
            "channel": info.get("channel")}


def default_caption_fetcher(video_id: str, language: str) -> list[dict] | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        return YouTubeTranscriptApi().fetch(video_id, languages=[language]).to_raw_data()
    except Exception:
        return None


def default_audio_downloader(url: str, workdir: Path) -> Path:
    import yt_dlp
    stem = uuid.uuid4().hex
    with yt_dlp.YoutubeDL({
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": str(workdir / f"{stem}.%(ext)s"),
    }) as ydl:
        ydl.download([url])
    files = list(workdir.glob(f"{stem}.*"))
    if not files:
        raise IngestError(f"yt-dlp produced no audio file for {url!r}")
    return files[0]


def default_audio_extractor(path: str, workdir: Path) -> Path:
    out = workdir / f"{uuid.uuid4().hex}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", str(out)],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise IngestError(f"ffmpeg audio extraction failed: {e}") from e
    return out


class Ingestor(Agent):
    name = "ingest"
    essential = True

    def __init__(self, workdir: str | Path = "data/work",
                 metadata_fetcher=default_metadata_fetcher,
                 caption_fetcher=default_caption_fetcher,
                 audio_downloader=default_audio_downloader,
                 audio_extractor=default_audio_extractor):
        self._workdir = Path(workdir)
        self._metadata_fetcher = metadata_fetcher
        self._caption_fetcher = caption_fetcher
        self._audio_downloader = audio_downloader
        self._audio_extractor = audio_extractor

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        self._workdir.mkdir(parents=True, exist_ok=True)
        if ctx.source.kind == SourceKind.YOUTUBE:
            await self._ingest_youtube(ctx)
        else:
            await self._ingest_local(ctx)
        return ctx

    async def _ingest_youtube(self, ctx: PipelineContext) -> None:
        try:
            meta = await asyncio.to_thread(self._metadata_fetcher, ctx.source.url)
            ctx.source = ctx.source.model_copy(update=meta)
        except Exception:
            pass  # metadata is nice-to-have; never fail the job over it
        if not ctx.options.force_whisper:
            video_id = extract_video_id(ctx.source.url)
            raw = await asyncio.to_thread(self._caption_fetcher, video_id, ctx.options.language)
            if raw:
                ctx.transcript = Transcript(
                    segments=[
                        TranscriptSegment(start_s=r["start"], end_s=r["start"] + r["duration"],
                                          text=r["text"])
                        for r in raw
                    ],
                    language=ctx.options.language,
                    origin=TranscriptOrigin.CAPTIONS,
                )
                return
        path = await asyncio.to_thread(self._audio_downloader, ctx.source.url, self._workdir)
        ctx.audio_path = str(path)

    async def _ingest_local(self, ctx: PipelineContext) -> None:
        src = Path(ctx.source.path or "")
        if not src.exists():
            raise IngestError(f"local file not found: {ctx.source.path}")
        ctx.source = ctx.source.model_copy(update={"title": ctx.source.title or src.stem})
        path = await asyncio.to_thread(self._audio_extractor, str(src), self._workdir)
        ctx.audio_path = str(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestor.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/ingestor.py tests/test_ingestor.py
git commit -m "feat: add ingestor agent (captions-first YouTube + local file audio)"
```

---

### Task 10: Transcriber agent

**Files:**
- Create: `src/video_intelligence/agents/transcriber.py`
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: `Agent`, schemas.
- Produces: `TranscribeError(Exception)`; `Transcriber(model_name="base", model_factory=None)` where `model_factory(model_name) -> model` and the model has faster-whisper's shape: `model.transcribe(audio_path) -> (iterable_of_segments, info)` with segment attrs `.start`, `.end`, `.text` and `info.language`. Skips (no-op) when `ctx.transcript` is already set.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transcriber.py
from types import SimpleNamespace

import pytest

from src.video_intelligence.agents.transcriber import Transcriber, TranscribeError
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, Transcript, TranscriptOrigin,
    TranscriptSegment, VideoSource,
)


class FakeWhisperModel:
    def transcribe(self, audio_path):
        segments = [
            SimpleNamespace(start=0.0, end=3.0, text=" Hello there. "),
            SimpleNamespace(start=3.0, end=6.0, text=" General Kenobi. "),
        ]
        return iter(segments), SimpleNamespace(language="en")


def base_ctx(**kwargs) -> PipelineContext:
    return PipelineContext(
        source=VideoSource(kind=SourceKind.LOCAL_FILE, path="x.mp4"),
        options=JobOptions(), **kwargs,
    )


async def test_transcribes_audio_with_whisper():
    t = Transcriber(model_factory=lambda name: FakeWhisperModel())
    ctx = base_ctx(audio_path="/tmp/a.wav")
    ctx = await t.run(ctx)
    assert ctx.transcript.origin == TranscriptOrigin.WHISPER
    assert ctx.transcript.language == "en"
    assert ctx.transcript.segments[0].text == "Hello there."
    assert ctx.transcript.segments[1].end_s == 6.0


async def test_skips_when_captions_already_present():
    def exploding_factory(name):
        raise AssertionError("whisper must not be loaded when captions exist")
    t = Transcriber(model_factory=exploding_factory)
    existing = Transcript(
        segments=[TranscriptSegment(start_s=0, end_s=1, text="cap")],
        language="en", origin=TranscriptOrigin.CAPTIONS,
    )
    ctx = base_ctx(transcript=existing)
    ctx = await t.run(ctx)
    assert ctx.transcript is existing


async def test_no_audio_and_no_transcript_raises():
    t = Transcriber(model_factory=lambda name: FakeWhisperModel())
    with pytest.raises(TranscribeError):
        await t.run(base_ctx())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcriber.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/agents/transcriber.py
"""Local speech-to-text via faster-whisper; skipped when captions exist."""
from __future__ import annotations

import asyncio

from ..schemas import PipelineContext, Transcript, TranscriptOrigin, TranscriptSegment
from .base import Agent


class TranscribeError(Exception):
    pass


def _default_model_factory(model_name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, compute_type="int8")


class Transcriber(Agent):
    name = "transcribe"
    essential = True

    def __init__(self, model_name: str = "base", model_factory=None):
        self._model_name = model_name
        self._model_factory = model_factory or _default_model_factory

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.transcript is not None:
            return ctx  # captions already fetched by the ingestor
        if not ctx.audio_path:
            raise TranscribeError("no audio file and no transcript — ingest must have failed")
        segments, language = await asyncio.to_thread(self._transcribe, ctx.audio_path)
        ctx.transcript = Transcript(segments=segments, language=language,
                                    origin=TranscriptOrigin.WHISPER)
        return ctx

    def _transcribe(self, audio_path: str) -> tuple[list[TranscriptSegment], str]:
        model = self._model_factory(self._model_name)
        raw_segments, info = model.transcribe(audio_path)
        segments = [
            TranscriptSegment(start_s=s.start, end_s=s.end, text=s.text.strip())
            for s in raw_segments
        ]
        return segments, info.language
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcriber.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/transcriber.py tests/test_transcriber.py
git commit -m "feat: add whisper transcriber agent with caption skip"
```

---

### Task 11: Chapterizer agent

**Files:**
- Create: `src/video_intelligence/agents/chapterizer.py`
- Test: `tests/test_chapterizer.py`

**Interfaces:**
- Consumes: `Agent`, `transcript_lines`, `chunk_text` (Task 8), `Router.complete` (Task 7), schemas.
- Produces: `ChapterList(BaseModel)` with `chapters: list[Chapter]`; `Chapterizer(router)` with `name="chapterize"`, `essential=False`, `MAX_CHARS = 24_000`. Calls router task `"chaptering"` once per chunk; concatenates chunk results into `ctx.chapters`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chapterizer.py
from src.video_intelligence.agents.chapterizer import ChapterList, Chapterizer
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    Chapter, JobOptions, PipelineContext, SourceKind, Transcript, TranscriptOrigin,
    TranscriptSegment, VideoSource,
)
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"chaptering": {"balanced": ["fake/model-x"]}}}


def make(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    return Chapterizer(router), fake


def ctx_with_transcript(n_segments: int, words_per_segment: int = 5) -> PipelineContext:
    segs = [
        TranscriptSegment(start_s=i * 15.0, end_s=(i + 1) * 15.0,
                          text=" ".join(["word"] * words_per_segment))
        for i in range(n_segments)
    ]
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
        options=JobOptions(),
        transcript=Transcript(segments=segs, language="en", origin=TranscriptOrigin.WHISPER),
    )


async def test_single_chunk_produces_chapters(tmp_path):
    chapterizer, fake = make(tmp_path)
    fake.enqueue(ChapterList(chapters=[
        Chapter(start_s=0, end_s=60, title="Intro", synopsis="Opening remarks."),
        Chapter(start_s=60, end_s=150, title="Main", synopsis="The core argument."),
    ]))
    ctx = await chapterizer.run(ctx_with_transcript(10))
    assert [c.title for c in ctx.chapters] == ["Intro", "Main"]
    assert len(fake.calls) == 1
    assert "[0:00]" in fake.calls[0]["prompt"]  # timestamped transcript embedded


async def test_long_transcript_is_chunked_and_results_concatenated(tmp_path):
    chapterizer, fake = make(tmp_path)
    # ~700 segments x ~50 chars ≈ 35k chars -> 2 chunks at MAX_CHARS=24_000
    fake.enqueue(ChapterList(chapters=[Chapter(start_s=0, end_s=1, title="A", synopsis="a")]))
    fake.enqueue(ChapterList(chapters=[Chapter(start_s=1, end_s=2, title="B", synopsis="b")]))
    ctx = await chapterizer.run(ctx_with_transcript(700, words_per_segment=8))
    assert len(fake.calls) == 2
    assert [c.title for c in ctx.chapters] == ["A", "B"]


def test_chapterizer_is_non_essential(tmp_path):
    chapterizer, _ = make(tmp_path)
    assert chapterizer.essential is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chapterizer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/agents/chapterizer.py
"""Timestamped chapter segmentation via a cheap model."""
from __future__ import annotations

from pydantic import BaseModel

from ..models.router import Router
from ..schemas import Chapter, PipelineContext
from .base import Agent
from .prompting import chunk_text, transcript_lines


class ChapterList(BaseModel):
    chapters: list[Chapter]


CHAPTER_PROMPT = """You segment video transcripts into chapters.

Given the timestamped transcript below, return ONLY a JSON object:
{"chapters": [{"start_s": <float>, "end_s": <float>, "title": "<string>", "synopsis": "<string>"}]}

Rules:
- 3 to 12 chapters covering the whole transcript, in order, non-overlapping.
- Titles under 8 words; synopsis is one sentence.
- start_s / end_s are seconds, taken from the [M:SS] timestamps.

TRANSCRIPT:
<<TRANSCRIPT>>"""


class Chapterizer(Agent):
    name = "chapterize"
    essential = False
    MAX_CHARS = 24_000

    def __init__(self, router: Router):
        self._router = router

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        text = transcript_lines(ctx.transcript)
        chapters: list[Chapter] = []
        for chunk in chunk_text(text, self.MAX_CHARS):
            result = await self._router.complete(
                task="chaptering",
                quality=ctx.options.quality,
                prompt=CHAPTER_PROMPT.replace("<<TRANSCRIPT>>", chunk),
                schema=ChapterList,
                trace_id=ctx.trace_id,
                stage=self.name,
            )
            chapters.extend(result.chapters)
        ctx.chapters = chapters
        return ctx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chapterizer.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/chapterizer.py tests/test_chapterizer.py
git commit -m "feat: add chapterizer agent with chunking"
```

---

### Task 12: Synthesizer agent

**Files:**
- Create: `src/video_intelligence/agents/synthesizer.py`
- Test: `tests/test_synthesizer.py`

**Interfaces:**
- Consumes: `Agent`, prompt helpers, `Router.complete`, schemas.
- Produces: `SynthesisResult(BaseModel)` with `summary: str`, `key_quotes: list[KeyQuote]`, `action_items: list[str]`; `PartialSummary(BaseModel)` with `summary: str`; `Synthesizer(router)` with `name="synthesize"`, `essential=True`, `MAX_CHARS = 48_000`. Uses router task `"synthesis"`; map stage is `"synthesize.map"` at `QualityPreference.CHEAP`. Builds `ctx.report` from result + `ctx.chapters` + `ctx.degraded_stages`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthesizer.py
from src.video_intelligence.agents.synthesizer import PartialSummary, Synthesizer, SynthesisResult
from src.video_intelligence.models.router import Router
from src.video_intelligence.schemas import (
    Chapter, JobOptions, KeyQuote, PipelineContext, QualityPreference, SourceKind,
    Transcript, TranscriptOrigin, TranscriptSegment, VideoSource,
)
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

CONFIG = {"tasks": {"synthesis": {
    "balanced": ["fake/big-model"],
    "cheap": ["fake/small-model"],
}}}


def make(tmp_path):
    fake = FakeProvider("fake")
    router = Router(CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    return Synthesizer(router), fake


def ctx_with_transcript(n_segments: int, words_per_segment: int = 5) -> PipelineContext:
    segs = [
        TranscriptSegment(start_s=i * 15.0, end_s=(i + 1) * 15.0,
                          text=" ".join(["word"] * words_per_segment))
        for i in range(n_segments)
    ]
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ",
                           title="My Talk"),
        options=JobOptions(),
        transcript=Transcript(segments=segs, language="en", origin=TranscriptOrigin.WHISPER),
        chapters=[Chapter(start_s=0, end_s=60, title="Intro", synopsis="s")],
        degraded_stages=[],
    )


RESULT = SynthesisResult(
    summary="A fine talk.",
    key_quotes=[KeyQuote(timestamp_s=42.0, text="quote")],
    action_items=["do the thing"],
)


async def test_builds_report_from_synthesis(tmp_path):
    synth, fake = make(tmp_path)
    fake.enqueue(RESULT)
    ctx = await synth.run(ctx_with_transcript(10))
    assert ctx.report.summary == "A fine talk."
    assert ctx.report.chapters[0].title == "Intro"
    assert ctx.report.key_quotes[0].timestamp_s == 42.0
    assert ctx.report.trace_id == ctx.trace_id
    assert "My Talk" in fake.calls[0]["prompt"]
    assert len(fake.calls) == 1


async def test_long_transcript_uses_map_reduce(tmp_path):
    synth, fake = make(tmp_path)
    # ~1400 segments ≈ 70k chars -> 2 map calls + 1 final call
    fake.enqueue(PartialSummary(summary="part one"))
    fake.enqueue(PartialSummary(summary="part two"))
    fake.enqueue(RESULT)
    ctx = await synth.run(ctx_with_transcript(1400, words_per_segment=8))
    assert len(fake.calls) == 3
    # map calls go to the cheap tier
    assert fake.calls[0]["model"] == "small-model"
    assert fake.calls[2]["model"] == "big-model"
    assert "part one" in fake.calls[2]["prompt"]
    assert ctx.report.summary == "A fine talk."


async def test_report_carries_degraded_stages(tmp_path):
    synth, fake = make(tmp_path)
    fake.enqueue(RESULT)
    ctx = ctx_with_transcript(10)
    ctx.chapters = None
    ctx.degraded_stages = ["chapterize"]
    ctx = await synth.run(ctx)
    assert ctx.report.chapters == []
    assert ctx.report.degraded_stages == ["chapterize"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesizer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/agents/synthesizer.py
"""Frontier-model synthesis: summary, key quotes, action items -> AnalysisReport."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models.router import Router
from ..schemas import AnalysisReport, KeyQuote, PipelineContext, QualityPreference
from .base import Agent
from .prompting import chunk_text, transcript_lines


class SynthesisResult(BaseModel):
    summary: str
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)


class PartialSummary(BaseModel):
    summary: str


SYNTH_PROMPT = """You are an analyst producing a structured report on a video.

Video title: <<TITLE>>
Chapters (may be empty): <<CHAPTERS>>

Timestamped transcript (or partial summaries for long videos):
<<TRANSCRIPT>>

Return ONLY a JSON object:
{"summary": "<markdown, 150-400 words>",
 "key_quotes": [{"timestamp_s": <float>, "speaker": null, "text": "<verbatim quote>"}],
 "action_items": ["<string>", ...]}

Rules:
- Write everything in language code: <<LANGUAGE>>.
- 3 to 6 key quotes with timestamps from the [M:SS] markers.
- action_items may be an empty list when the video contains none."""


MAP_PROMPT = """Summarize this portion of a video transcript in under 300 words.
Preserve the most notable statements verbatim with their [M:SS] timestamps.
Return ONLY a JSON object: {"summary": "<string>"}

TRANSCRIPT PORTION:
<<TRANSCRIPT>>"""


class Synthesizer(Agent):
    name = "synthesize"
    essential = True
    MAX_CHARS = 48_000

    def __init__(self, router: Router):
        self._router = router

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        text = transcript_lines(ctx.transcript)
        if len(text) > self.MAX_CHARS:
            text = await self._reduce(ctx, text)
        chapters_str = "; ".join(f"{c.title} ({c.synopsis})" for c in (ctx.chapters or [])) or "none"
        prompt = (SYNTH_PROMPT
                  .replace("<<TITLE>>", ctx.source.title or "unknown")
                  .replace("<<CHAPTERS>>", chapters_str)
                  .replace("<<TRANSCRIPT>>", text)
                  .replace("<<LANGUAGE>>", ctx.options.language))
        result = await self._router.complete(
            task="synthesis", quality=ctx.options.quality, prompt=prompt,
            schema=SynthesisResult, trace_id=ctx.trace_id, stage=self.name,
        )
        ctx.report = AnalysisReport(
            summary=result.summary,
            chapters=ctx.chapters or [],
            key_quotes=result.key_quotes,
            action_items=result.action_items,
            language=ctx.options.language,
            trace_id=ctx.trace_id,
            degraded_stages=list(ctx.degraded_stages),
        )
        return ctx

    async def _reduce(self, ctx: PipelineContext, text: str) -> str:
        parts: list[str] = []
        for chunk in chunk_text(text, self.MAX_CHARS):
            partial = await self._router.complete(
                task="synthesis", quality=QualityPreference.CHEAP,
                prompt=MAP_PROMPT.replace("<<TRANSCRIPT>>", chunk),
                schema=PartialSummary, trace_id=ctx.trace_id, stage="synthesize.map",
            )
            parts.append(partial.summary)
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesizer.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: add synthesizer agent with map-reduce for long transcripts"
```

---

### Task 13: Pipeline orchestrator + integration test

**Files:**
- Create: `src/video_intelligence/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `PipelineError(Exception)` with `.stage` and `.reason` attributes; `EventCallback = Callable[[StageEvent], Awaitable[None]]`; `Pipeline(agents: list[Agent], on_event: EventCallback | None = None)` with `async run(source: VideoSource, options: JobOptions) -> AnalysisReport`; factory `build_pipeline(config_path="config/models.yaml", db_path="data/traces.db", workdir="data/work", on_event=None) -> Pipeline` wiring real providers/router/agents (whisper model name read from `config["transcription"]["whisper_model"]`). **The API adapter (Task 14) calls `build_pipeline(on_event=...)` — keep that keyword.**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import pytest

from src.video_intelligence.agents.base import Agent
from src.video_intelligence.agents.chapterizer import ChapterList, Chapterizer
from src.video_intelligence.agents.ingestor import Ingestor
from src.video_intelligence.agents.synthesizer import Synthesizer, SynthesisResult
from src.video_intelligence.agents.transcriber import Transcriber
from src.video_intelligence.models.providers.base import ProviderError
from src.video_intelligence.models.router import Router
from src.video_intelligence.pipeline import Pipeline, PipelineError
from src.video_intelligence.schemas import Chapter, JobOptions, SourceKind, VideoSource
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider

RAW_CAPTIONS = [
    {"text": "hello", "start": 0.0, "duration": 2.0},
    {"text": "world", "start": 2.0, "duration": 2.0},
]

CONFIG = {"tasks": {
    "chaptering": {"balanced": ["fake/small"]},
    "synthesis": {"balanced": ["fake/big"], "cheap": ["fake/small"]},
}}


def build_test_pipeline(tmp_path, fake: FakeProvider, on_event=None) -> Pipeline:
    store = TraceStore(tmp_path / "t.db")
    router = Router(CONFIG, {"fake": fake}, store)
    ingestor = Ingestor(
        workdir=tmp_path,
        metadata_fetcher=lambda url: {"title": "T", "duration_s": 4.0, "channel": "C"},
        caption_fetcher=lambda vid, lang: RAW_CAPTIONS,
        audio_downloader=lambda url, wd: (_ for _ in ()).throw(AssertionError("no download")),
        audio_extractor=lambda p, wd: (_ for _ in ()).throw(AssertionError("no extract")),
    )
    transcriber = Transcriber(model_factory=lambda name: (_ for _ in ()).throw(
        AssertionError("whisper must not load")))
    return Pipeline(
        [ingestor, transcriber, Chapterizer(router), Synthesizer(router)],
        on_event=on_event,
    ), store


def youtube_source() -> VideoSource:
    return VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ")


async def test_full_pipeline_produces_report_and_events(tmp_path):
    fake = FakeProvider("fake")
    fake.enqueue(ChapterList(chapters=[Chapter(start_s=0, end_s=4, title="All", synopsis="s")]))
    fake.enqueue(SynthesisResult(summary="Nice video."))
    events = []

    async def on_event(ev):
        events.append((ev.stage, ev.type))

    pipeline, store = build_test_pipeline(tmp_path, fake, on_event=on_event)
    report = await pipeline.run(youtube_source(), JobOptions())

    assert report.summary == "Nice video."
    assert report.chapters[0].title == "All"
    assert report.degraded_stages == []
    # every stage traced or evented
    assert ("ingest", "started") in events
    assert ("synthesize", "completed") in events
    # spans exist for the two model calls
    stages = [s.stage for s in store.spans(report.trace_id)]
    assert stages == ["chapterize", "synthesize"]


async def test_non_essential_failure_degrades_but_completes(tmp_path):
    fake = FakeProvider("fake")
    # chapterizer: initial + retry both fail -> RouterError -> degraded
    fake.enqueue(ProviderError("boom"))
    fake.enqueue(ProviderError("boom"))
    fake.enqueue(SynthesisResult(summary="Still fine."))
    pipeline, _ = build_test_pipeline(tmp_path, fake)
    report = await pipeline.run(youtube_source(), JobOptions())
    assert report.summary == "Still fine."
    assert report.chapters == []
    assert report.degraded_stages == ["chapterize"]


async def test_essential_failure_raises_pipeline_error(tmp_path):
    fake = FakeProvider("fake")
    fake.enqueue(ChapterList(chapters=[]))
    fake.enqueue(ProviderError("boom"))
    fake.enqueue(ProviderError("boom"))
    pipeline, _ = build_test_pipeline(tmp_path, fake)
    with pytest.raises(PipelineError) as exc:
        await pipeline.run(youtube_source(), JobOptions())
    assert exc.value.stage == "synthesize"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/video_intelligence/pipeline.py
"""Async orchestrator: runs agents in order, emits stage events, enforces error policy."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from .agents.base import Agent
from .schemas import AnalysisReport, JobOptions, PipelineContext, StageEvent, VideoSource

EventCallback = Callable[[StageEvent], Awaitable[None]]


class PipelineError(Exception):
    def __init__(self, stage: str, reason: str):
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage}: {reason}")


class Pipeline:
    def __init__(self, agents: list[Agent], on_event: EventCallback | None = None):
        self._agents = agents
        self._on_event = on_event

    async def _emit(self, stage: str, type_: str, message: str | None = None) -> None:
        if self._on_event is not None:
            await self._on_event(StageEvent(stage=stage, type=type_, message=message))

    async def run(self, source: VideoSource, options: JobOptions) -> AnalysisReport:
        ctx = PipelineContext(source=source, options=options)
        for agent in self._agents:
            await self._emit(agent.name, "started")
            try:
                ctx = await agent.run(ctx)
            except Exception as e:
                if agent.essential:
                    await self._emit(agent.name, "failed", str(e))
                    raise PipelineError(agent.name, str(e)) from e
                ctx.degraded_stages.append(agent.name)
                await self._emit(agent.name, "failed", f"degraded: {e}")
                continue
            await self._emit(agent.name, "completed")
        if ctx.report is None:
            raise PipelineError("synthesize", "pipeline finished without a report")
        return ctx.report


def build_pipeline(config_path: str = "config/models.yaml",
                   db_path: str = "data/traces.db",
                   workdir: str = "data/work",
                   on_event: EventCallback | None = None) -> Pipeline:
    """Wire the production pipeline: real providers, router, and agents."""
    from .agents.chapterizer import Chapterizer
    from .agents.ingestor import Ingestor
    from .agents.synthesizer import Synthesizer
    from .agents.transcriber import Transcriber
    from .models.providers.anthropic import AnthropicProvider
    from .models.providers.ollama import OllamaProvider
    from .models.providers.openai import OpenAIProvider
    from .models.router import Router, load_model_config
    from .tracing import TraceStore

    config = load_model_config(config_path)
    store = TraceStore(db_path)
    providers = {
        "ollama": OllamaProvider(),
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
    }
    router = Router(config, providers, store)
    whisper_model = config.get("transcription", {}).get("whisper_model", "base")
    return Pipeline(
        [
            Ingestor(workdir=workdir),
            Transcriber(model_name=whisper_model),
            Chapterizer(router),
            Synthesizer(router),
        ],
        on_event=on_event,
    )
```

- [ ] **Step 4: Run tests to verify they pass (full suite — this is the integration gate)**

Run: `pytest -v`
Expected: ALL PASSED (schemas, tracing, providers, router, prompting, agents, pipeline)

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator with degraded-stage policy and build_pipeline factory"
```

---

### Task 14: FastAPI adapter (job store, endpoints, SSE)

**Files:**
- Create: `src/api/jobs.py`, `src/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `build_pipeline` signature (Task 13), `Pipeline.run`, `PipelineError`, schemas, `TraceStore`.
- Produces HTTP API (all under `/api`):
  - `POST /api/jobs` body `{"url": str, "options": JobOptions?}` → `{"job_id": str}`
  - `POST /api/jobs/upload` multipart (`file`, form fields `language`, `quality`, `force_whisper`) → `{"job_id": str}`
  - `GET /api/jobs/{id}` → `{"job_id", "status": queued|running|completed|failed, "report": AnalysisReport|null, "error": str|null}`
  - `GET /api/jobs/{id}/trace` → `{"spans": [TraceSpan], "total_cost_usd": float}`
  - `GET /api/jobs/{id}/events` → SSE stream of `StageEvent` JSON, ending after a terminal event
  - `create_app(pipeline_factory=build_pipeline, db_path=..., trace_db=..., upload_dir=...) -> FastAPI` — `pipeline_factory` must accept `on_event=` keyword.
- Produces `JobStore(db_path)`: `.create(job_id, source, options)`, `.update(job_id, status=None, report=None, error=None, trace_id=None)`, `.get(job_id) -> dict | None`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.api.jobs'` (or `main`)

- [ ] **Step 3: Write `src/api/jobs.py`**

```python
# src/api/jobs.py
"""SQLite job store for the API adapter."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.video_intelligence.schemas import AnalysisReport, JobOptions, VideoSource


class JobStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    report_json TEXT,
                    error TEXT,
                    trace_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, job_id: str, source: VideoSource, options: JobOptions) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, source_json, options_json) VALUES (?, 'queued', ?, ?)",
                (job_id, source.model_dump_json(), options.model_dump_json()),
            )

    def update(self, job_id: str, status: str | None = None,
               report: AnalysisReport | None = None, error: str | None = None,
               trace_id: str | None = None) -> None:
        sets, vals = [], []
        if status is not None:
            sets.append("status = ?"); vals.append(status)
        if report is not None:
            sets.append("report_json = ?"); vals.append(report.model_dump_json())
        if error is not None:
            sets.append("error = ?"); vals.append(error)
        if trace_id is not None:
            sets.append("trace_id = ?"); vals.append(trace_id)
        if not sets:
            return
        vals.append(job_id)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)

    def get(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return {
            "job_id": row["id"],
            "status": row["status"],
            "source": json.loads(row["source_json"]),
            "options": json.loads(row["options_json"]),
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
            "error": row["error"],
            "trace_id": row["trace_id"],
        }
```

- [ ] **Step 4: Write `src/api/main.py`**

```python
# src/api/main.py
"""FastAPI adapter over the video_intelligence pipeline."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.video_intelligence.pipeline import PipelineError, build_pipeline
from src.video_intelligence.schemas import (
    JobOptions, QualityPreference, SourceKind, StageEvent, VideoSource,
)
from src.video_intelligence.tracing import TraceStore

from .jobs import JobStore


class CreateJobRequest(BaseModel):
    url: str
    options: JobOptions = JobOptions()


def create_app(pipeline_factory=build_pipeline,
               db_path: str | Path = "data/app.db",
               trace_db: str | Path = "data/traces.db",
               upload_dir: str | Path = "data/uploads") -> FastAPI:
    app = FastAPI(title="Video Intelligence")
    store = JobStore(db_path)
    trace_store = TraceStore(trace_db)
    upload_dir = Path(upload_dir)
    queues: dict[str, asyncio.Queue] = {}

    async def _run_job(job_id: str, source: VideoSource, options: JobOptions) -> None:
        queue = queues[job_id]

        async def on_event(ev: StageEvent) -> None:
            await queue.put(ev)

        pipeline = pipeline_factory(on_event=on_event)
        store.update(job_id, status="running")
        try:
            report = await pipeline.run(source, options)
        except PipelineError as e:
            store.update(job_id, status="failed", error=str(e))
            await queue.put(StageEvent(stage=e.stage, type="failed", message=e.reason))
        else:
            store.update(job_id, status="completed", report=report, trace_id=report.trace_id)
            await queue.put(StageEvent(stage="pipeline", type="completed"))
        await queue.put(None)  # sentinel: closes the SSE stream

    def _start_job(background: BackgroundTasks, source: VideoSource,
                   options: JobOptions) -> dict:
        job_id = uuid.uuid4().hex
        store.create(job_id, source, options)
        queues[job_id] = asyncio.Queue()
        background.add_task(_run_job, job_id, source, options)
        return {"job_id": job_id}

    @app.post("/api/jobs")
    async def create_job(req: CreateJobRequest, background: BackgroundTasks) -> dict:
        source = VideoSource(kind=SourceKind.YOUTUBE, url=req.url)
        return _start_job(background, source, req.options)

    @app.post("/api/jobs/upload")
    async def upload_job(background: BackgroundTasks,
                         file: UploadFile = File(...),
                         language: str = Form("en"),
                         quality: str = Form("balanced"),
                         force_whisper: bool = Form(False)) -> dict:
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / f"{uuid.uuid4().hex}-{file.filename}"
        dest.write_bytes(await file.read())
        source = VideoSource(kind=SourceKind.LOCAL_FILE, path=str(dest), title=file.filename)
        options = JobOptions(language=language, quality=QualityPreference(quality),
                             force_whisper=force_whisper)
        return _start_job(background, source, options)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/api/jobs/{job_id}/trace")
    async def get_trace(job_id: str) -> dict:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        trace_id = job.get("trace_id")
        spans = trace_store.spans(trace_id) if trace_id else []
        return {"spans": [s.model_dump() for s in spans],
                "total_cost_usd": trace_store.total_cost(trace_id) if trace_id else 0.0}

    @app.get("/api/jobs/{job_id}/events")
    async def stream_events(job_id: str) -> StreamingResponse:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        async def gen():
            queue = queues.get(job_id)
            if queue is None or job["status"] in ("completed", "failed"):
                # job already finished (or process restarted): replay terminal status
                ev = StageEvent(stage="pipeline", type=job["status"], message=job.get("error"))
                yield f"data: {ev.model_dump_json()}\n\n"
                return
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield f"data: {ev.model_dump_json()}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = create_app()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: 6 PASSED

- [ ] **Step 6: Manual smoke check of the server booting**

```bash
python -c "from src.api.main import app; print(type(app).__name__)"
```
Expected: `FastAPI`

- [ ] **Step 7: Commit**

```bash
git add src/api/jobs.py src/api/main.py tests/test_api.py
git commit -m "feat: add FastAPI adapter with job store, SSE progress, and trace endpoint"
```

---

### Task 15: Frontend scaffold + submit/progress flow

**Files:**
- Create: `frontend/` (Vite scaffold), `frontend/src/api.ts`, `frontend/src/App.tsx`, `frontend/src/components/SubmitForm.tsx`, `frontend/src/components/JobProgress.tsx`
- Modify: `frontend/vite.config.ts`, `frontend/src/index.css`

**Interfaces:**
- Consumes: the HTTP API from Task 14 (paths and JSON shapes exactly as listed there).
- Produces: TypeScript types `Report`, `Chapter`, `KeyQuote`, `TraceSpan`, `Job`, `StageEvent`; api functions `createJob(url, options)`, `uploadJob(file, options)`, `getJob(id)`, `getTrace(id)`, `eventsUrl(id)`; components `SubmitForm` (props `{onSubmitted(jobId: string): void}`) and `JobProgress` (props `{jobId: string, onFinished(): void}`). Task 16 imports these types and adds `ReportView`/`TraceTable` into `App.tsx` where marked.

- [ ] **Step 1: Scaffold the app**

```bash
cd /path/to/repo  # repo root
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install tailwindcss @tailwindcss/vite
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom
```

- [ ] **Step 2: Configure Vite (proxy + tailwind + vitest)**

Replace `frontend/vite.config.ts`:

```ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
  },
})
```

Replace the contents of `frontend/src/index.css` with:

```css
@import "tailwindcss";
```

Create `frontend/src/test/setup.ts`:

```ts
import '@testing-library/jest-dom'
```

Add to `frontend/package.json` scripts: `"test": "vitest run"`.

- [ ] **Step 3: Write the API client**

```ts
// frontend/src/api.ts
export interface Chapter { start_s: number; end_s: number; title: string; synopsis: string }
export interface KeyQuote { timestamp_s: number; speaker: string | null; text: string }
export interface Report {
  summary: string; chapters: Chapter[]; key_quotes: KeyQuote[]
  action_items: string[]; language: string; trace_id: string; degraded_stages: string[]
}
export interface Job {
  job_id: string; status: 'queued' | 'running' | 'completed' | 'failed'
  report: Report | null; error: string | null
}
export interface TraceSpan {
  stage: string; model_used: string; tokens_in: number; tokens_out: number
  cost_usd: number; latency_ms: number; status: string; fallback_from: string | null
}
export interface Trace { spans: TraceSpan[]; total_cost_usd: number }
export interface StageEvent { stage: string; type: string; message: string | null }
export interface JobOptions { language: string; quality: 'cheap' | 'balanced' | 'best'; force_whisper: boolean }

async function json<T>(respPromise: Promise<Response>): Promise<T> {
  const resp = await respPromise
  if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`)
  return resp.json()
}

export const createJob = (url: string, options: JobOptions) =>
  json<{ job_id: string }>(fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, options }),
  }))

export const uploadJob = (file: File, options: JobOptions) => {
  const form = new FormData()
  form.append('file', file)
  form.append('language', options.language)
  form.append('quality', options.quality)
  form.append('force_whisper', String(options.force_whisper))
  return json<{ job_id: string }>(fetch('/api/jobs/upload', { method: 'POST', body: form }))
}

export const getJob = (id: string) => json<Job>(fetch(`/api/jobs/${id}`))
export const getTrace = (id: string) => json<Trace>(fetch(`/api/jobs/${id}/trace`))
export const eventsUrl = (id: string) => `/api/jobs/${id}/events`
```

- [ ] **Step 4: Write SubmitForm and JobProgress**

```tsx
// frontend/src/components/SubmitForm.tsx
import { useState } from 'react'
import { createJob, uploadJob, type JobOptions } from '../api'

export default function SubmitForm({ onSubmitted }: { onSubmitted: (jobId: string) => void }) {
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [quality, setQuality] = useState<JobOptions['quality']>('balanced')
  const [language, setLanguage] = useState('en')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true); setError(null)
    const options: JobOptions = { language, quality, force_whisper: false }
    try {
      const { job_id } = file ? await uploadJob(file, options) : await createJob(url, options)
      onSubmitted(job_id)
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4 rounded-xl border border-slate-200 p-6">
      <input
        className="w-full rounded-lg border border-slate-300 px-3 py-2"
        placeholder="YouTube URL"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
      />
      <div className="flex items-center gap-4 text-sm">
        <span className="text-slate-500">or upload:</span>
        <input type="file" accept="video/*,audio/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>
      <div className="flex gap-4">
        <select className="rounded-lg border border-slate-300 px-2 py-1"
                value={quality} onChange={(e) => setQuality(e.target.value as JobOptions['quality'])}>
          <option value="cheap">Cheap</option>
          <option value="balanced">Balanced</option>
          <option value="best">Best</option>
        </select>
        <select className="rounded-lg border border-slate-300 px-2 py-1"
                value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="en">English</option>
          <option value="fr">Français</option>
        </select>
        <button disabled={busy || (!url && !file)}
                className="rounded-lg bg-slate-900 px-4 py-1.5 text-white disabled:opacity-40">
          {busy ? 'Submitting…' : 'Analyze'}
        </button>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
    </form>
  )
}
```

```tsx
// frontend/src/components/JobProgress.tsx
import { useEffect, useState } from 'react'
import { eventsUrl, type StageEvent } from '../api'

const STAGES = ['ingest', 'transcribe', 'chapterize', 'synthesize']

export default function JobProgress({ jobId, onFinished }: { jobId: string; onFinished: () => void }) {
  const [events, setEvents] = useState<StageEvent[]>([])

  useEffect(() => {
    const es = new EventSource(eventsUrl(jobId))
    es.onmessage = (msg) => {
      const ev: StageEvent = JSON.parse(msg.data)
      setEvents((prev) => [...prev, ev])
      if (ev.stage === 'pipeline' || ev.type === 'failed') {
        es.close()
        onFinished()
      }
    }
    es.onerror = () => { es.close(); onFinished() }
    return () => es.close()
  }, [jobId, onFinished])

  const statusOf = (stage: string) => {
    const evs = events.filter((e) => e.stage === stage)
    if (evs.some((e) => e.type === 'completed')) return '✓'
    if (evs.some((e) => e.type === 'failed')) return '✗'
    if (evs.some((e) => e.type === 'started')) return '…'
    return '·'
  }

  return (
    <ol className="flex gap-6 text-sm">
      {STAGES.map((s) => (
        <li key={s} className="flex items-center gap-2">
          <span className="font-mono">{statusOf(s)}</span>
          <span className="capitalize">{s}</span>
        </li>
      ))}
    </ol>
  )
}
```

- [ ] **Step 5: Wire App.tsx**

```tsx
// frontend/src/App.tsx
import { useCallback, useState } from 'react'
import { getJob, type Job } from './api'
import SubmitForm from './components/SubmitForm'
import JobProgress from './components/JobProgress'

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<Job | null>(null)

  const onFinished = useCallback(async () => {
    if (jobId) setJob(await getJob(jobId))
  }, [jobId])

  return (
    <main className="mx-auto max-w-3xl space-y-8 p-8">
      <h1 className="text-2xl font-semibold">Video Intelligence</h1>
      <SubmitForm onSubmitted={(id) => { setJobId(id); setJob(null) }} />
      {jobId && !job && <JobProgress jobId={jobId} onFinished={onFinished} />}
      {job?.status === 'failed' && <p className="text-red-600">Failed: {job.error}</p>}
      {job?.status === 'completed' && job.report && (
        <pre className="overflow-x-auto rounded-lg bg-slate-100 p-4 text-xs">
          {JSON.stringify(job.report, null, 2)}
        </pre> /* replaced by ReportView in the next task */
      )}
    </main>
  )
}
```

- [ ] **Step 6: Verify build + typecheck**

```bash
cd frontend && npm run build
```
Expected: build succeeds (tsc + vite build, no errors).

- [ ] **Step 7: Manual end-to-end sanity check** (requires network + a summarization key)

```bash
# terminal 1, repo root
uvicorn src.api.main:app --port 8000
# terminal 2
cd frontend && npm run dev
```
Open http://localhost:5173, submit a short captioned YouTube URL with quality "cheap"; expect stage ticks then a JSON report. If no API keys/Ollama are configured, expect a failed job with a clear synthesize error — that's the correct degradation.

- [ ] **Step 8: Commit**

```bash
git add frontend
git commit -m "feat: add React SPA scaffold with submit form and SSE progress"
```

---

### Task 16: Frontend report + trace views

**Files:**
- Create: `frontend/src/components/ReportView.tsx`, `frontend/src/components/TraceTable.tsx`, `frontend/src/components/ReportView.test.tsx`
- Modify: `frontend/src/App.tsx` (replace the `<pre>` placeholder)

**Interfaces:**
- Consumes: `Report`, `Trace`, `getTrace` from `frontend/src/api.ts` (Task 15).
- Produces: `ReportView({ report }: { report: Report })`, `TraceTable({ trace }: { trace: Trace })`.

- [ ] **Step 1: Write the failing render test**

```tsx
// frontend/src/components/ReportView.test.tsx
import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ReportView from './ReportView'
import type { Report } from '../api'

const report: Report = {
  summary: 'A great talk about pipelines.',
  chapters: [{ start_s: 0, end_s: 90, title: 'Introduction', synopsis: 'Opening.' }],
  key_quotes: [{ timestamp_s: 42, speaker: null, text: 'Ship it.' }],
  action_items: ['Try the demo'],
  language: 'en',
  trace_id: 'tr1',
  degraded_stages: [],
}

describe('ReportView', () => {
  it('renders summary, chapters with timestamps, quotes, and action items', () => {
    render(<ReportView report={report} />)
    expect(screen.getByText('A great talk about pipelines.')).toBeInTheDocument()
    expect(screen.getByText('Introduction')).toBeInTheDocument()
    expect(screen.getByText('0:00')).toBeInTheDocument()
    expect(screen.getByText('"Ship it."')).toBeInTheDocument()
    expect(screen.getByText('Try the demo')).toBeInTheDocument()
  })

  it('shows a degraded banner when stages were skipped', () => {
    render(<ReportView report={{ ...report, degraded_stages: ['chapterize'] }} />)
    expect(screen.getByText(/degraded/i)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL — cannot resolve `./ReportView`

- [ ] **Step 3: Write ReportView and TraceTable**

```tsx
// frontend/src/components/ReportView.tsx
import type { Report } from '../api'

export function formatTs(s: number): string {
  const total = Math.floor(s)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const sec = total % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
           : `${m}:${String(sec).padStart(2, '0')}`
}

export default function ReportView({ report }: { report: Report }) {
  return (
    <article className="space-y-6">
      {report.degraded_stages.length > 0 && (
        <p className="rounded-lg bg-amber-50 px-4 py-2 text-sm text-amber-800">
          Degraded report — skipped stages: {report.degraded_stages.join(', ')}
        </p>
      )}
      <section>
        <h2 className="mb-2 text-lg font-semibold">Summary</h2>
        <p className="whitespace-pre-wrap text-slate-700">{report.summary}</p>
      </section>
      {report.chapters.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Chapters</h2>
          <ol className="space-y-1">
            {report.chapters.map((c, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-16 shrink-0 font-mono text-sm text-slate-500">{formatTs(c.start_s)}</span>
                <span><strong>{c.title}</strong> — {c.synopsis}</span>
              </li>
            ))}
          </ol>
        </section>
      )}
      {report.key_quotes.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Key quotes</h2>
          <ul className="space-y-2">
            {report.key_quotes.map((q, i) => (
              <li key={i} className="border-l-2 border-slate-300 pl-3">
                <span className="mr-2 font-mono text-sm text-slate-500">{formatTs(q.timestamp_s)}</span>
                <em>"{q.text}"</em>
              </li>
            ))}
          </ul>
        </section>
      )}
      {report.action_items.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Action items</h2>
          <ul className="list-inside list-disc space-y-1">
            {report.action_items.map((a, i) => <li key={i}>{a}</li>)}
          </ul>
        </section>
      )}
    </article>
  )
}
```

```tsx
// frontend/src/components/TraceTable.tsx
import type { Trace } from '../api'

export default function TraceTable({ trace }: { trace: Trace }) {
  return (
    <section>
      <h2 className="mb-2 text-lg font-semibold">
        Trace <span className="text-sm font-normal text-slate-500">
          (total ${trace.total_cost_usd.toFixed(4)})
        </span>
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b text-slate-500">
              <th className="py-1 pr-4">Stage</th><th className="pr-4">Model</th>
              <th className="pr-4">Tokens in/out</th><th className="pr-4">Cost</th>
              <th className="pr-4">Latency</th><th>Fallback from</th>
            </tr>
          </thead>
          <tbody>
            {trace.spans.map((s, i) => (
              <tr key={i} className="border-b border-slate-100">
                <td className="py-1 pr-4">{s.stage}</td>
                <td className="pr-4 font-mono text-xs">{s.model_used}</td>
                <td className="pr-4">{s.tokens_in}/{s.tokens_out}</td>
                <td className="pr-4">${s.cost_usd.toFixed(4)}</td>
                <td className="pr-4">{s.latency_ms} ms</td>
                <td className="font-mono text-xs">{s.fallback_from ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
```

- [ ] **Step 4: Replace the App.tsx placeholder**

In `frontend/src/App.tsx`, replace the completed-job `<pre>` block with:

```tsx
      {job?.status === 'completed' && job.report && (
        <>
          <ReportView report={job.report} />
          {trace && <TraceTable trace={trace} />}
        </>
      )}
```

and add trace loading (imports at top: `import ReportView from './components/ReportView'`, `import TraceTable from './components/TraceTable'`, extend the api import to `import { getJob, getTrace, type Job, type Trace } from './api'`; state `const [trace, setTrace] = useState<Trace | null>(null)`; in `onFinished` after `setJob(...)`: `setTrace(await getTrace(jobId))`; reset it in `onSubmitted`: `setTrace(null)`).

- [ ] **Step 5: Run tests + build to verify**

```bash
cd frontend && npm test && npm run build
```
Expected: 2 tests PASSED; build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src
git commit -m "feat: add report and trace views to SPA"
```

---

### Task 17: Whisper smoke test (slow) + audio fixture

**Files:**
- Create: `scripts/make_fixture.sh`, `tests/test_whisper_smoke.py`
- Create (generated, committed): `tests/fixtures/spoken_30s.wav`

**Interfaces:**
- Consumes: `Transcriber` (Task 10).
- Produces: a committed ~30s spoken-audio fixture and a `slow`-marked regression test for real faster-whisper transcription.

- [ ] **Step 1: Write the fixture generator**

```bash
# scripts/make_fixture.sh
#!/usr/bin/env bash
# Generates tests/fixtures/spoken_30s.wav using macOS `say` + ffmpeg.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tests/fixtures
TEXT="This is a test recording for the video intelligence pipeline. \
The pipeline transcribes audio with whisper, segments it into chapters, \
and synthesizes a structured report with key quotes and action items. \
This sentence exists purely so the smoke test has known words to find: \
pipeline, whisper, chapters, report."
say -o /tmp/vi_fixture.aiff "$TEXT"
ffmpeg -y -i /tmp/vi_fixture.aiff -ac 1 -ar 16000 tests/fixtures/spoken_30s.wav
echo "wrote tests/fixtures/spoken_30s.wav"
```

```bash
chmod +x scripts/make_fixture.sh && ./scripts/make_fixture.sh
```
Expected: `wrote tests/fixtures/spoken_30s.wav` (file ~15–30s of audio, <1 MB).

- [ ] **Step 2: Write the slow test**

```python
# tests/test_whisper_smoke.py
from pathlib import Path

import pytest

from src.video_intelligence.agents.transcriber import Transcriber
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, TranscriptOrigin, VideoSource,
)

FIXTURE = Path(__file__).parent / "fixtures" / "spoken_30s.wav"


@pytest.mark.slow
@pytest.mark.skipif(not FIXTURE.exists(), reason="run scripts/make_fixture.sh first")
async def test_real_whisper_transcribes_fixture():
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.LOCAL_FILE, path=str(FIXTURE)),
        options=JobOptions(),
        audio_path=str(FIXTURE),
    )
    ctx = await Transcriber(model_name="base").run(ctx)
    assert ctx.transcript.origin == TranscriptOrigin.WHISPER
    text = ctx.transcript.full_text.lower()
    assert "pipeline" in text
    assert "whisper" in text
    assert ctx.transcript.segments[0].start_s < 2.0
```

- [ ] **Step 3: Run the slow test for real**

Run: `pytest -m slow -v`
Expected: 1 PASSED (first run downloads the faster-whisper `base` model, ~150 MB). Also verify `pytest` (default) still DESELECTS it.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_fixture.sh tests/test_whisper_smoke.py tests/fixtures/spoken_30s.wav
git commit -m "test: add real-whisper smoke test with generated audio fixture"
```

---

### Task 18: README rewrite + final verification

**Files:**
- Modify: `README.md` (full rewrite), `.env.example`

**Interfaces:**
- Consumes: everything — this task documents the finished phase 1.
- Produces: accurate top-level docs.

- [ ] **Step 1: Rewrite `.env.example`**

```
# Cloud synthesis providers (either or both; router falls back across them)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

- [ ] **Step 2: Rewrite `README.md`**

```markdown
# Video Intelligence

Agentic video analysis: a multi-agent pipeline that turns YouTube videos and
local recordings into structured reports — summary, timestamped chapters, key
quotes, and action items — with cost-aware model routing and per-stage tracing.

## Architecture

```
Ingestor ──▶ Transcriber ──▶ Chapterizer ──▶ Synthesizer
(yt-dlp,     (faster-        (small local    (frontier model:
 captions)    whisper)        model/Ollama)   Claude / GPT)
```

- **Core library** `src/video_intelligence/` — agents, model router, tracing.
  Zero web dependencies; the FastAPI app is a thin adapter.
- **Model router** — `config/models.yaml` maps each task to candidate models
  per quality tier (`cheap | balanced | best`); the router checks availability,
  falls back down the list, and records every decision as a trace span.
- **Captions first** — videos with YouTube captions skip Whisper entirely
  (override with `force_whisper`).
- **Tracing** — every model call records model, tokens, cost, latency, and
  fallback provenance to SQLite; the UI shows the full cost breakdown per job.

## Requirements

- Python 3.12, Node 20+, `ffmpeg` on PATH
- Optional: [Ollama](https://ollama.com) running locally (free chaptering)
- Optional: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in `.env` (synthesis)

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API key(s)

# backend
uvicorn src.api.main:app --port 8000

# frontend (second terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173, paste a YouTube URL, pick a quality tier, and
watch the pipeline stages stream in.

## Configuration

All model choices live in `config/models.yaml` — candidates per task and
quality tier, plus per-model pricing used for cost tracing. Whisper model
size is set there too (`transcription.whisper_model`).

## Tests

```bash
pytest              # fast suite (all fakes, no network)
pytest -m slow      # real faster-whisper smoke test (generate fixture first:
                    #   ./scripts/make_fixture.sh)
cd frontend && npm test
```

## Roadmap

- **Phase 2** — MCP server exposing `analyze_video` / `extract_chapters`
- **Phase 3** — Visual agent: slide/code/chart detection + OCR
- **Phase 4** — Fact-checker: claims vs. web search
- **Phase 5** — Live streams: rolling summaries over the SSE event channel

Design docs live in `docs/superpowers/specs/`.
```

- [ ] **Step 3: Full verification**

```bash
pytest -v
cd frontend && npm test && npm run build && cd ..
```
Expected: all Python tests pass; frontend tests pass; build clean.

- [ ] **Step 4: Commit**

```bash
git add README.md .env.example
git commit -m "docs: rewrite README for the agentic pipeline architecture"
```

---

## Self-Review Notes

- **Spec coverage:** pipeline agents (Tasks 9–12), hand-rolled orchestration + degraded policy (13), captions-first + `force_whisper` (9), model router + YAML + pricing (7), providers Ollama/OpenAI/Anthropic (5–6), tracing SQLite + trace endpoint (3, 14), FastAPI + SSE + BackgroundTasks (14), React SPA (15–16), map-reduce chunking (11–12), LED/Streamlit retirement + evaluation kept unwired (1), whisper smoke test (17), README (18). ✓
- **Known judgment call:** Task 1 also deletes `src/data`, `src/monitoring`, `src/training` — they are dependencies of deleted code only (verified by grep in Step 1). `src/evaluation` is kept per spec.
- **Model IDs in `config/models.yaml` are illustrative** — the implementer must verify current IDs/prices (see claude-api skill / provider docs) at Task 7.
