# Visual Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, non-essential Visual agent that samples video frames, OCRs on-screen content (slides/code/charts) with an optional vision-LLM escalation, and folds the result into the `AnalysisReport` and the Synthesizer prompt.

**Architecture:** One new core agent (`Visualizer`) between the Chapterizer and Synthesizer, gated by a new `JobOptions.analyze_visuals` flag. The Ingestor optionally fetches low-res video; a frame-sampling/OCR helper module feeds the agent through injectable seams (fakes in the default suite). A new optional `complete_vision` provider method + `visual_description` router task power the escalation. Output rides the existing Pydantic contract out to MCP, FastAPI, and the SPA.

**Tech Stack:** Python 3.12 (StrEnum, `X | None`), Pydantic v2, asyncio, ffmpeg (scene detection), RapidOCR (onnxruntime), pytest + pytest-asyncio, React/Vite/TS/Tailwind, Vitest.

## Global Constraints

- Model IDs live ONLY in `config/models.yaml`, never in code (per phase-1).
- New agent is `essential = False` — failures degrade the report, never fail the job.
- When `analyze_visuals` is `False`, pipeline behavior is byte-for-byte unchanged (caption cost win preserved).
- Default test suite uses fakes: no network, no real ffmpeg, no OCR-model download. Real ffmpeg/OCR only under `@pytest.mark.slow`.
- Vision escalation runs ONLY when `quality == best` AND a vision provider is available; otherwise OCR text stands alone with no degradation.
- Follow existing patterns: agents subclass `Agent`, use `Router.complete`, tests use `tests/fakes.py`.
- New dependency this phase: `rapidocr-onnxruntime` (ffmpeg and yt-dlp already required).
- All new Python uses `from __future__ import annotations` (match existing files).

---

### Task 1: Schemas — VisualArtifact + new fields

**Files:**
- Modify: `src/video_intelligence/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `class VisualKind(StrEnum)` with members `SLIDE="slide"`, `CODE="code"`, `CHART="chart"`, `OTHER="other"`.
  - `class VisualArtifact(BaseModel)`: `timestamp_s: float`, `kind: VisualKind`, `text: str = ""`, `description: str | None = None`, `frame_path: str | None = None`.
  - `AnalysisReport.visual_highlights: list[VisualArtifact] = Field(default_factory=list)`.
  - `JobOptions.analyze_visuals: bool = False`.
  - `PipelineContext.video_path: str | None = None`, `PipelineContext.visual_artifacts: list[VisualArtifact] | None = None`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas.py`:

```python
from src.video_intelligence.schemas import (
    AnalysisReport, JobOptions, PipelineContext, SourceKind, VideoSource,
    VisualArtifact, VisualKind,
)


def test_visual_artifact_defaults():
    va = VisualArtifact(timestamp_s=12.0, kind=VisualKind.SLIDE)
    assert va.text == ""
    assert va.description is None
    assert va.frame_path is None
    assert va.kind == "slide"


def test_report_visual_highlights_default_empty():
    r = AnalysisReport(summary="s", language="en", trace_id="t")
    assert r.visual_highlights == []


def test_job_options_analyze_visuals_defaults_false():
    assert JobOptions().analyze_visuals is False


def test_pipeline_context_visual_fields_default_none():
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="u"), options=JobOptions())
    assert ctx.video_path is None
    assert ctx.visual_artifacts is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -k "visual or analyze_visuals" -v`
Expected: FAIL with `ImportError` / `cannot import name 'VisualArtifact'`.

- [ ] **Step 3: Write minimal implementation**

In `src/video_intelligence/schemas.py`, add after the `TranscriptOrigin` enum:

```python
class VisualKind(StrEnum):
    SLIDE = "slide"
    CODE = "code"
    CHART = "chart"
    OTHER = "other"
```

Add after the `Chapter` model:

```python
class VisualArtifact(BaseModel):
    timestamp_s: float
    kind: VisualKind
    text: str = ""
    description: str | None = None
    frame_path: str | None = None
```

In `AnalysisReport`, add field:

```python
    visual_highlights: list[VisualArtifact] = Field(default_factory=list)
```

In `JobOptions`, add field:

```python
    analyze_visuals: bool = False
```

In `PipelineContext`, add fields (next to `audio_path`):

```python
    video_path: str | None = None
    visual_artifacts: list[VisualArtifact] | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (existing + new tests).

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/schemas.py tests/test_schemas.py
git commit -m "feat: add VisualArtifact schema and visual pipeline fields"
```

---

### Task 2: Ingestor — optional low-res video fetch

**Files:**
- Modify: `src/video_intelligence/agents/ingestor.py`
- Test: `tests/test_ingestor.py`

**Interfaces:**
- Consumes: `PipelineContext` (Task 1), `JobOptions.analyze_visuals`, `ctx.video_path`.
- Produces:
  - `default_video_downloader(url: str, workdir: Path) -> Path` (module function).
  - `Ingestor.__init__` gains keyword arg `video_downloader=default_video_downloader`.
  - When `ctx.options.analyze_visuals` is true: YouTube sets `ctx.video_path` to a downloaded low-res video; local files set `ctx.video_path = ctx.source.path`. Video-download failure logs a warning and leaves `video_path` None (never fails the job).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ingestor.py`. First extend `make_ingestor` to inject a fake video downloader (replace the existing helper's `Ingestor(...)` construction):

```python
def make_ingestor(tmp_path, captions=RAW_CAPTIONS, downloads=None):
    calls = {"downloaded": False, "video_downloaded": False}

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

    def video_downloader(url, workdir: Path):
        calls["video_downloaded"] = True
        out = workdir / "v.mp4"
        out.write_bytes(b"fakevideo")
        return out

    ing = Ingestor(workdir=tmp_path, metadata_fetcher=metadata_fetcher,
                   caption_fetcher=caption_fetcher, audio_downloader=audio_downloader,
                   audio_extractor=audio_extractor, video_downloader=video_downloader)
    return ing, calls
```

Then add tests:

```python
async def test_visuals_off_does_not_download_video(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"))
    ctx = await ing.run(ctx)
    assert calls["video_downloaded"] is False
    assert ctx.video_path is None


async def test_visuals_on_downloads_video_even_with_captions(tmp_path):
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
                  analyze_visuals=True)
    ctx = await ing.run(ctx)
    assert ctx.transcript is not None          # captions still used
    assert calls["video_downloaded"] is True
    assert ctx.video_path is not None


async def test_visuals_on_local_file_sets_video_path_to_source(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"data")
    ing, calls = make_ingestor(tmp_path)
    ctx = ctx_for(VideoSource(kind=SourceKind.LOCAL_FILE, path=str(src)),
                  analyze_visuals=True)
    ctx = await ing.run(ctx)
    assert ctx.video_path == str(src)


async def test_video_download_failure_is_non_fatal(tmp_path):
    def boom(url, workdir):
        raise RuntimeError("network down")
    ing, calls = make_ingestor(tmp_path)
    ing._video_downloader = boom
    ctx = ctx_for(VideoSource(kind=SourceKind.YOUTUBE, url="https://youtu.be/dQw4w9WgXcQ"),
                  analyze_visuals=True)
    ctx = await ing.run(ctx)          # must not raise
    assert ctx.video_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingestor.py -k visuals -v`
Expected: FAIL — `Ingestor.__init__` has no `video_downloader` kwarg (`TypeError`).

- [ ] **Step 3: Write minimal implementation**

In `ingestor.py`, add a module-level default after `default_audio_extractor`:

```python
def default_video_downloader(url: str, workdir: Path) -> Path:
    import yt_dlp
    stem = uuid.uuid4().hex
    with yt_dlp.YoutubeDL({
        "quiet": True,
        # low-res is enough for OCR; keep bandwidth/latency down
        "format": "worst[height>=360]/worst",
        "outtmpl": str(workdir / f"{stem}.%(ext)s"),
    }) as ydl:
        ydl.download([url])
    files = list(workdir.glob(f"{stem}.*"))
    if not files:
        raise IngestError(f"yt-dlp produced no video file for {url!r}")
    return files[0]
```

Add the constructor arg (append to the signature and body of `Ingestor.__init__`):

```python
    def __init__(self, workdir: str | Path = "data/work",
                 metadata_fetcher=default_metadata_fetcher,
                 caption_fetcher=default_caption_fetcher,
                 audio_downloader=default_audio_downloader,
                 audio_extractor=default_audio_extractor,
                 video_downloader=default_video_downloader):
        self._workdir = Path(workdir)
        self._metadata_fetcher = metadata_fetcher
        self._caption_fetcher = caption_fetcher
        self._audio_downloader = audio_downloader
        self._audio_extractor = audio_extractor
        self._video_downloader = video_downloader
```

Add a helper method and call it from both ingest paths. In `_ingest_youtube`, at the very end of the method (after the caption `return` block and the audio-download lines), append a call so video is fetched regardless of caption/audio path:

```python
    async def _maybe_download_video(self, ctx: PipelineContext) -> None:
        if not ctx.options.analyze_visuals:
            return
        try:
            path = await asyncio.to_thread(self._video_downloader, ctx.source.url, self._workdir)
            ctx.video_path = str(path)
        except Exception as e:
            logger.warning("video download failed for %s: %s", ctx.source.url, e)
```

Restructure `_ingest_youtube` so it always attempts video download when the flag is on. Replace the caption early-`return` with setting the transcript then falling through:

```python
    async def _ingest_youtube(self, ctx: PipelineContext) -> None:
        try:
            meta = await asyncio.to_thread(self._metadata_fetcher, ctx.source.url)
            ctx.source = ctx.source.model_copy(update=meta)
        except Exception as e:
            logger.warning("metadata fetch failed for %s: %s", ctx.source.url, e)
        have_transcript = False
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
                have_transcript = True
        if not have_transcript:
            path = await asyncio.to_thread(self._audio_downloader, ctx.source.url, self._workdir)
            ctx.audio_path = str(path)
        await self._maybe_download_video(ctx)
```

In `_ingest_local`, set the video path from the source when visuals are on (append before the method returns):

```python
        if ctx.options.analyze_visuals:
            ctx.video_path = str(src)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingestor.py -v`
Expected: PASS (existing caption/download tests still green + new ones).

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/ingestor.py tests/test_ingestor.py
git commit -m "feat: ingestor fetches low-res video when analyze_visuals is on"
```

---

### Task 3: Frame sampling + OCR + classification helpers

**Files:**
- Create: `src/video_intelligence/agents/visual_frames.py`
- Modify: `requirements.txt`
- Test: `tests/test_visual_frames.py`

**Interfaces:**
- Consumes: `VisualKind` (Task 1).
- Produces (all in `visual_frames.py`):
  - `class SampledFrame(BaseModel)`: `timestamp_s: float`, `image_path: str`.
  - `sample_scene_frames(video_path: str, workdir: str, *, scene_threshold: float, max_frames: int, min_interval_s: float) -> list[SampledFrame]` — ffmpeg default sampler.
  - `rapidocr_text(image_path: str) -> str` — default OCR.
  - `classify(text: str) -> VisualKind` — pure heuristic.
  - `normalize(text: str) -> str` and `is_near_duplicate(a: str, b: str) -> bool` — dedup helpers.
- Note: `classify` and `is_near_duplicate` are pure/text-only (unit-tested without ffmpeg/OCR). `sample_scene_frames`/`rapidocr_text` are the real-IO defaults used only under `@pytest.mark.slow` (Task 12) and production.

- [ ] **Step 1: Add the runtime dependency**

In `requirements.txt`, under the "Ingestion / transcription" group add:

```
rapidocr-onnxruntime>=1.3
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_visual_frames.py`:

```python
from src.video_intelligence.agents.visual_frames import (
    classify, is_near_duplicate, normalize,
)
from src.video_intelligence.schemas import VisualKind


def test_classify_code_by_symbol_and_indent_density():
    code = "def add(a, b):\n    return a + b\n    # sum\nx = add(1, 2)"
    assert classify(code) == VisualKind.CODE


def test_classify_slide_by_title_and_bullets():
    slide = "Roadmap 2026\n- Ship visual agent\n- Fact checker\n- Live streams"
    assert classify(slide) == VisualKind.SLIDE


def test_classify_chart_when_text_sparse():
    assert classify("42%") == VisualKind.CHART
    assert classify("") == VisualKind.CHART


def test_classify_other_for_prose():
    prose = "This is a paragraph of ordinary spoken narration shown on screen."
    assert classify(prose) == VisualKind.OTHER


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Hello   World \n") == "hello world"


def test_is_near_duplicate_true_for_same_slide():
    assert is_near_duplicate("Roadmap 2026", "roadmap 2026 ") is True


def test_is_near_duplicate_false_for_different_text():
    assert is_near_duplicate("Intro", "Deep dive into caching") is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_visual_frames.py -v`
Expected: FAIL with `ModuleNotFoundError: ...visual_frames`.

- [ ] **Step 4: Write minimal implementation**

Create `src/video_intelligence/agents/visual_frames.py`:

```python
"""Frame sampling (ffmpeg), OCR (RapidOCR), and text-based classification helpers.

The ffmpeg/OCR functions are the production defaults; agents inject fakes in the
default test suite. classify/normalize/dedup are pure and unit-tested directly.
"""
from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path

from pydantic import BaseModel

from ..schemas import VisualKind


class SampledFrame(BaseModel):
    timestamp_s: float
    image_path: str


_CODE_SYMBOLS = set("{}()[];=<>+/*_#\\|")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_near_duplicate(a: str, b: str, threshold: float = 0.9) -> bool:
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    wa, wb = set(na.split()), set(nb.split())
    overlap = len(wa & wb) / max(len(wa | wb), 1)
    return overlap >= threshold


def classify(text: str) -> VisualKind:
    stripped = text.strip()
    if len(stripped) < 15:
        return VisualKind.CHART
    lines = [ln for ln in text.splitlines() if ln.strip()]
    symbol_ratio = sum(c in _CODE_SYMBOLS for c in stripped) / max(len(stripped), 1)
    indented = sum(1 for ln in text.splitlines() if ln[:1] in (" ", "\t"))
    if symbol_ratio > 0.08 or indented >= 2:
        return VisualKind.CODE
    bullets = sum(1 for ln in lines if ln.lstrip()[:1] in ("-", "*", "•"))
    if bullets >= 2 or (len(lines) >= 2 and all(len(ln) < 60 for ln in lines)):
        return VisualKind.SLIDE
    return VisualKind.OTHER


def sample_scene_frames(video_path: str, workdir: str, *, scene_threshold: float,
                        max_frames: int, min_interval_s: float) -> list[SampledFrame]:
    out_dir = Path(workdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    pattern = str(out_dir / f"{stem}-%04d.jpg")
    # emit a frame at each scene cut, timestamped via showinfo
    vf = f"select='gt(scene,{scene_threshold})',showinfo"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-vsync", "vfr", pattern],
        capture_output=True, text=True,
    )
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
    files = sorted(out_dir.glob(f"{stem}-*.jpg"))
    frames: list[SampledFrame] = []
    last_t = -1e9
    for i, f in enumerate(files):
        t = times[i] if i < len(times) else float(i)
        if t - last_t < min_interval_s:
            continue
        frames.append(SampledFrame(timestamp_s=t, image_path=str(f)))
        last_t = t
        if len(frames) >= max_frames:
            break
    return frames


def rapidocr_text(image_path: str) -> str:
    from rapidocr_onnxruntime import RapidOCR
    engine = _get_engine()
    result, _ = engine(image_path)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_visual_frames.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/video_intelligence/agents/visual_frames.py tests/test_visual_frames.py requirements.txt
git commit -m "feat: add frame-sampling, OCR, and classification helpers"
```

---

### Task 4: Visualizer agent (OCR path, no vision yet)

**Files:**
- Create: `src/video_intelligence/agents/visualizer.py`
- Test: `tests/test_visualizer.py`

**Interfaces:**
- Consumes: `SampledFrame`, `classify`, `is_near_duplicate` (Task 3); `Router` (unused until Task 6 — accept and store it); `PipelineContext`, `VisualArtifact`, `VisualKind` (Task 1).
- Produces:
  - `class Visualizer(Agent)` with `name = "visual"`, `essential = False`.
  - `__init__(self, router, *, frame_sampler=..., ocr=..., scene_threshold=0.4, max_frames=24, min_interval_s=8.0, workdir="data/work")` where `frame_sampler(video_path, workdir, scene_threshold=, max_frames=, min_interval_s=) -> list[SampledFrame]` and `ocr(image_path) -> str`.
  - `run(ctx)`: no-op passthrough when `not ctx.options.analyze_visuals`; raises `ValueError("visual requires a video")` when visuals on but `ctx.video_path is None`; otherwise sets `ctx.visual_artifacts: list[VisualArtifact]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualizer.py`:

```python
import pytest

from src.video_intelligence.agents.visualizer import Visualizer
from src.video_intelligence.agents.visual_frames import SampledFrame
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, VideoSource, VisualKind,
)


def ctx_for(video_path=None, **opts):
    return PipelineContext(
        source=VideoSource(kind=SourceKind.YOUTUBE, url="u"),
        options=JobOptions(**opts), video_path=video_path)


def make(frames, ocr_map):
    def sampler(video_path, workdir, *, scene_threshold, max_frames, min_interval_s):
        return frames

    def ocr(image_path):
        return ocr_map.get(image_path, "")

    return Visualizer(router=None, frame_sampler=sampler, ocr=ocr)


async def test_skips_when_analyze_visuals_off():
    vis = make([], {})
    ctx = await make([], {}).run(ctx_for(video_path="v.mp4", analyze_visuals=False))
    assert ctx.visual_artifacts is None


async def test_raises_when_visuals_on_but_no_video():
    vis = make([], {})
    with pytest.raises(ValueError, match="requires a video"):
        await vis.run(ctx_for(video_path=None, analyze_visuals=True))


async def test_produces_classified_artifacts():
    frames = [
        SampledFrame(timestamp_s=5.0, image_path="a.jpg"),
        SampledFrame(timestamp_s=20.0, image_path="b.jpg"),
    ]
    ocr_map = {"a.jpg": "def f():\n    return 1", "b.jpg": "Roadmap\n- one\n- two"}
    vis = make(frames, ocr_map)
    ctx = await vis.run(ctx_for(video_path="v.mp4", analyze_visuals=True))
    kinds = [a.kind for a in ctx.visual_artifacts]
    assert VisualKind.CODE in kinds and VisualKind.SLIDE in kinds
    assert ctx.visual_artifacts[0].timestamp_s == 5.0
    assert ctx.visual_artifacts[0].frame_path == "a.jpg"


async def test_dedups_consecutive_near_identical_frames():
    frames = [
        SampledFrame(timestamp_s=5.0, image_path="a.jpg"),
        SampledFrame(timestamp_s=6.0, image_path="b.jpg"),
    ]
    ocr_map = {"a.jpg": "Roadmap 2026", "b.jpg": "roadmap 2026 "}
    vis = make(frames, ocr_map)
    ctx = await vis.run(ctx_for(video_path="v.mp4", analyze_visuals=True))
    assert len(ctx.visual_artifacts) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualizer.py -v`
Expected: FAIL with `ModuleNotFoundError: ...visualizer`.

- [ ] **Step 3: Write minimal implementation**

Create `src/video_intelligence/agents/visualizer.py`:

```python
"""Visual agent: sample frames -> OCR -> classify -> VisualArtifacts (non-essential)."""
from __future__ import annotations

from ..schemas import PipelineContext, VisualArtifact
from .base import Agent
from .visual_frames import classify, is_near_duplicate, rapidocr_text, sample_scene_frames


class Visualizer(Agent):
    name = "visual"
    essential = False

    def __init__(self, router, *, frame_sampler=sample_scene_frames, ocr=rapidocr_text,
                 scene_threshold: float = 0.4, max_frames: int = 24,
                 min_interval_s: float = 8.0, workdir: str = "data/work"):
        self._router = router
        self._frame_sampler = frame_sampler
        self._ocr = ocr
        self._scene_threshold = scene_threshold
        self._max_frames = max_frames
        self._min_interval_s = min_interval_s
        self._workdir = workdir

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if not ctx.options.analyze_visuals:
            return ctx
        if ctx.video_path is None:
            raise ValueError("visual requires a video")
        frames = self._frame_sampler(
            ctx.video_path, self._workdir, scene_threshold=self._scene_threshold,
            max_frames=self._max_frames, min_interval_s=self._min_interval_s)
        artifacts: list[VisualArtifact] = []
        last_text = ""
        for frame in frames:
            text = self._ocr(frame.image_path)
            if text and last_text and is_near_duplicate(text, last_text):
                continue
            last_text = text
            artifacts.append(VisualArtifact(
                timestamp_s=frame.timestamp_s, kind=classify(text),
                text=text, frame_path=frame.image_path))
        ctx.visual_artifacts = artifacts
        return ctx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/visualizer.py tests/test_visualizer.py
git commit -m "feat: add Visualizer agent OCR path"
```

---

### Task 5: Provider vision interface + router `complete_vision`

**Files:**
- Modify: `src/video_intelligence/models/providers/base.py`
- Modify: `src/video_intelligence/models/router.py`
- Modify: `config/models.yaml`
- Test: `tests/test_router.py`, `tests/test_provider_base.py`

**Interfaces:**
- Consumes: existing `Router`, `Provider`, `Usage`, `TraceSpan`.
- Produces:
  - `class NotSupported(ProviderError)` in `base.py`.
  - `Provider.complete_vision(self, model: str, prompt: str, images: list[bytes], schema: type[T]) -> tuple[T, Usage]` — concrete default method (NOT abstract) raising `NotSupported`.
  - `Router.complete_vision(self, *, task: str, quality: QualityPreference, prompt: str, images: list[bytes], schema: type[T], trace_id: str, stage: str) -> T` — mirrors `complete`, calls `provider.complete_vision`, treats `NotSupported`/unavailable as fallthrough, traces under `stage`.
  - `config/models.yaml`: `tasks.visual_description.best` candidate list + top-level `visual` sampling block.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_provider_base.py`:

```python
import pytest

from src.video_intelligence.models.providers.base import NotSupported, Provider, Usage


class _MinimalProvider(Provider):
    name = "min"
    async def is_available(self):
        return True
    async def complete(self, model, prompt, schema):
        raise NotImplementedError


async def test_complete_vision_defaults_to_not_supported():
    p = _MinimalProvider()
    with pytest.raises(NotSupported):
        await p.complete_vision("m", "p", [b"img"], Usage)
```

Add to `tests/test_router.py` (reuse its existing config/provider fakes; add a fake with vision):

```python
import pytest
from pydantic import BaseModel

from src.video_intelligence.models.providers.base import NotSupported, Usage
from src.video_intelligence.models.router import Router, RouterError
from src.video_intelligence.schemas import QualityPreference
from src.video_intelligence.tracing import TraceStore
from tests.fakes import FakeProvider


class VDesc(BaseModel):
    description: str


VISION_CONFIG = {"tasks": {"visual_description": {"best": ["fake/vision-model"]}}}


class VisionFake(FakeProvider):
    def __init__(self, name="fake", supports=True):
        super().__init__(name)
        self._supports = supports
        self.vision_calls = []
    async def complete_vision(self, model, prompt, images, schema):
        if not self._supports:
            raise NotSupported("no vision")
        self.vision_calls.append({"model": model, "images": images})
        return self._queue.pop(0), Usage(tokens_in=10, tokens_out=5)


async def test_router_complete_vision_returns_parsed(tmp_path):
    fake = VisionFake()
    fake.enqueue(VDesc(description="a bar chart"))
    router = Router(VISION_CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    out = await router.complete_vision(
        task="visual_description", quality=QualityPreference.BEST, prompt="describe",
        images=[b"img"], schema=VDesc, trace_id="t", stage="visual")
    assert out.description == "a bar chart"
    assert fake.vision_calls[0]["model"] == "vision-model"


async def test_router_complete_vision_raises_when_all_unsupported(tmp_path):
    fake = VisionFake(supports=False)
    router = Router(VISION_CONFIG, {"fake": fake}, TraceStore(tmp_path / "t.db"))
    with pytest.raises(RouterError):
        await router.complete_vision(
            task="visual_description", quality=QualityPreference.BEST, prompt="d",
            images=[b"img"], schema=VDesc, trace_id="t", stage="visual")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider_base.py::test_complete_vision_defaults_to_not_supported tests/test_router.py -k complete_vision -v`
Expected: FAIL — `NotSupported` import error / `Router` has no `complete_vision`.

- [ ] **Step 3: Write minimal implementation**

In `base.py`, add after `ProviderError`:

```python
class NotSupported(ProviderError):
    """The provider does not implement this capability (e.g. vision)."""
```

Add a concrete default method to `Provider` (after the abstract `complete`):

```python
    async def complete_vision(self, model: str, prompt: str, images: list[bytes],
                              schema: type[T]) -> tuple[T, Usage]:
        raise NotSupported(f"{self.name} does not support vision")
```

In `router.py`, import `NotSupported`:

```python
from .providers.base import NotSupported, Provider, ProviderError, Usage
```

Add the method (mirrors `complete`; `NotSupported` is caught as a fallthrough):

```python
    async def complete_vision(self, *, task: str, quality: QualityPreference, prompt: str,
                              images: list[bytes], schema: type[T], trace_id: str,
                              stage: str) -> T:
        fallback_from: str | None = None
        errors: list[str] = []
        for candidate in self.candidates(task, quality):
            provider_name, model = candidate.split("/", 1)
            provider = self._providers.get(provider_name)
            if provider is None or not await provider.is_available():
                errors.append(f"{candidate}: unavailable")
                fallback_from = candidate
                continue
            start = time.monotonic()
            try:
                parsed, usage = await provider.complete_vision(model, prompt, images, schema)
            except ProviderError as e:  # includes NotSupported
                errors.append(f"{candidate}: {e}")
                fallback_from = candidate
                continue
            self._store.add_span(trace_id, TraceSpan(
                stage=stage, model_used=candidate, tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out, cost_usd=self._cost(candidate, usage),
                latency_ms=int((time.monotonic() - start) * 1000), status="ok",
                fallback_from=fallback_from))
            return parsed
        self._store.add_span(trace_id, TraceSpan(stage=stage, model_used="none", status="error"))
        raise RouterError(f"all vision candidates failed for task={task}: {'; '.join(errors)}")
```

In `config/models.yaml`, add under `tasks:` (a vision-capable candidate list; IDs illustrative, pin at implementation time):

```yaml
  visual_description:
    best:     ["anthropic/claude-sonnet-5", "openai/gpt-4o"]
```

And add a top-level block (after the `pricing:` block):

```yaml
visual:              # frame-sampling defaults for the Visual agent
  scene_threshold: 0.4
  max_frames: 24
  min_interval_s: 8
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider_base.py tests/test_router.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/models/providers/base.py src/video_intelligence/models/router.py config/models.yaml tests/test_router.py tests/test_provider_base.py
git commit -m "feat: add optional vision provider method and router.complete_vision"
```

---

### Task 6: Visualizer vision escalation (quality=best)

**Files:**
- Modify: `src/video_intelligence/agents/visualizer.py`
- Test: `tests/test_visualizer.py`

**Interfaces:**
- Consumes: `Router.complete_vision` (Task 5), `NotSupported`/`RouterError`, `QualityPreference`.
- Produces: escalation inside `Visualizer.run` — for `CHART`/text-sparse artifacts, when `ctx.options.quality == QualityPreference.BEST`, call `router.complete_vision(task="visual_description", ...)` and set `artifact.description`. Router failure (no vision provider) is swallowed; OCR text stands, no degradation. Add nested `VisionDescription(BaseModel)` with `description: str`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_visualizer.py`:

```python
from src.video_intelligence.models.router import Router
from src.video_intelligence.tracing import TraceStore
from src.video_intelligence.agents.visualizer import VisionDescription
from tests.fakes import FakeProvider


class _VisionFake(FakeProvider):
    async def complete_vision(self, model, prompt, images, schema):
        from src.video_intelligence.models.providers.base import Usage
        return self._queue.pop(0), Usage()


def make_with_router(frames, ocr_map, quality, provider):
    def sampler(video_path, workdir, *, scene_threshold, max_frames, min_interval_s):
        return frames
    def ocr(image_path):
        return ocr_map.get(image_path, "")
    import tempfile
    cfg = {"tasks": {"visual_description": {"best": ["fake/v"]}}}
    router = Router(cfg, {"fake": provider}, TraceStore(tempfile.mktemp()))
    vis = Visualizer(router=router, frame_sampler=sampler, ocr=ocr)
    return vis


async def test_chart_frame_escalates_on_best(tmp_path):
    frames = [SampledFrame(timestamp_s=5.0, image_path=str(tmp_path / "c.jpg"))]
    (tmp_path / "c.jpg").write_bytes(b"img")
    provider = _VisionFake()
    provider.enqueue(VisionDescription(description="a bar chart of revenue"))
    vis = make_with_router(frames, {str(tmp_path / "c.jpg"): "42%"},
                           JobOptions().quality, provider)
    ctx = await vis.run(ctx_for(video_path="v.mp4", analyze_visuals=True, quality="best"))
    assert ctx.visual_artifacts[0].kind == VisualKind.CHART
    assert ctx.visual_artifacts[0].description == "a bar chart of revenue"


async def test_no_escalation_below_best(tmp_path):
    frames = [SampledFrame(timestamp_s=5.0, image_path=str(tmp_path / "c.jpg"))]
    (tmp_path / "c.jpg").write_bytes(b"img")
    provider = _VisionFake()
    vis = make_with_router(frames, {str(tmp_path / "c.jpg"): "42%"},
                           JobOptions().quality, provider)
    ctx = await vis.run(ctx_for(video_path="v.mp4", analyze_visuals=True, quality="balanced"))
    assert ctx.visual_artifacts[0].description is None


async def test_escalation_failure_keeps_ocr_text(tmp_path):
    frames = [SampledFrame(timestamp_s=5.0, image_path=str(tmp_path / "c.jpg"))]
    (tmp_path / "c.jpg").write_bytes(b"img")
    provider = FakeProvider("fake")   # base complete_vision raises NotSupported
    vis = make_with_router(frames, {str(tmp_path / "c.jpg"): "42%"},
                           JobOptions().quality, provider)
    ctx = await vis.run(ctx_for(video_path="v.mp4", analyze_visuals=True, quality="best"))
    assert ctx.visual_artifacts[0].description is None    # no crash, OCR stands
    assert ctx.visual_artifacts[0].text == "42%"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visualizer.py -k "escalat or best" -v`
Expected: FAIL — `cannot import name 'VisionDescription'`.

- [ ] **Step 3: Write minimal implementation**

In `visualizer.py`, add imports and the model near the top:

```python
from pydantic import BaseModel

from ..models.router import RouterError
from ..schemas import PipelineContext, QualityPreference, VisualArtifact, VisualKind


class VisionDescription(BaseModel):
    description: str


_VISION_PROMPT = (
    "Describe the meaningful visual content of this video frame in one or two "
    "sentences (a chart, diagram, or figure). Return ONLY JSON: "
    '{"description": "<string>"}'
)
```

At the end of `run`, before `ctx.visual_artifacts = artifacts` return, add escalation over the collected artifacts:

```python
        if ctx.options.quality == QualityPreference.BEST and self._router is not None:
            for art in artifacts:
                if art.kind != VisualKind.CHART:
                    continue
                try:
                    with open(art.frame_path, "rb") as fh:
                        image = fh.read()
                    result = await self._router.complete_vision(
                        task="visual_description", quality=ctx.options.quality,
                        prompt=_VISION_PROMPT, images=[image], schema=VisionDescription,
                        trace_id=ctx.trace_id, stage=self.name)
                    art.description = result.description
                except (RouterError, OSError):
                    continue  # vision unavailable/unreadable: OCR text stands
        ctx.visual_artifacts = artifacts
        return ctx
```

Remove the now-duplicated `ctx.visual_artifacts = artifacts` / `return ctx` lines from Task 4's version so they appear once, after the escalation block.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS (all Task 4 + Task 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/visualizer.py tests/test_visualizer.py
git commit -m "feat: add vision escalation to Visualizer on best quality"
```

---

### Task 7: Wire Visualizer into the pipeline

**Files:**
- Modify: `src/video_intelligence/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Visualizer` (Tasks 4/6), `Router`.
- Produces: `build_pipeline` inserts `Visualizer(router, scene_threshold=, max_frames=, min_interval_s=, workdir=)` between `Chapterizer(router)` and `Synthesizer(router)`, reading params from `config["visual"]` (with the Task 5 defaults as fallback).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py` (this file already builds pipelines with fake agents — follow its existing style; here is a standalone test using a fake Visualizer-free ordering check on `build_pipeline`):

```python
def test_build_pipeline_includes_visual_stage(tmp_path):
    from src.video_intelligence.pipeline import build_pipeline
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "transcription: {whisper_model: base}\n"
        "tasks: {chaptering: {balanced: []}, synthesis: {balanced: []}}\n"
        "visual: {scene_threshold: 0.5, max_frames: 10, min_interval_s: 4}\n"
    )
    pipe = build_pipeline(config_path=str(cfg), db_path=str(tmp_path / "t.db"),
                          workdir=str(tmp_path / "w"))
    names = [a.name for a in pipe._agents]
    assert names == ["ingest", "transcribe", "chapterize", "visual", "synthesize"]
    visual = pipe._agents[3]
    assert visual._max_frames == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_build_pipeline_includes_visual_stage -v`
Expected: FAIL — `visual` not in agent names.

- [ ] **Step 3: Write minimal implementation**

In `pipeline.py` `build_pipeline`, add the import and wire the agent. Add to the imports inside the function:

```python
    from .agents.visualizer import Visualizer
```

Read visual params and insert the agent:

```python
    visual_cfg = config.get("visual", {})
    return Pipeline(
        [
            Ingestor(workdir=workdir),
            Transcriber(model_name=whisper_model),
            Chapterizer(router),
            Visualizer(
                router,
                scene_threshold=visual_cfg.get("scene_threshold", 0.4),
                max_frames=visual_cfg.get("max_frames", 24),
                min_interval_s=visual_cfg.get("min_interval_s", 8),
                workdir=workdir,
            ),
            Synthesizer(router),
        ],
        on_event=on_event,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire Visualizer into build_pipeline"
```

---

### Task 8: Synthesizer — inject visual highlights

**Files:**
- Modify: `src/video_intelligence/agents/synthesizer.py`
- Test: `tests/test_synthesizer.py`

**Interfaces:**
- Consumes: `ctx.visual_artifacts` (Task 1), `_ts` from `prompting`.
- Produces: when `ctx.visual_artifacts` is non-empty, a `<<VISUALS>>` block of `[M:SS] <kind>: <text|description>` lines is injected into `SYNTH_PROMPT`; `report.visual_highlights = ctx.visual_artifacts or []`. When empty/None, the prompt gains a `none` placeholder and behavior is otherwise unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synthesizer.py`:

```python
from src.video_intelligence.schemas import VisualArtifact, VisualKind


async def test_visual_highlights_injected_and_attached(tmp_path):
    synth, fake = make(tmp_path)
    fake.enqueue(RESULT)
    ctx = ctx_with_transcript(10)
    ctx.visual_artifacts = [
        VisualArtifact(timestamp_s=30.0, kind=VisualKind.SLIDE, text="Roadmap 2026"),
        VisualArtifact(timestamp_s=90.0, kind=VisualKind.CHART, text="",
                       description="a revenue bar chart"),
    ]
    ctx = await synth.run(ctx)
    assert "Roadmap 2026" in fake.calls[0]["prompt"]
    assert "a revenue bar chart" in fake.calls[0]["prompt"]
    assert len(ctx.report.visual_highlights) == 2


async def test_no_visuals_leaves_highlights_empty(tmp_path):
    synth, fake = make(tmp_path)
    fake.enqueue(RESULT)
    ctx = await synth.run(ctx_with_transcript(10))
    assert ctx.report.visual_highlights == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthesizer.py -k visual -v`
Expected: FAIL — highlights empty / text not in prompt.

- [ ] **Step 3: Write minimal implementation**

In `synthesizer.py`, import the timestamp helper:

```python
from .prompting import chunk_text, transcript_lines, _ts
```

Add a `<<VISUALS>>` placeholder to `SYNTH_PROMPT` — insert a line after the chapters line:

```python
Chapters (may be empty): <<CHAPTERS>>
On-screen visuals (may be empty): <<VISUALS>>
```

Add a helper and wire it in `run` (build the string before `prompt = (...)`):

```python
    @staticmethod
    def _visual_lines(ctx: PipelineContext) -> str:
        arts = ctx.visual_artifacts or []
        if not arts:
            return "none"
        return "\n".join(
            f"[{_ts(a.timestamp_s)}] {a.kind.value}: {a.description or a.text}".rstrip()
            for a in arts
        )
```

In `run`, add the replacement to the prompt chain:

```python
        prompt = (SYNTH_PROMPT
                  .replace("<<TITLE>>", ctx.source.title or "unknown")
                  .replace("<<CHAPTERS>>", chapters_str)
                  .replace("<<VISUALS>>", self._visual_lines(ctx))
                  .replace("<<TRANSCRIPT>>", text)
                  .replace("<<LANGUAGE>>", ctx.options.language))
```

And set the field on the report:

```python
        ctx.report = AnalysisReport(
            summary=result.summary,
            chapters=ctx.chapters or [],
            key_quotes=result.key_quotes,
            action_items=result.action_items,
            language=ctx.options.language,
            trace_id=ctx.trace_id,
            degraded_stages=list(ctx.degraded_stages),
            visual_highlights=ctx.visual_artifacts or [],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthesizer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/agents/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: inject visual highlights into synthesis and report"
```

---

### Task 9: MCP adapter — analyze_visuals argument

**Files:**
- Modify: `src/mcp_server/runtime.py`
- Modify: `src/mcp_server/server.py`
- Test: `tests/test_mcp_runtime.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `JobOptions.analyze_visuals` (Task 1).
- Produces:
  - `Runtime.analyze(..., analyze_visuals: bool = False)` threads the flag into `JobOptions`.
  - `Runtime.extract_chapters(..., analyze_visuals: bool = False)`.
  - `analyze_video` / `extract_chapters` MCP tools expose `analyze_visuals: bool = False`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mcp_runtime.py` (follow its existing fake-pipeline setup; assert the option reaches the pipeline). Minimal check on threading:

```python
async def test_analyze_threads_analyze_visuals(tmp_path):
    # uses the module's existing helper to build a Runtime with a capturing pipeline
    captured = {}

    async def fake_run(source, options):
        captured["analyze_visuals"] = options.analyze_visuals
        from src.video_intelligence.schemas import AnalysisReport
        return AnalysisReport(summary="s", language="en", trace_id="t")

    runtime = make_runtime(tmp_path, run=fake_run)   # see existing helper in this file
    await runtime.analyze(url="https://youtu.be/x", analyze_visuals=True)
    assert captured["analyze_visuals"] is True
```

If `test_mcp_runtime.py` lacks a reusable `make_runtime(..., run=...)` seam, use the same construction the file's other tests already use (a fake pipeline with a settable `run`); mirror that pattern rather than inventing a new one.

Add to `tests/test_mcp_server.py` a schema check:

```python
def test_analyze_video_exposes_analyze_visuals(server):
    # `server` fixture builds the FastMCP instance used by other tests in this file
    tool = {t.name: t for t in server._tool_manager.list_tools()}["analyze_video"]
    assert "analyze_visuals" in tool.inputSchema["properties"]
```

Match the existing tool-introspection style already used in `test_mcp_server.py` (it already asserts on generated schemas); reuse that fixture/accessor instead of the illustrative `_tool_manager` above if the file uses a different one.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mcp_runtime.py -k analyze_visuals tests/test_mcp_server.py -k analyze_visuals -v`
Expected: FAIL — unexpected keyword `analyze_visuals`.

- [ ] **Step 3: Write minimal implementation**

In `runtime.py` `analyze`, add the parameter and thread it into `JobOptions`:

```python
    async def analyze(self, url: str, quality: str = "balanced",
                      language: str = "en", force_whisper: bool = False,
                      analyze_visuals: bool = False,
                      async_: bool = False,
                      on_event: EventCallback | None = None) -> dict:
```

```python
        options = JobOptions(language=language,
                             quality=quality_pref,
                             force_whisper=force_whisper,
                             analyze_visuals=analyze_visuals)
```

In `extract_chapters`, add `analyze_visuals: bool = False` and pass it through the `self.analyze(...)` call.

In `server.py`, add the arg to both tools and pass through:

```python
    @mcp.tool()
    async def analyze_video(url: str, ctx: Context,
                            quality: QualityPreference = QualityPreference.BALANCED,
                            language: str = "en", force_whisper: bool = False,
                            analyze_visuals: bool = False,
                            async_: bool = False) -> dict:
        """Analyze a YouTube video into summary, chapters, quotes, and action items."""
        return await runtime.analyze(
            url=url, quality=quality, language=language,
            force_whisper=force_whisper, analyze_visuals=analyze_visuals,
            async_=async_, on_event=_progress(ctx))
```

```python
    @mcp.tool()
    async def extract_chapters(url: str, ctx: Context,
                               quality: QualityPreference = QualityPreference.BALANCED,
                               language: str = "en",
                               analyze_visuals: bool = False) -> list | dict:
        """Timestamped chapters for a YouTube video."""
        return await runtime.extract_chapters(
            url=url, quality=quality, language=language,
            analyze_visuals=analyze_visuals, on_event=_progress(ctx))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mcp_runtime.py tests/test_mcp_server.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mcp_server/runtime.py src/mcp_server/server.py tests/test_mcp_runtime.py tests/test_mcp_server.py
git commit -m "feat: expose analyze_visuals on MCP tools"
```

---

### Task 10: FastAPI adapter — analyze_visuals passthrough

**Files:**
- Modify: `src/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `JobOptions.analyze_visuals` (Task 1).
- Produces: the JSON `POST /api/jobs` path already accepts `options: JobOptions`, so `analyze_visuals` flows for free; the multipart upload path gains `analyze_visuals: bool = Form(False)` threaded into `JobOptions`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py` (follow its existing TestClient + fake-pipeline pattern; assert the flag reaches the created job's options):

```python
def test_create_job_accepts_analyze_visuals(client, captured_options):
    resp = client.post("/api/jobs", json={
        "url": "https://youtu.be/x",
        "options": {"language": "en", "quality": "balanced",
                    "force_whisper": False, "analyze_visuals": True},
    })
    assert resp.status_code == 200
    assert captured_options[-1].analyze_visuals is True
```

Use whatever capture seam `test_api.py` already provides (its fake pipeline / job store). If none captures `JobOptions`, assert instead that the response is 200 and the stored job’s options reflect the flag via the existing job-inspection helper in that file.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -k analyze_visuals -v`
Expected: FAIL (before wiring, upload path) or PASS-by-accident on JSON path — if the JSON test already passes because `JobOptions` carries the field, ALSO add the upload-form assertion below so the task has a failing target:

```python
def test_upload_job_accepts_analyze_visuals(client, captured_options):
    resp = client.post("/api/jobs/upload",
                       files={"file": ("a.wav", b"data", "audio/wav")},
                       data={"language": "en", "quality": "balanced",
                             "force_whisper": "false", "analyze_visuals": "true"})
    assert resp.status_code == 200
    assert captured_options[-1].analyze_visuals is True
```

Expected: FAIL — upload endpoint ignores `analyze_visuals`.

- [ ] **Step 3: Write minimal implementation**

In `src/api/main.py` upload endpoint, add the form field and thread it (around lines 84–92):

```python
                         quality: str = Form("balanced"),
                         force_whisper: bool = Form(False),
                         analyze_visuals: bool = Form(False)) -> dict:
```

```python
        options = JobOptions(language=language, quality=QualityPreference(quality),
                             force_whisper=force_whisper, analyze_visuals=analyze_visuals)
```

The JSON `CreateJobRequest.options: JobOptions` path needs no change.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api/main.py tests/test_api.py
git commit -m "feat: accept analyze_visuals on FastAPI upload endpoint"
```

---

### Task 11: Frontend — toggle + Visuals section

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/components/SubmitForm.tsx`
- Modify: `frontend/src/components/ReportView.tsx`
- Test: `frontend/src/components/ReportView.test.tsx`

**Interfaces:**
- Consumes: `Report`, `JobOptions` types.
- Produces:
  - `api.ts`: `VisualArtifact` interface, `Report.visual_highlights`, `JobOptions.analyze_visuals`, and `uploadJob` appends `analyze_visuals`.
  - `SubmitForm`: an `analyze_visuals` checkbox threaded into `options`.
  - `ReportView`: a "Visuals" section rendering timestamp, a kind badge, and `description ?? text`.

- [ ] **Step 1: Write the failing test**

In `ReportView.test.tsx`, extend the sample report and add an assertion. Add to the mock `report` object:

```tsx
  visual_highlights: [
    { timestamp_s: 30, kind: 'slide', text: 'Roadmap 2026', description: null, frame_path: null },
    { timestamp_s: 90, kind: 'chart', text: '', description: 'Revenue bar chart', frame_path: null },
  ],
```

Add a test case:

```tsx
  it('renders the visuals section with kinds and text', () => {
    render(<ReportView report={report} />)
    expect(screen.getByText('Visuals')).toBeInTheDocument()
    expect(screen.getByText('Roadmap 2026')).toBeInTheDocument()
    expect(screen.getByText('Revenue bar chart')).toBeInTheDocument()
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/ReportView.test.tsx`
Expected: FAIL — `visual_highlights` not on `Report` type / "Visuals" not rendered.

- [ ] **Step 3: Write minimal implementation**

In `api.ts`, add the interface and extend `Report` + `JobOptions`:

```ts
export interface VisualArtifact {
  timestamp_s: number; kind: 'slide' | 'code' | 'chart' | 'other'
  text: string; description: string | null; frame_path: string | null
}
export interface Report {
  summary: string; chapters: Chapter[]; key_quotes: KeyQuote[]
  action_items: string[]; language: string; trace_id: string; degraded_stages: string[]
  visual_highlights: VisualArtifact[]
}
export interface JobOptions {
  language: string; quality: 'cheap' | 'balanced' | 'best'
  force_whisper: boolean; analyze_visuals: boolean
}
```

In `uploadJob`, append the flag:

```ts
  form.append('analyze_visuals', String(options.analyze_visuals))
```

In `SubmitForm.tsx`, add state and a checkbox, and include it in `options`:

```tsx
  const [analyzeVisuals, setAnalyzeVisuals] = useState(false)
```

```tsx
    const options: JobOptions = { language, quality, force_whisper: false, analyze_visuals: analyzeVisuals }
```

Add the control inside the options row:

```tsx
        <label className="flex items-center gap-2 text-sm text-slate-600">
          <input type="checkbox" checked={analyzeVisuals}
                 onChange={(e) => setAnalyzeVisuals(e.target.checked)} />
          Analyze visuals
        </label>
```

In `ReportView.tsx`, add a section after "Chapters" (uses the existing `formatTs`):

```tsx
      {report.visual_highlights.length > 0 && (
        <section>
          <h2 className="mb-2 text-lg font-semibold">Visuals</h2>
          <ul className="space-y-2">
            {report.visual_highlights.map((v, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-16 shrink-0 font-mono text-sm text-slate-500">{formatTs(v.timestamp_s)}</span>
                <span className="shrink-0 rounded bg-slate-100 px-2 py-0.5 text-xs uppercase text-slate-600">{v.kind}</span>
                <span>{v.description ?? v.text}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run && npx tsc --noEmit`
Expected: PASS (render test green, typecheck clean).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.ts frontend/src/components/SubmitForm.tsx frontend/src/components/ReportView.tsx frontend/src/components/ReportView.test.tsx
git commit -m "feat: add analyze-visuals toggle and Visuals report section"
```

---

### Task 12: Slow smoke test — real ffmpeg + RapidOCR

**Files:**
- Create: `tests/test_visual_smoke.py`
- Test: itself (marked `@pytest.mark.slow`)

**Interfaces:**
- Consumes: `sample_scene_frames`, `rapidocr_text` (Task 3); `Visualizer` (Task 4).
- Produces: a slow end-to-end check that generates a tiny slide video with ffmpeg, runs the real sampler + OCR through the `Visualizer`, and asserts at least one artifact with recognizable text. Skips cleanly if ffmpeg is unavailable.

- [ ] **Step 1: Write the test**

Create `tests/test_visual_smoke.py`:

```python
import shutil
import subprocess

import pytest

from src.video_intelligence.agents.visualizer import Visualizer
from src.video_intelligence.schemas import (
    JobOptions, PipelineContext, SourceKind, VideoSource,
)

pytestmark = pytest.mark.slow


def _make_slide_video(path: str) -> None:
    # 3s video: white background with the word "ROADMAP" drawn large
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=white:s=640x480:d=3",
        "-vf", "drawtext=text='ROADMAP':fontcolor=black:fontsize=96:x=(w-tw)/2:y=(h-th)/2",
        "-r", "5", path,
    ], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_visualizer_reads_slide_text(tmp_path):
    video = str(tmp_path / "slide.mp4")
    _make_slide_video(video)
    vis = Visualizer(router=None, scene_threshold=0.0, max_frames=3,
                     min_interval_s=0.0, workdir=str(tmp_path / "frames"))
    ctx = PipelineContext(
        source=VideoSource(kind=SourceKind.LOCAL_FILE, path=video),
        options=JobOptions(analyze_visuals=True), video_path=video)
    ctx = await vis.run(ctx)
    assert ctx.visual_artifacts
    joined = " ".join(a.text.upper() for a in ctx.visual_artifacts)
    assert "ROADMAP" in joined
```

- [ ] **Step 2: Run the slow test (opt-in)**

Run: `pytest tests/test_visual_smoke.py -m slow -v`
Expected: PASS when ffmpeg + rapidocr-onnxruntime are installed; otherwise the test is skipped (ffmpeg) — install `rapidocr-onnxruntime` first if the OCR import fails.

- [ ] **Step 3: Verify the default suite still excludes it**

Run: `pytest -q`
Expected: PASS; the slow smoke test is deselected by the default marker config (matches `test_whisper_smoke.py` / `test_mcp_stdio_smoke.py`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_visual_smoke.py
git commit -m "test: add slow ffmpeg+OCR smoke test for the Visual agent"
```

---

### Task 13: Docs — README + spec status

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-15-visual-agent-design.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Update the README**

Add the Visual agent to the pipeline description (the README already documents Ingestor → Transcriber → Chapterizer → Synthesizer): insert `→ Visualizer` before Synthesizer, and add a short "Visual analysis (opt-in)" note describing `analyze_visuals`, low-res video fetch, OCR + optional vision escalation on `best`, and `rapidocr-onnxruntime` + ffmpeg requirements.

- [ ] **Step 2: Flip the spec status**

In `docs/superpowers/specs/2026-08-15-visual-agent-design.md`, change `**Status:** Approved (design), pending spec review` to `**Status:** Implemented`.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-08-15-visual-agent-design.md
git commit -m "docs: document the Visual agent and mark spec implemented"
```

---

## Final verification

- [ ] Run the full default suite: `pytest -q` — expect all green (78 prior + new).
- [ ] Frontend: `cd frontend && npx vitest run && npx tsc --noEmit` — expect green.
- [ ] Optional slow gate: `pytest -m slow -q` (needs ffmpeg + rapidocr).
- [ ] Confirm `analyze_visuals=False` path unchanged: `pytest tests/test_pipeline.py tests/test_ingestor.py -q`.
