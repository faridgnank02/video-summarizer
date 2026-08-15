# Visual Agent — Design

**Date:** 2026-08-15
**Status:** Implemented
**Phase:** 3 of the "Agentic Multimodal Video Intelligence" transformation
**Branch:** `claude/video-summarizer-phase-3-ee2fc7` (off `claude/mcp-server-phase-2`)

## Goal

Add a **Visual agent** to the `video_intelligence` pipeline that samples video
frames, reads on-screen content (slides, code, charts) via OCR — with an
optional vision-LLM escalation — and folds those visual highlights into the
`AnalysisReport` and the Synthesizer's summary. This is the first phase to add a
new **core agent** and the first to take **multimodal input**, extending the
"$2 vs $20" cost story to visual analysis without breaking it.

## Scope

**In:**

- New non-essential `Visualizer` agent between the Chapterizer and Synthesizer.
- Opt-in per job via a new `analyze_visuals` flag (default off); when off, the
  pipeline is byte-for-byte unchanged and phase-1's caption cost win is fully
  preserved.
- Ingestor change: when `analyze_visuals` is on, also fetch a low-resolution
  video into `ctx.video_path` (even on the captions path); local files reuse the
  source path.
- Frame sampling via ffmpeg scene-change detection, capped and deduplicated.
- Local OCR via **RapidOCR** (pure-Python / onnxruntime, no system Tesseract).
- Heuristic classification into `slide | code | chart | other`.
- Hybrid vision escalation (path C): chart-candidate / text-sparse frames go to
  a vision model **only** when `quality=best` and a vision provider is
  available. Delivered after the OCR path in an A→C sequence.
- `VisualArtifact` schema, `AnalysisReport.visual_highlights`, and injection of a
  compact visual block into the Synthesizer prompt.
- Thin adapter passthrough: `analyze_visuals` argument on the MCP
  `analyze_video` / `extract_chapters` tools and the FastAPI job submit; a
  "Visuals" section + submit toggle in the SPA.
- Tests in phase-1 style (fakes; no network / no OCR download in the default
  suite) plus one slow real-ffmpeg + real-RapidOCR smoke test.

**Out (deferred, unchanged from earlier roadmap):**

- Fact-Checker agent + `fact_check_claims` (Phase 4).
- Live-stream mode (Phase 5).
- Deployment re-fit (Phase 2e).
- Evaluation re-wire.
- Speaker diarization, face/object detection, per-frame thumbnails served over
  HTTP (frame files stay local artifacts; only `frame_path` is recorded).
- Local-file upload over MCP (URLs only, unchanged from phase 2).

## Architecture

One new agent in the existing core library; everything else is a small,
additive change over the phase-1/phase-2 seams. No framework, still hand-rolled
async orchestration.

```
src/video_intelligence/
  schemas.py                 # + VisualArtifact, VisualKind; + report/context/options fields
  pipeline.py                # + Visualizer wired after Chapterizer (gated by analyze_visuals)
  agents/
    ingestor.py              # + optional low-res video fetch when analyze_visuals is on
    visualizer.py            # NEW: sample -> OCR -> classify -> (optional) vision escalate
    synthesizer.py           # + inject visual highlights into the prompt when present
  models/
    router.py                # + visual_description task selection
    providers/
      base.py                # + optional complete_vision(prompt, images, schema)
      anthropic.py           # implements complete_vision
      openai.py              # implements complete_vision
      ollama.py              # complete_vision optional (llava); may raise NotSupported
config/models.yaml           # + visual: sampling params, ocr engine, vision candidates
```

Key decisions:

- **New core agent, gated and non-essential.** `Visualizer.essential = False`,
  so a failure degrades the report (like the Chapterizer) rather than failing
  the job. It runs only when `ctx.options.analyze_visuals` is true; otherwise the
  pipeline skips it entirely.
- **The one deliberate exception to "don't download."** Visual analysis needs
  pixels, which the caption path never fetches. Gating video download behind the
  opt-in flag confines that cost to jobs that asked for it.
- **OCR before vision (A→C).** The essential first cut is scene detection + OCR +
  classification — cheap, deterministic, offline. Vision escalation is a bounded
  follow-on task behind the `quality=best` gate; it is the only reason the
  provider interface grows.
- **Types stay the contract.** New output is a Pydantic `VisualArtifact` that
  flows through `AnalysisReport` to every adapter (MCP, FastAPI, SPA) for free.

## Data Flow

1. A job arrives with `analyze_visuals=True` (MCP arg, FastAPI field, or SPA
   toggle).
2. **Ingestor**: resolves transcript/audio as today; additionally downloads a
   low-res video to `ctx.video_path` (YouTube ~360p via yt-dlp) or points
   `video_path` at the local source.
3. **Transcriber**, **Chapterizer**: unchanged.
4. **Visualizer** (only if `analyze_visuals` and `video_path` present):
   a. ffmpeg scene-change detection → candidate frame timestamps, capped by
      `max_frames` and `min_interval_s`; near-identical consecutive frames
      deduplicated.
   b. RapidOCR reads each kept frame → text.
   c. Heuristic classification → `VisualKind`.
   d. For `quality=best`: chart-candidate / text-sparse frames are sent to the
      router's `visual_description` model → `description`.
   e. Produces `ctx.visual_artifacts: list[VisualArtifact]`.
5. **Synthesizer**: if `ctx.visual_artifacts` is non-empty, injects a compact
   `[M:SS] <kind>: <text|description>` block into the prompt; sets
   `report.visual_highlights`.
6. `AnalysisReport.visual_highlights` is persisted and surfaces in MCP results,
   the FastAPI report endpoint, and the SPA.

## Schemas (`schemas.py`)

```python
class VisualKind(StrEnum):
    SLIDE = "slide"
    CODE = "code"
    CHART = "chart"
    OTHER = "other"

class VisualArtifact(BaseModel):
    timestamp_s: float
    kind: VisualKind
    text: str = ""                 # OCR text (may be empty for charts)
    description: str | None = None # vision-LLM description when escalated
    frame_path: str | None = None  # local artifact path, not served

# AnalysisReport: + visual_highlights: list[VisualArtifact] = Field(default_factory=list)
# JobOptions:     + analyze_visuals: bool = False
# PipelineContext:+ video_path: str | None = None
#                 + visual_artifacts: list[VisualArtifact] | None = None
```

## The Visualizer Agent

Constructed with injectable seams so the default suite needs no ffmpeg/OCR:

- `frame_sampler(video_path, params) -> list[(timestamp_s, image_path)]`
  (default: ffmpeg scene detection).
- `ocr(image_path) -> str` (default: RapidOCR).
- `router` (for the optional vision escalation).

Behavior:

- **Cap & dedup:** honor `max_frames` and `min_interval_s`; drop a frame whose
  OCR text is near-identical (normalized) to the previous kept frame.
- **Classify:** `code` when indentation + symbol density are high; `slide` when
  short title-like lines + bullets; `chart` when text is sparse but the frame is
  visually dense; else `other`.
- **Escalate (quality=best only):** send `chart`-candidate / text-sparse frames
  to `router` task `visual_description`; store the result in `description`. When
  no vision provider is available, skip silently — the OCR result stands.
- **Failure:** any exception → agent raises; pipeline records it in
  `degraded_stages` and ships the report without visuals (non-essential policy).

## Provider / Router

- `ProviderBase` gains an **optional** `complete_vision(prompt, images, schema)
  -> (parsed, usage)`; the text `complete` contract is unchanged. A default
  implementation raises `NotSupported` so providers opt in.
- `config/models.yaml` gains:

```yaml
visual:
  sampling:
    scene_threshold: 0.4
    max_frames: 24
    min_interval_s: 8
  ocr: rapidocr
  description:            # vision escalation candidates, best-quality only
    best: [anthropic/claude-sonnet-vision, openai/gpt-4o]
```

(Model IDs illustrative; pinned in config at implementation time.)

- The router adds a `visual_description` task that routes to `complete_vision`,
  checks provider availability at call time (as with text tasks), and logs the
  decision + cost to the trace under stage `visual`.

## Adapters

- **MCP** (`src/mcp_server/`): `analyze_video` and `extract_chapters` gain
  `analyze_visuals: bool = False`. `visual_highlights` rides along in the
  returned `AnalysisReport`. No new tool this phase.
- **FastAPI** (`src/api/`): job-submit accepts `analyze_visuals`; the report
  endpoint returns `visual_highlights` unchanged in shape.
- **SPA** (`frontend/`): a submit toggle and a "Visuals" section in the report
  view (timestamp, kind badge, OCR text / description).

## Error Handling

- **Ingestor video fetch fails** while `analyze_visuals` is on: log and continue
  without `video_path`; the Visualizer then no-ops and the stage is degraded —
  the audio/transcript job still succeeds.
- **Scene detection / OCR fails:** Visualizer raises; non-essential policy marks
  `visual` degraded; report ships without visuals.
- **Vision escalation fails or unavailable:** skip escalation, keep OCR text; no
  degradation (the artifact is still useful).
- **No frames found:** empty `visual_highlights`, not an error.

## Testing

Phase-1 style: fakes injected through the agent's seams; no network and no OCR
model download in the default suite.

- **Visualizer units** (fake sampler + fake OCR):
  - frame cap and `min_interval_s` respected;
  - dedup drops near-identical consecutive frames;
  - classification heuristics (code / slide / chart / other);
  - `quality=best` triggers vision escalation via `FakeProvider`; lower
    qualities do not;
  - vision-unavailable path keeps OCR text and does not degrade;
  - sampler/OCR exception → stage degraded, no visuals in report.
- **Ingestor unit:** `analyze_visuals=True` sets `video_path` (fake downloader);
  `False` leaves the caption path untouched.
- **Synthesizer unit:** visual block injected into the prompt only when
  artifacts present; `report.visual_highlights` populated.
- **Pipeline integration (fakes):** end-to-end with `analyze_visuals=True`
  yields a report with `visual_highlights` and a `visual` trace span.
- **Adapter tests:** MCP tool schema exposes `analyze_visuals`; FastAPI submit
  accepts it.
- **Slow smoke** (`@pytest.mark.slow`): real ffmpeg scene detection + real
  RapidOCR on a tiny generated slide-video fixture; asserts at least one
  `slide` artifact with non-empty text.
- **Frontend:** one render test of the Visuals section.

## Dependencies

Add `rapidocr-onnxruntime` to `requirements.txt` (ffmpeg already required; yt-dlp
already present). Vision escalation reuses the existing OpenAI/Anthropic SDKs.
No other new dependencies.

## Migration / Cleanup

None required. Downloaded video and sampled frames are runtime artifacts under
`data/work`, already gitignored. `frame_path` records local paths only; serving
thumbnails over HTTP is explicitly out of scope.

## Roadmap (remaining phases, unchanged)

- **Phase 4 — Fact-Checker:** claims vs. web search; unlocks `fact_check_claims`
  as a sixth MCP tool.
- **Phase 5 — Live streams:** chunked rolling summaries on the existing event
  channel.
- **Phase 2e — Deployment re-fit:** Docker/compose/nginx/AWS for FastAPI + SPA +
  MCP HTTP transport.
- **Evaluation re-wire:** point `src/evaluation/` at the new pipeline's reports.
