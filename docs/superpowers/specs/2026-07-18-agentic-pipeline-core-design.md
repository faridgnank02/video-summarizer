# Agentic Pipeline Core — Design

**Date:** 2026-07-18
**Status:** Approved
**Phase:** 1 of the "Agentic Multimodal Video Intelligence" transformation

## Goal

Transform the video summarizer from a single-model Streamlit app into a
multi-agent pipeline that produces structured video reports (summary, chapters,
key quotes, action items) with cost-aware model routing and per-stage tracing.
The project must work as both a daily-driver tool and a portfolio/blog
showcase.

## Scope

**In:** agent pipeline (Ingestor → Transcriber → Chapterizer → Synthesizer),
model router (Ollama local + OpenAI + Anthropic cloud), structured tracing,
FastAPI backend, React SPA frontend, YouTube + local-file input.

**Out (later phases):** MCP server (phase 2), Visual Agent (phase 3),
Fact-Checker (phase 4), live streams (phase 5), Docker/compose, Kubernetes,
re-wiring the quality-evaluation module.

## Architecture

One Python core library, `video_intelligence`, containing everything
intelligent (agents, router, tracing, schemas) with zero web dependencies.
FastAPI is a thin adapter over it; a React SPA talks to FastAPI. Streamlit and
the LED model are retired. Future adapters (MCP server, CLI) sit beside
FastAPI over the same core.

```
src/
  video_intelligence/          # core library — importable, testable, no web deps
    schemas.py                 # Pydantic models (below)
    pipeline.py                # async orchestrator: runs agents in sequence, emits events
    agents/
      base.py                  # Agent protocol: async run(context) -> context
      ingestor.py              # URL/file → audio + metadata (yt-dlp, ffmpeg);
                               # grabs YouTube captions when available
      transcriber.py           # faster-whisper (local) → timestamped segments;
                               # skipped when captions were fetched
      chapterizer.py           # small local model (Ollama) → timestamped chapters
      synthesizer.py           # frontier model → summary, key quotes, action items
    models/
      router.py                # task → model selection, config-driven, logged to trace
      providers/               # ollama.py, openai.py, anthropic.py — one thin client each
    tracing.py                 # per-stage spans → SQLite
  api/                         # FastAPI adapter: submit job, status, SSE progress, report
frontend/                      # React + Vite + TypeScript + Tailwind SPA
```

Key decisions:

- **Hand-rolled async orchestration** (no LangGraph/agent framework). The flow
  is a mostly linear DAG; plain asyncio with typed agent classes is simpler,
  fully debuggable, and framework-free.
- **The router's first decision is "don't transcribe at all."** For YouTube
  videos, the Ingestor fetches captions via `youtube-transcript-api` when any
  exist in the requested language (auto-generated included) and Whisper is
  skipped — the cheapest routing win in the pipeline. A per-job
  `force_whisper` option overrides this when caption quality is suspect.
- **Jobs run via FastAPI `BackgroundTasks`** — no Redis/Celery. The pipeline
  emits progress events; the API relays them over SSE. Live-stream mode
  (phase 5) will ride this same event channel.
- **LED is retired.** It's an extractive 2020-era model that no longer earns
  its complexity next to LLM output.

## Data Flow

1. `POST /jobs` with a source (YouTube URL or uploaded file) and options
   (target language; quality preference: `cheap | balanced | best`).
2. API creates a job row (SQLite), starts the pipeline as a background task,
   returns `job_id`.
3. Agents run in order, reading/writing a shared `PipelineContext` and
   emitting `StageEvent`s (`started`, `progress`, `completed`, `failed`)
   streamed over `GET /jobs/{id}/events` (SSE).
4. Final `AnalysisReport` is persisted. `GET /jobs/{id}` returns status +
   report; `GET /jobs/{id}/trace` returns the cost/latency breakdown.

## Schemas (Pydantic, `schemas.py`)

- `VideoSource` — `kind: youtube | local_file`, url/path, resolved metadata
  (title, duration, channel).
- `Transcript` — `segments: list[{start_s, end_s, text}]`, `language`,
  `origin: captions | whisper`. Caption and Whisper paths normalize into this
  one shape; downstream agents never know which produced it.
- `Chapter` — `start_s`, `end_s`, `title`, one-line `synopsis`.
- `AnalysisReport` — `summary` (markdown), `chapters`,
  `key_quotes: list[{timestamp_s, speaker?, text}]`, `action_items: list[str]`,
  `language`, `trace_id`.
- `TraceSpan` — `stage`, `model_used`, `tokens_in/out`, `cost_usd`,
  `latency_ms`, `status`, `fallback_from?`.

## Model Routing

`config/models.yaml` maps each task to ordered candidate lists per quality
preference:

```yaml
tasks:
  transcription:
    candidates: [captions, faster-whisper-base]
  chaptering:
    cheap:    [ollama/llama3.1:8b, openai/gpt-4o-mini]
    balanced: [ollama/llama3.1:8b, anthropic/claude-haiku]
    best:     [anthropic/claude-sonnet]
  synthesis:
    cheap:    [openai/gpt-4o-mini]
    balanced: [anthropic/claude-sonnet, openai/gpt-4o]
    best:     [anthropic/claude-opus, openai/gpt-5]
```

(Model names above are illustrative defaults — exact IDs are pinned in config
at implementation time, never in code.)

The router checks availability at call time (Ollama reachable? API key set?),
picks the first viable candidate, and logs the decision + reason to the trace.
Providers implement one interface —
`complete(prompt, schema) -> (parsed, usage)` — so adding a provider is one
file.

## Error Handling

- Each model-calling agent gets one retry, then falls back to the router's
  next candidate; the trace records `fallback_from`.
- Non-essential stage fails outright (chapterizer): pipeline continues, report
  ships without chapters, flagged as degraded.
- Essential stage fails (ingest, transcribe, synthesize): job fails with stage
  name and a human-readable reason.
- Long videos: transcript is chunked for chapterizer/synthesizer with a
  map-reduce pass when it exceeds the model's practical context window;
  chunk boundaries respect segment edges.

## Observability

Per-stage `TraceSpan`s stored in SQLite (replaces `metrics.db`), exposed via
`GET /jobs/{id}/trace` and rendered in the UI (cost, latency, tokens, model
chosen, fallbacks). This is the data source for the "$2 vs $20 query" story.

## Testing

- Unit tests per agent with a `FakeProvider` (canned responses, scripted
  failures): routing fallbacks, degraded reports, chunking.
- One integration test: full pipeline with fakes, asserting report shape and
  a trace span per stage.
- One `@pytest.mark.slow` smoke test: ~30s real audio fixture through real
  faster-whisper.
- Frontend: typecheck + one render test of the report view.

## Migration / Cleanup (this phase)

Delete `src/ui/` (Streamlit), `src/models/led_model.py` + LED config, and the
old `src/models/openai_model.py` / `model_manager.py` (replaced by providers +
router). `src/evaluation/` stays in-repo but unwired. `metrics.db` is replaced
by the trace store.

## Roadmap (later phases, each with its own spec)

- **Phase 2 — MCP server:** sibling adapter over the core library
  (`analyze_video`, `extract_chapters`).
- **Phase 3 — Visual Agent:** slide/code/chart detection + OCR between
  Transcriber and Synthesizer.
- **Phase 4 — Fact-Checker:** claims vs. web search, agentic loop.
- **Phase 5 — Live streams:** chunked rolling summaries on the existing event
  channel.
- Docker/compose and the quality-evaluation port slot in on demand.
