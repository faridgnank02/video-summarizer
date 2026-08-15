# Video Intelligence

Agentic video analysis: a multi-agent pipeline that turns YouTube videos and
local recordings into structured reports — summary, timestamped chapters, key
quotes, and action items — with cost-aware model routing and per-stage tracing.

## Architecture

```
Ingestor ──▶ Transcriber ──▶ Chapterizer ──▶ Visualizer ──▶ Synthesizer
(yt-dlp,     (faster-        (small local    (OCR + optional  (frontier model:
 captions)    whisper)        model/Ollama)   vision LLM)      Claude / GPT)
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
- **Visual analysis (opt-in)** — set `analyze_visuals` (default off) to fetch
  a low-res copy of the video and sample frames on scene changes. Each frame
  is OCR'd with RapidOCR to pull on-screen content (slides, code, charts),
  with an optional vision-LLM escalation for the `best` quality tier. Results
  land in the report's `visual_highlights` and feed into the summary.
  Requires `ffmpeg` and `rapidocr-onnxruntime`.

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

## Run with Docker

```bash
cp .env.example .env   # add your OPENAI_API_KEY / ANTHROPIC_API_KEY
docker compose up -d --build
```

Then open http://localhost/. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for
the optional local-model profile and the AWS follow-up.

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
