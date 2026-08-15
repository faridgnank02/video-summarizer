# Deployment Re-fit — Design

**Date:** 2026-08-15
**Status:** Approved
**Phase:** 2e of the "Agentic Multimodal Video Intelligence" transformation
**Branch:** `claude/deployment-refit-phase-2e` (off `main`)

## Goal

Give the new stack a real, runnable deployment. Adapt the containerization and
cloud tooling from the retired `ollama-integration` branch (Dockerfile,
docker-compose, nginx, AWS ECS scripts) to the current architecture: a FastAPI
backend, a built React SPA, and the MCP server's Streamable HTTP transport —
retiring every assumption that no longer holds (Streamlit, a bundled Ollama
model, supervisord, per-package pip pins).

Primary deliverable this phase: a polished local `docker compose up`. The images
are built to be AWS-ready so an ECS Fargate deploy can follow on the same
artifacts as a documented next slice.

## Scope

**In:**
- A Python `app` image (Python 3.12 + ffmpeg + `requirements.txt`) that runs as
  either the FastAPI `api` service or the MCP `mcp` service.
- A multi-stage `web` image: Node builds the SPA, nginx serves the static build
  and reverse-proxies `/api/*` and `/mcp`.
- A `docker-compose.yml` wiring `web` (public :80), `api` (internal), `mcp`
  (internal), and an optional `ollama` service behind a `local-models` profile.
- Config/secrets via `.env` (refreshed `.env.example`); named volumes for the
  SQLite stores + work/uploads, the faster-whisper model cache, and Ollama
  models.
- One small core change: make the Ollama base URL env-driven
  (`OLLAMA_BASE_URL`) so identical code works on a laptop and in compose.
- A `scripts/smoke.sh` and a `docs/` "run it" runbook covering both the
  cloud-only and `local-models` paths.

**Out (documented follow-up, not executed/validated this phase):**
- The live AWS ECS Fargate deploy (ECR push, task definition, ALB, EventBridge
  schedule). A runbook adapts the old scripts; it is not run here.
- Any authentication on the public surfaces. The MCP HTTP endpoint is exposed
  unauthenticated for local use; the AWS runbook MUST require an auth gate
  before public exposure (see Security).
- Kubernetes/Helm, CI/CD pipelines, TLS/cert automation, autoscaling.
- Re-wiring `src/evaluation/` (its own later slice).

## Architecture

Two images; four compose services (three default + one profiled). nginx is the
only public entrypoint.

```
                       ┌─────────────── web (nginx) :80 ───────────────┐
   browser ───────────▶│  /            → SPA static (dist, SPA fallback) │
   MCP host  ──────────▶│  /api/*       → api:8000                       │
                       │  /mcp         → mcp:8000                        │
                       └───────┬───────────────────┬────────────────────┘
                               │                   │
                        ┌──────▼──────┐     ┌───────▼───────┐
                        │ api (app)   │     │ mcp (app)     │
                        │ uvicorn     │     │ -m mcp_server │
                        │ src.api.main│     │ --transport   │
                        │             │     │   http        │
                        └──────┬──────┘     └───────┬───────┘
                               │                    │
                               └─────────┬──────────┘
                                         │ OLLAMA_BASE_URL (when profile on)
                                  ┌──────▼───────┐
                                  │ ollama       │  profile: local-models
                                  │ (optional)   │  (absent → router falls
                                  └──────────────┘   back to cloud)
```

| Service | Image | Command | Exposed | Notes |
|---------|-------|---------|---------|-------|
| `web` | `web` (multi-stage) | nginx | `:80` public | serves SPA, proxies `/api`, `/mcp` |
| `api` | `app` | `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` | internal | FastAPI + SSE |
| `mcp` | `app` | `python -m src.mcp_server --transport http --host 0.0.0.0 --port 8000` | internal | Streamable HTTP |
| `ollama` | `ollama/ollama` | default | internal | **`--profile local-models`** only |

Key decisions:

- **One `app` image, two roles.** `api` and `mcp` are the same image with
  different commands, so their dependency set and Python version can never
  drift. Matches the "adapters over one core" architecture.
- **nginx front door.** Standard SPA deployment: static assets served directly,
  API/MCP reverse-proxied, one public port. Maps cleanly onto an AWS ALB later
  (the ALB assumes nginx's routing role, or nginx rides along in the task).
- **Ollama is optional and external.** The old image baked Ollama + a model
  (~2.5–3.5 GB). Here Ollama is a separate service enabled only via
  `--profile local-models`; the default `compose up` is cloud-only and the
  model router auto-falls-back to cloud candidates when Ollama is unreachable.
  Keeps the `app` image lean and Fargate cold-starts fast.
- **Retire the old multi-process container.** No Streamlit, no supervisord, no
  `startup.sh` orchestration, no per-package pip pins. Each service is one
  process; compose owns process lifecycle.

## The one core change

`src/video_intelligence/models/providers/ollama.py` currently hard-defaults
`base_url="http://localhost:11434"`. Change the default to read
`os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")` so the compose
`api`/`mcp` services can point at `http://ollama:11434` without a code edit. The
constructor argument stays (tests inject it); only the default source changes.
This is the sole edit under `src/`; everything else is Dockerfiles, compose,
nginx, scripts, and docs.

## SPA build & serving

- `frontend/` builds with `npm ci && npm run build` → `dist/`.
- The `web` image is multi-stage: a `node:20` stage produces `dist/`, copied
  into an `nginx:alpine` stage. No Node in the runtime image.
- The SPA already calls the API with **relative** URLs (confirmed in
  `frontend/src/api.ts`: `fetch('/api/jobs')`, `/api/jobs/${id}/events`, etc.),
  so nginx's proxy makes it origin-agnostic with no build-time API host baked in
  and **no frontend code change required**.
- nginx config: `try_files $uri $uri/ /index.html` for SPA routing; `location
  /api/` and `location /mcp` proxy blocks with SSE-friendly timeouts and
  buffering disabled on the SSE path.

## Config, secrets, persistence

- **`.env`** (git-ignored) holds `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  optional `TAVILY_API_KEY`, and optional `OLLAMA_BASE_URL`. A rewritten
  `.env.example` documents them. `api` and `mcp` receive them via compose
  `env_file`.
- **Volumes** (named): `app_data` → `/app/data` (SQLite `app.db`/`traces.db`,
  `work/`, `uploads/`); `whisper_cache` → the faster-whisper model download
  location; `ollama_models` → `/root/.ollama` (profile only).
- **Legacy cruft** from the old branch (`build.log`, `fix.txt`, root `metrics.db`,
  root `.env`, `config/model_config.yaml`, `supervisord.conf`) is not carried
  into the new tree; the new `.gitignore` already excludes `/data/*.db`.

## Security

- Local compose exposes only `web:80`. `api`, `mcp`, and `ollama` are on the
  internal compose network, not published to the host.
- `/mcp` is **unauthenticated** — acceptable on a local/trusted network. The
  AWS follow-up runbook MUST add an auth gate (e.g. a reverse-proxy bearer
  token or ALB auth) before any public exposure; this is called out as a
  hard requirement of that slice, not an optional nicety.
- No secrets in images or compose files — only via `.env`/environment.

## AWS follow-up (documented, deferred)

A `docs/` runbook (adapting `deploy_aws.sh`, `setup_eventbridge_schedule.sh`,
`get_app_url.sh` from the old branch) will describe: build both images for
`linux/amd64`, push to ECR, define a Fargate task running `web`+`api`+`mcp`
(Ollama omitted or a separate task), front it with an ALB, and optionally
schedule run-hours via EventBridge. Written this phase, executed later — live
AWS cannot be validated in this environment.

## Testing / verification

Deployment is infrastructure; verification is build-and-smoke, not unit tests:

1. `docker compose build` succeeds for `app` and `web`.
2. `docker compose up -d` (cloud-only): `web` serves the SPA at `/`; `GET
   /api/jobs/<bogus>` returns a clean 404 (proxy works); `POST /mcp`
   completes an MCP `initialize` + `list_tools` handshake returning the tool
   set.
3. `docker compose --profile local-models up -d`: `api` reaches `http://
   ollama:11434/api/tags`; a chaptering call routes to the local model (span
   `model_used` starts `ollama/`).
4. `scripts/smoke.sh` curls the three surfaces and exits non-zero on failure.
5. The existing Python suite (`pytest`, 157 passing) is unaffected — the only
   code change is the Ollama default, covered by an updated provider test
   asserting `OLLAMA_BASE_URL` is honored.

## Deliverables

- `Dockerfile` (app image), `frontend/Dockerfile` (web image), `.dockerignore`
- `docker-compose.yml` (+ `local-models` profile), `nginx.conf`
- `.env.example` (rewritten)
- `scripts/smoke.sh`
- `docs/DEPLOYMENT.md` (local run + AWS follow-up runbook)
- `src/video_intelligence/models/providers/ollama.py` (env-driven default) +
  test update

## Roadmap (remaining after this phase)

- **AWS ECS Fargate deploy** — execute the documented runbook.
- **Evaluation re-wire** — point `src/evaluation/` (BERTScore/quality metrics)
  at the new pipeline's reports.
- Deferred smaller items: true live ingestion (yt-dlp HLS into phase-5's
  `SegmentFeed` seam), frontend live view, live-mode chapters.
