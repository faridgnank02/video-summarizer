# Deployment Re-fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Containerize the current stack — FastAPI backend, built React SPA, and the MCP Streamable-HTTP transport — behind an nginx front door, runnable with `docker compose up`, with Ollama as an optional profile and the image kept AWS-ready.

**Architecture:** Two images. One Python `app` image (Python 3.12 + ffmpeg + `requirements.txt`) runs as either the `api` service (`uvicorn src.api.main:app`) or the `mcp` service (`python -m src.mcp_server --transport http`). One multi-stage `web` image (node build → nginx) serves the SPA and reverse-proxies `/api/*` → `api` and `/mcp` → `mcp`. Ollama is a separate service behind the `local-models` compose profile; when absent the model router auto-falls-back to cloud candidates.

**Tech Stack:** Docker, docker compose (v2, `profiles`), nginx (alpine), Node 20 (Vite build), Python 3.12-slim, uvicorn, the official MCP SDK's Streamable-HTTP transport.

**Spec:** `docs/superpowers/specs/2026-08-15-deployment-refit-design.md`

## Global Constraints

- Only one file under `src/` changes: `src/video_intelligence/models/providers/ollama.py` (env-driven base-URL default). No other core/adapter code edits.
- Python 3.12 base image; app runtime deps come only from `requirements.txt` (no per-package pins in the Dockerfile).
- No secrets in images or committed files — only via `.env` / environment. `.env` stays git-ignored; only `.env.example` is committed.
- The SPA already uses relative `/api/...` URLs (`frontend/src/api.ts`) — do not add a build-time API host.
- Images must build for `linux/amd64` (AWS-ready); do not bake an Ollama model into the `app` image.
- The existing Python suite (157 passing, run with plain `pytest` from repo root) must stay green.
- Commit after every task with a conventional-commit message.

---

### Task 1: Env-driven Ollama base URL

**Files:**
- Modify: `src/video_intelligence/models/providers/ollama.py`
- Test: `tests/test_ollama_provider.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `OllamaProvider(base_url: str | None = None, transport=None)` — when `base_url` is `None`, the effective URL is `os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")`. Explicit `base_url=` still wins (tests rely on this).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ollama_provider.py`:

```python
def test_base_url_defaults_to_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    provider = OllamaProvider()
    assert provider._base_url == "http://ollama:11434"


def test_base_url_falls_back_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    provider = OllamaProvider()
    assert provider._base_url == "http://localhost:11434"


def test_explicit_base_url_overrides_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    provider = OllamaProvider(base_url="http://custom:9999")
    assert provider._base_url == "http://custom:9999"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ollama_provider.py::test_base_url_defaults_to_env_var -v`
Expected: FAIL — default is the literal `"http://localhost:11434"`, env var ignored.

- [ ] **Step 3: Implement the env-driven default**

In `src/video_intelligence/models/providers/ollama.py`, add `import os` at the top (after `from __future__ import annotations`), and change the constructor:

```python
    def __init__(self, base_url: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434")
        self._transport = transport
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ollama_provider.py -v`
Expected: PASS (the three new tests plus the two existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/video_intelligence/models/providers/ollama.py tests/test_ollama_provider.py
git commit -m "feat: make Ollama base URL configurable via OLLAMA_BASE_URL"
```

---

### Task 2: App image (Dockerfile + .dockerignore)

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: `requirements.txt`, `src/`, `config/`.
- Produces: an image tagged `vi-app:latest` at build time (tag applied by compose in Task 4). WORKDIR `/app`; `PYTHONPATH=/app` so `src.api.main` / `src.mcp_server` import. `ffmpeg` on PATH. Default `CMD` runs the API on `0.0.0.0:8000`; the `mcp` service overrides the command in compose.

- [ ] **Step 1: Write `.dockerignore`**

Create `.dockerignore`:

```
.git
.github
.claude
.serena
.superpowers
video-summarizer-env
**/node_modules
frontend/dist
data
logs
**/__pycache__
*.pyc
.pytest_cache
docs
*.db
*.log
.env
```

- [ ] **Step 2: Write the `app` Dockerfile**

Create `Dockerfile`:

```dockerfile
# App image: runs either the FastAPI API or the MCP server (same image, two commands).
FROM --platform=linux/amd64 python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY config/ /app/config/

RUN mkdir -p /app/data/work /app/data/uploads

EXPOSE 8000

# Default role: the API. The mcp service overrides this command in compose.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build the image**

Run: `docker build -t vi-app:latest .`
Expected: build completes; final line shows the `vi-app:latest` tag. (First build is slow — faster-whisper/onnxruntime deps.)

- [ ] **Step 4: Smoke-check both roles start**

Run:
```bash
docker run --rm -d --name vi-api-test -p 8000:8000 vi-app:latest
sleep 5
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/jobs/nope   # expect 404
docker logs vi-api-test | tail -5
docker rm -f vi-api-test
```
Expected: `404` (FastAPI is up and routing; unknown job → 404). Logs show uvicorn "Application startup complete".

Then the MCP role:
```bash
docker run --rm -d --name vi-mcp-test vi-app:latest \
  python -m src.mcp_server --transport http --host 0.0.0.0 --port 8000
sleep 5
docker logs vi-mcp-test | tail -5
docker rm -f vi-mcp-test
```
Expected: logs show the MCP server started on port 8000 (uvicorn/Streamable-HTTP), no traceback.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add app Docker image for API and MCP roles"
```

---

### Task 3: Web image (frontend/Dockerfile + nginx.conf)

**Files:**
- Create: `frontend/Dockerfile`
- Create: `nginx.conf`

**Interfaces:**
- Consumes: `frontend/` sources, `nginx.conf`. Build context is the repo root (so the Dockerfile can copy both `frontend/` and `nginx.conf`).
- Produces: an image tagged `vi-web:latest` (tag applied by compose in Task 4). Serves the SPA on port 80, proxies `/api/` → `http://api:8000`, `/mcp` → `http://mcp:8000`. SPA client-side routing via `try_files ... /index.html`. SSE/streaming paths have buffering disabled.

- [ ] **Step 1: Write `nginx.conf`**

Create `nginx.conf` (repo root):

```nginx
events { worker_connections 1024; }

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    sendfile on;
    keepalive_timeout 65;
    client_max_body_size 500M;   # video/audio uploads

    server {
        listen 80;
        server_name _;
        root /usr/share/nginx/html;
        index index.html;

        # SPA static assets + client-side routing fallback
        location / {
            try_files $uri $uri/ /index.html;
        }

        # FastAPI (JSON + SSE). Buffering off so SSE events flush immediately.
        location /api/ {
            proxy_pass http://api:8000;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_buffering off;
            proxy_read_timeout 3600;
            proxy_send_timeout 3600;
        }

        # MCP Streamable-HTTP. Preserve /mcp; buffering off for streamed responses.
        location /mcp {
            proxy_pass http://mcp:8000;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_buffering off;
            proxy_read_timeout 3600;
            proxy_send_timeout 3600;
        }
    }
}
```

- [ ] **Step 2: Write the multi-stage `web` Dockerfile**

Create `frontend/Dockerfile`:

```dockerfile
# Web image: build the SPA with Node, serve it (and reverse-proxy) with nginx.
FROM --platform=linux/amd64 node:20-slim AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM --platform=linux/amd64 nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

- [ ] **Step 3: Build the web image**

Run: `docker build -t vi-web:latest -f frontend/Dockerfile .`
Expected: both stages complete; `vi-web:latest` tagged. (If `npm ci` fails on a missing lockfile, run `npm install` in `frontend/` once to generate `package-lock.json`, commit it, and rebuild.)

- [ ] **Step 4: Smoke-check the SPA is served**

Run:
```bash
docker run --rm -d --name vi-web-test -p 8080:80 vi-web:latest
sleep 2
curl -s http://localhost:8080/ | grep -o "<title>[^<]*</title>"
docker rm -f vi-web-test
```
Expected: the SPA's `<title>` line prints (index.html served). Proxy locations are exercised end-to-end in Task 4.

- [ ] **Step 5: Commit**

```bash
git add frontend/Dockerfile nginx.conf frontend/package-lock.json
git commit -m "feat: add web image (Vite build + nginx front door)"
```

---

### Task 4: Compose stack + .env.example

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`

**Interfaces:**
- Consumes: the `app` Dockerfile (Task 2), `frontend/Dockerfile` (Task 3), `.env`.
- Produces: `web` (public :80), `api` (internal), `mcp` (internal), and `ollama` (profile `local-models`). `api`/`mcp` read `OLLAMA_BASE_URL=http://ollama:11434` from environment. Named volumes: `app_data` (`/app/data`), `whisper_cache`, `ollama_models`.

- [ ] **Step 1: Write `.env.example`**

Create `.env.example`:

```
# Cloud synthesis providers (either or both; router falls back across them)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Optional: web-search backend for the fact-checker (else DuckDuckGo, keyless)
TAVILY_API_KEY=

# Optional: override the Ollama endpoint. In compose this is set to
# http://ollama:11434 for the api/mcp services automatically.
# OLLAMA_BASE_URL=http://localhost:11434
```

- [ ] **Step 2: Write `docker-compose.yml`**

Create `docker-compose.yml`:

```yaml
services:
  web:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    image: vi-web:latest
    ports:
      - "80:80"
    depends_on:
      - api
      - mcp
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: vi-app:latest
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    env_file: .env
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - app_data:/app/data
      - whisper_cache:/root/.cache/huggingface
    restart: unless-stopped

  mcp:
    image: vi-app:latest
    command: python -m src.mcp_server --transport http --host 0.0.0.0 --port 8000
    env_file: .env
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - app_data:/app/data
      - whisper_cache:/root/.cache/huggingface
    depends_on:
      - api
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    profiles: ["local-models"]
    volumes:
      - ollama_models:/root/.ollama
    restart: unless-stopped

volumes:
  app_data:
  whisper_cache:
  ollama_models:
```

Note: `mcp` reuses the `vi-app:latest` image built by `api` (no second build). `depends_on: [api]` guarantees the image exists before `mcp` starts.

- [ ] **Step 3: Validate compose config**

Run: `docker compose config`
Expected: prints the resolved config with no errors; `ollama` is present (config shows all profiles) but is not in the default `up` set.

- [ ] **Step 4: Build and start the cloud-only stack**

Run:
```bash
cp .env.example .env    # fill in a real key if you want live analysis; smoke works without
docker compose up -d --build
sleep 8
docker compose ps
```
Expected: `web`, `api`, `mcp` are `running` (or healthy); `ollama` is NOT started.

- [ ] **Step 5: End-to-end proxy smoke through nginx**

Run:
```bash
curl -s -o /dev/null -w "spa:%{http_code}\n" http://localhost:80/
curl -s -o /dev/null -w "api:%{http_code}\n" http://localhost:80/api/jobs/nope
```
Expected: `spa:200` and `api:404` (nginx serves the SPA and proxies to the API).

- [ ] **Step 6: Tear down and commit**

```bash
docker compose down
git add docker-compose.yml .env.example
git commit -m "feat: add docker-compose stack with optional local-models profile"
```

---

### Task 5: Smoke script + deployment docs

**Files:**
- Create: `scripts/smoke.sh`
- Create: `docs/DEPLOYMENT.md`
- Modify: `README.md` (add a short "Run with Docker" pointer)

**Interfaces:**
- Consumes: the running compose stack from Task 4.
- Produces: `scripts/smoke.sh` (exits non-zero on any failed surface); `docs/DEPLOYMENT.md` (local run + AWS follow-up runbook).

- [ ] **Step 1: Write `scripts/smoke.sh`**

Create `scripts/smoke.sh`:

```bash
#!/usr/bin/env bash
# Smoke-test a running compose stack (default: http://localhost:80).
set -euo pipefail
BASE="${1:-http://localhost:80}"
fail() { echo "SMOKE FAIL: $1"; exit 1; }

echo "1/3 SPA served at / ..."
code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/")
[ "$code" = "200" ] || fail "SPA returned $code (want 200)"

echo "2/3 API reachable via /api ..."
code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/api/jobs/nope")
[ "$code" = "404" ] || fail "API returned $code (want 404 for unknown job)"

echo "3/3 MCP handshake via /mcp ..."
# initialize is a POST; a 200/2xx or a JSON-RPC body means the endpoint is live.
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}')
case "$code" in
  2*|400) echo "   mcp endpoint responded ($code)";;
  *) fail "MCP returned $code";;
esac

echo "ALL SMOKE CHECKS PASSED"
```

(The MCP check accepts `400` too: a bare `initialize` without a session header still proves the endpoint is mounted and reachable through nginx, which is what this smoke asserts.)

- [ ] **Step 2: Make it executable and run it against the live stack**

Run:
```bash
chmod +x scripts/smoke.sh
docker compose up -d --build
sleep 8
./scripts/smoke.sh
docker compose down
```
Expected: `ALL SMOKE CHECKS PASSED`.

- [ ] **Step 3: Write `docs/DEPLOYMENT.md`**

Create `docs/DEPLOYMENT.md`:

```markdown
# Deployment

## Local (Docker Compose)

Prerequisites: Docker + docker compose v2.

```bash
cp .env.example .env          # add OPENAI_API_KEY and/or ANTHROPIC_API_KEY for live analysis
docker compose up -d --build  # web :80, api + mcp internal
./scripts/smoke.sh            # verify the three surfaces
```

- App UI: http://localhost/ (React SPA)
- API: proxied at http://localhost/api/ (FastAPI + SSE)
- MCP: Streamable-HTTP at http://localhost/mcp

### Local models (optional)

Cloud synthesis is the default; the model router falls back to cloud when
Ollama is absent. To run the free local-model tier for chaptering:

```bash
docker compose --profile local-models up -d
docker compose exec ollama ollama pull llama3.1:8b   # first time only
```

`api` and `mcp` reach it at `http://ollama:11434` via `OLLAMA_BASE_URL`.

### Data & secrets

- SQLite stores + work/uploads persist in the `app_data` volume.
- Whisper models cache in `whisper_cache`; Ollama models in `ollama_models`.
- Secrets come only from `.env` (git-ignored). Never commit real keys.

## AWS ECS Fargate (follow-up — not yet automated)

The `vi-app` and `vi-web` images are built `linux/amd64` and 12-factor, so an
ECS Fargate deploy is a packaging exercise, adapted from the retired
`ollama-integration` scripts (`scripts/deploy_aws.sh`,
`scripts/setup_eventbridge_schedule.sh`, `scripts/get_app_url.sh`):

1. Build both images for `linux/amd64`; push to ECR.
2. Fargate task definition running `web` + `api` + `mcp` (Ollama omitted, or a
   separate task); an ALB fronts port 80 and takes nginx's routing role (or
   nginx rides along in the task).
3. Provide `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` via ECS secrets (SSM /
   Secrets Manager), never in the image.
4. Optional run-hours schedule via EventBridge (start/stop the service).

**Security requirement (hard):** the `/mcp` endpoint is unauthenticated. Before
any public AWS exposure it MUST sit behind an auth gate (reverse-proxy bearer
token or ALB authentication). Do not expose `/mcp` publicly without it.
```

- [ ] **Step 4: Add a README pointer**

In `README.md`, under the Requirements/Quick-start area, add:

```markdown
## Run with Docker

```bash
cp .env.example .env   # add your OPENAI_API_KEY / ANTHROPIC_API_KEY
docker compose up -d --build
```

Then open http://localhost/. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for
the optional local-model profile and the AWS follow-up.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke.sh docs/DEPLOYMENT.md README.md
git commit -m "docs: add smoke script and deployment guide"
```

---

## Notes for the implementer

- **Docker must be running** to execute Tasks 2–5. If the environment has no Docker daemon, implement and commit the files, run the one testable code change (Task 1) with `pytest`, and mark the build/smoke steps as blocked in the task notes rather than faking their output.
- **Do not touch** `src/` beyond Task 1. If a build seems to need a code change, it almost certainly means a Dockerfile/compose/nginx fix instead.
- The MCP Streamable-HTTP endpoint path is `/mcp` (the SDK's default mount); nginx `location /mcp` proxies to `mcp:8000` preserving that path.
- `whisper_cache` is mounted at `/root/.cache/huggingface` because faster-whisper downloads CTranslate2 models there; adjust only if a build shows a different cache path in the logs.
- Keep `.env` out of git — it is already covered by the root `.gitignore` (`.env` line); only `.env.example` is committed.
