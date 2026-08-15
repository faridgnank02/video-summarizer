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

### MCP HTTP host validation

The MCP Streamable-HTTP transport ships with DNS-rebinding protection that
rejects any `Host` header not in its allowlist — which rejects every request
that arrives through a reverse proxy. Because the `mcp` service is reachable
only behind nginx here, the entrypoint disables that protection by default. To
re-enable it (recommended once a fixed public hostname exists), set on the
`mcp` service:

- `MCP_ALLOWED_HOSTS` — comma-separated Host values to allow (e.g.
  `mcp.example.com`).
- `MCP_ALLOWED_ORIGINS` — comma-separated allowed `Origin` values.

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
