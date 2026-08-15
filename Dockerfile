# App image: runs either the FastAPI API or the MCP server (same image, two commands).
# No hardcoded --platform: builds native for the host (fast on Apple Silicon).
# For an amd64 target (AWS ECS/Fargate) build with: docker buildx build --platform linux/amd64
FROM python:3.12-slim

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
