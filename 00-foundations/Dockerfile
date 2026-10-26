# API image. Build and run with the database: docker compose --profile api up -d --build
FROM python:3.12-slim

# uv binary copied from its official image, pinned to the version used locally.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer: rebuilding after a code change reuses
# this cached layer instead of reinstalling every package.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Then the application itself.
COPY src ./src
COPY data ./data
COPY sql ./sql
RUN uv sync --locked --no-dev

# Don't run as root: if the app is ever compromised, the attacker isn't root in the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# 0.0.0.0 inside the container so Docker can forward traffic to it.
CMD ["/app/.venv/bin/uvicorn", "foundations.api:app", "--host", "0.0.0.0", "--port", "8000"]
