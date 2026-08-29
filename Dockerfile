FROM python:3.12-slim

# psycopg[binary] ships its own libpq, so no build toolchain and no libpq-dev
# are needed here. Nothing in this image compiles from source.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies first so the layer caches across source edits.
COPY requirements.lock ./
RUN pip install --no-deps -r requirements.lock

# The docs are copied because tests assert the README's quoted numbers are
# real. A number in a README that nothing checks is the same class as an
# assertion that cannot fail.
COPY pyproject.toml alembic.ini README.md ARCHITECTURE.md ./
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY evals/ ./evals/
COPY tests/ ./tests/
COPY prompts/ ./prompts/

# Phase 1 has no long-running server: the only channel is the CLI. The
# container stays up so `docker compose exec app ...` works, which is what the
# Phase 1 checkpoint needs.
CMD ["sleep", "infinity"]
