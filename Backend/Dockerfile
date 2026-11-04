FROM ghcr.io/astral-sh/uv:0.6.14-python3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_LINK_MODE=copy

WORKDIR /app


ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch --index-url ${TORCH_INDEX_URL}


COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

RUN if echo "${TORCH_INDEX_URL}" | grep -q '/cpu$'; then \
      if uv pip list --system | grep -qi '^nvidia'; then \
        echo "ERROR: CUDA packages were installed although CPU-only torch was requested" >&2; exit 1; \
      fi; \
    fi


COPY app/ ./
COPY alembic/ ./alembic/
COPY alembic.ini ./alembic.ini
COPY --chmod=755 entrypoint.sh /entrypoint.sh


RUN useradd -m appuser \
    && mkdir -p /app/storage/uploads /app/storage/embeddingcache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]