#!/bin/bash
set -e

echo "Running database migrations..."
alembic upgrade head

# Hand over to the CMD (uvicorn). Without this the container exits after the migrations.
exec "$@"
