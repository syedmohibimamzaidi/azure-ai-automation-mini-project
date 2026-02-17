#!/bin/bash
set -e
set -x

echo "Starting FastAPI app"
echo "PORT=$PORT"
which python
python --version

# DO NOT install packages here
exec python -m uvicorn main:app \
  --host 0.0.0.0 \
  --port ${PORT} \
  --log-level info
