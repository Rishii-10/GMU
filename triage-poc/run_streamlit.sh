#!/bin/sh
# Wrapper so the launch harness's assigned $PORT (autoPort) reaches Streamlit,
# which only accepts a port via --server.port, not a generic PORT env var.
DIR="$(cd "$(dirname "$0")" && pwd)"
# Run from the app directory so Streamlit finds .streamlit/config.toml
# (its theme), which it resolves relative to the working directory.
cd "$DIR"
exec "$DIR/.venv/bin/streamlit" run "$DIR/streamlit_app.py" \
  --server.headless true \
  --server.port "${PORT:-8501}"
