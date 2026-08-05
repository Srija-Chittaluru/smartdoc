#!/bin/bash
# Start the FastAPI backend and the Streamlit UI together.
# Stop both with Ctrl+C.

set -e
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "No virtual environment found. Run these first:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "Starting backend on http://127.0.0.1:8000 ..."
$PYTHON -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Make sure the backend dies with this script rather than lingering.
trap "kill $BACKEND_PID 2>/dev/null" EXIT

sleep 3
echo "Starting Streamlit UI on http://localhost:8501 ..."
$PYTHON -m streamlit run app.py
