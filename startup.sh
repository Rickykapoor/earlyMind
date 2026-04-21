#!/bin/bash
# =============================================================================
# startup.sh — Launch FastAPI (Uvicorn) + Streamlit concurrently
# HuggingFace Spaces uses this as the container entrypoint.
# =============================================================================

set -e

echo "[EarlyMind] Starting services..."
echo "[EarlyMind] Device: $EARLYMIND_DEVICE"
echo "[EarlyMind] Checkpoint dir: $EARLYMIND_CKPT_DIR"

# Wait a moment for any volume mounts to settle
sleep 2

# Start FastAPI (Uvicorn) in background
echo "[EarlyMind] Starting FastAPI on :8000..."
uvicorn api.main:app \
    --host "0.0.0.0" \
    --port 8000 \
    --log-level info \
    &

UVICORN_PID=$!

# Start Streamlit in background
echo "[EarlyMind] Starting Streamlit on :8501..."
streamlit run app.py \
    --server.port 8501 \
    --server.address "0.0.0.0" \
    --server.headless true \
    --browser.gatherUsageStats false \
    &

STREAMLIT_PID=$!

echo "[EarlyMind] All services started."
echo "[EarlyMind] FastAPI PID: $UVICORN_PID"
echo "[EarlyMind] Streamlit PID: $STREAMLIT_PID"
echo "[EarlyMind] API docs: http://localhost:8000/docs"
echo "[EarlyMind] Streamlit UI: http://localhost:8501"

# Keep container alive, forward signals
trap "kill $UVICORN_PID $STREAMLIT_PID 2>/dev/null; exit 0" SIGTERM SIGINT

wait
