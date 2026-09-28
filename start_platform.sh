#!/usr/bin/env bash
# ===============================================================================
# graphforANN Enterprise AML Platform Launcher (Linux / macOS)
# ===============================================================================

echo "==============================================================================="
echo "  graphforANN: Heterogeneous GNN Platform for Enterprise AML Surveillance"
echo "==============================================================================="

if ! command -v python3 &> /dev/null
then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[1/3] Starting FastAPI Secure Backend on http://127.0.0.1:8000 ..."
python3 backend/api.py &
BACKEND_PID=$!

sleep 3

echo "[2/3] Starting Streamlit Research Portal on http://localhost:8501 ..."
streamlit run frontend/app.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!

echo "==============================================================================="
echo "  Platform successfully initialized!"
echo "  - Backend API: http://127.0.0.1:8000/docs (PID: $BACKEND_PID)"
echo "  - Research UI: http://localhost:8501 (PID: $FRONTEND_PID)"
echo "==============================================================================="

trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait
