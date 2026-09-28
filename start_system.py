"""
start_system.py
===============
Unified secure launcher for the AMECF 3-Tier Enterprise System.
This script launches the secure FastAPI backend and the Streamlit frontend.
"""

import subprocess
import time
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_system():
    print("=" * 60)
    print("   STARTING SECURE AMECF ENTERPRISE PLATFORM")
    print("=" * 60)
    print("\n[1/2] Starting Secure FastAPI Backend...")
    
    # Start Backend
    backend_env = os.environ.copy()
    backend_process = subprocess.Popen(
        [sys.executable, str(ROOT / "backend" / "api.py")],
        env=backend_env
    )
    
    # Wait for backend to initialize
    time.sleep(3)
    
    if backend_process.poll() is not None:
        print("[ERROR] Backend failed to start. Check logs.")
        sys.exit(1)
        
    print("\n[2/2] Starting Secure Streamlit Frontend...")
    
    # Start Frontend
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "app.py")],
        env=backend_env
    )
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n[!] Shutting down secure system...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("[!] Shutdown complete.")

if __name__ == "__main__":
    run_system()
