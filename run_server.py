"""
IHearYou — Unified Server (FastAPI + WebSocket + Static)

Menjalankan satu server yang melayani:
  - Frontend HTML/CSS/JS (static)
  - REST API (/health, /api/*, /predict, /stt)
  - WebSocket real-time (/ws dan /ws/predict)

Cara jalankan:
  python run_server.py

Atau manual:
  uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload

Buka: http://localhost:8001
"""
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT / "backend")

print("=" * 55)
print("  IHearYou — Server")
print("  URL  : http://localhost:8001")
print("  WS   : ws://localhost:8001/ws/predict")
print("=" * 55)

subprocess.run([
    sys.executable, "-m", "uvicorn",
    "app.main:app",
    "--host", "0.0.0.0",
    "--port", "8001",
    "--reload",
    "--reload-dir", str(ROOT / "backend"),
], check=True)
