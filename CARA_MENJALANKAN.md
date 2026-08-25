# IHearYou — Cara Menjalankan

## ⚠️ PENTING: Gunakan uvicorn, BUKAN Flask

Server yang benar adalah **FastAPI + uvicorn** (mendukung WebSocket).  
`run_server.py` lama menggunakan Flask yang **tidak mendukung WebSocket** → fitur translasi tidak berjalan.

## Langkah Menjalankan

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Jalankan server (dari root project)
```bash
python run_server.py
```
atau manual:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Buka browser
```
http://localhost:8001
```

## Kenapa Error WebSocket?

Error `WebSocket connection to 'ws://localhost:5000/ws/predict' failed` terjadi karena:
1. Server lama (Flask) berjalan di port 5000 dan **tidak mendukung WebSocket**
2. `app.js` mencoba connect ke `/ws/predict` yang tidak ada di Flask
3. **Fix v4**: WS URL sekarang dinamis (`location.host`) — bekerja di port apapun
4. `run_server.py` sekarang menjalankan uvicorn FastAPI yang mendukung WS

## Perubahan v4

- ✅ `run_server.py` → uvicorn FastAPI (bukan Flask)
- ✅ `app.js` WS_URL dinamis menggunakan `location.host`
- ✅ Fitur Region dihapus dari UI, JS, dan backend
- ✅ `apply_regional_mapping()` dikembalikan sebagai passthrough
