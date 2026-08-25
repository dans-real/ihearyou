# IHearYou Backend

Backend untuk translasi BISINDO real-time berbasis FastAPI + WebSocket, mengikuti alur CRISP-ML(Q):

1. Frontend MediaPipe Holistic mengekstraksi landmark.
2. Backend menerima landmark sequence dan inferensi Sign-to-Speech.
3. Backend menyediakan Speech-to-Text (Whisper).
4. Backend melakukan regional lexical mapping.
5. Monitoring dasar (latensi, request count, error) tersedia untuk observability.

## Menjalankan Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoint HTTP

- `GET /` -> halaman frontend.
- `GET /health` -> status service, status model, jumlah klien websocket aktif.
- `GET /metrics` -> ringkasan monitoring inferensi.
- `GET /regional?region=<nama_region>` -> daftar region + mapping leksikal.
- `POST /predict` -> inferensi sign dari sequence landmark.
- `POST /stt` -> transkripsi audio (Whisper).
- `GET /api/model-info` -> metadata model (num_classes, akurasi, arsitektur).
- `GET /api/classes` -> daftar kelas gesture dari metadata.
- `GET /api/stt-status` -> status engine STT.

### Contoh `POST /predict`

```json
{
	"sequence": [
		{"pose": [], "face": [], "left_hand": [], "right_hand": []}
	],
	"region": "bandung"
}
```

### Contoh `POST /stt`

```json
{
	"audio_data": "data:audio/webm;base64,..."
}
```

## Endpoint WebSocket

- `WS /ws`

Message type yang didukung:

- `set_region`: set region aktif koneksi.
- `landmarks`: kirim landmark frame-by-frame dari MediaPipe JS.
- `audio_chunk`: kirim chunk audio base64 untuk transkripsi.

## Catatan Model

- Placeholder model ada di `models/bisindo_translator.pth`.
- Jika checkpoint belum cocok/tersedia, sistem fallback ke dummy prediction agar demo pipeline tetap berjalan.

## Training Metadata (Kombinasi dengan Folder `files`)

Script `train_model.py` telah terintegrasi untuk melatih pipeline klasik (MLP + feature pre-extracted)
dan menulis output metadata ke `data/model_metadata.json` agar langsung dipakai endpoint:

- `GET /api/model-info`
- `GET /api/classes`

Menjalankan training:

```bash
cd backend
python train_model.py --data-dir data
```

Output default:

- `models/pipeline_mlp.pkl`
- `models/classification_report.txt`
- `data/model_metadata.json`

Catatan: training membutuhkan file dataset pre-extracted di folder data:

- `img_features.pkl`
- `vid_features.pkl`
- `img_classes.json`
- `vid_classes.json`
