/**
 * IHearYou — app.js v4
 *
 * FIXES v4:
 *  - WS_URL menggunakan port yang sama dengan halaman (tidak hardcode 8001/5000)
 *    → bekerja baik via Flask maupun uvicorn tanpa edit manual
 *  - Fitur REGION dihapus sepenuhnya (select, payload, tampilan)
 *  - history.region dihapus dari render
 *  - loadModelInfo() tidak lagi mencoba populate region-select
 *  - region tidak lagi dikirim ke server
 */

// ── Config ────────────────────────────────────────────────────────────────────
// Prioritas backend:
// 1) query param ?api=https://your-backend
// 2) localStorage: ihear_api
// 3) origin saat ini
const API_QUERY = new URLSearchParams(location.search).get('api');
const API_OVERRIDE = (API_QUERY || localStorage.getItem('ihear_api') || '').trim();
const DEFAULT_BASE = `${location.protocol}//${location.host}`;
const API_BASE = API_OVERRIDE ? API_OVERRIDE.replace(/\/+$/, '') : DEFAULT_BASE;

let apiUrl;
try {
  apiUrl = new URL(API_BASE);
} catch {
  apiUrl = new URL(DEFAULT_BASE);
}

const WS_URL = `${apiUrl.protocol === 'https:' ? 'wss' : 'ws'}://${apiUrl.host}/ws/predict`;
const IS_GH_PAGES = location.hostname.endsWith('github.io');

// ── State ─────────────────────────────────────────────────────────────────────
let ws = null, useWS = false;
let holisticMain = null, holisticGL = null;
let camMain = null, camGL = null;
let scanTimer = null, scanInterval = 1000;
let glRecognizing = false, glRecording = false;
let history = [];
let ttsEnabled = true;
let image_model_ready = false;   // set true setelah /health confirm model loaded

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    $(`tab-${tab}`).classList.add('active');
    if (tab === 'gesture' && !holisticGL) initGLCamera();
    if (tab === 'stt') initSTT();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// 1. WEBSOCKET
// ═══════════════════════════════════════════════════════════════════════════════
function connectWS() {
  const sock = new WebSocket(WS_URL);

  sock.onopen = () => {
    ws = sock; useWS = true;
    setChip($('chip-ws'), '● WS Terhubung', 'connected');
    loadModelInfo();
    loadClasses();
    loadGestureList();
    initMainCamera();
    // Cek health setelah 500ms agar server startup complete
    setTimeout(async () => {
      try {
        const r = await fetch(`${API_BASE}/health`);
        const d = await r.json();
        image_model_ready = !!d.image_model_loaded;
        if (!image_model_ready)
          showHint('⚠ Model belum loaded — jalankan training dulu');
      } catch { image_model_ready = true; } // fallback: coba saja
    }, 500);
  };

  sock.onmessage = ev => {
    try { handleServerMessage(JSON.parse(ev.data)); } catch(e) { console.warn('WS parse error', e); }
  };

  sock.onerror = () => { useWS = false; };

  sock.onclose = () => {
    useWS = false;
    ws = null;
    setChip($('chip-ws'), '↻ Menyambungkan…', '');
    setTimeout(connectWS, 3000);
  };
}

function sendWS(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
    return true;
  }
  return false;
}

async function checkBackend() {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    if (d.image_model_loaded) {
      image_model_ready = true;
    } else {
      showHint('⚠ Model belum loaded — pastikan pipeline_mlp.pkl ada di backend/models/');
    }
  } catch {
    setChip($('chip-ws'), '✕ Backend tidak ditemukan', 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. HANDLE SERVER MESSAGES
// ═══════════════════════════════════════════════════════════════════════════════
function handleServerMessage(data) {
  const type = data.type;

  if (type === 'error') {
    showHint('⚠ ' + data.error);
    return;
  }

  if (type === 'translation') {
    const source = data.source || 'model';
    if (source === 'gesture_learning') {
      handleGLTranslation(data);
    } else {
      handleTranslation(data);
    }
    return;
  }

  // Kandidat prediksi (belum cukup stabil) — tampilkan sebagai preview
  if (type === 'prediction_candidate') {
    const wordEl = $('current-word');
    if (wordEl) {
      wordEl.textContent = data.top_prediction + '?';
      wordEl.style.opacity = '0.45';
    }
    const confFill = $('conf-fill');
    const confPct  = $('conf-pct');
    if (confFill) confFill.style.width = Math.round(data.confidence * 100) + '%';
    if (confPct)  confPct.textContent  = Math.round(data.confidence * 100) + '% (menunggu ' + data.frames_needed + ' frame)';
    $('conf-card').style.display = 'block';
    return;
  }

  if (type === 'transcription') {
    appendTranscript(data.text);
    return;
  }

  if (type === 'gesture_record_started') {
    $('gl-badge').textContent = `⏺ Merekam "${data.label}"`;
    $('rec-label').textContent = `Merekam: ${data.label}`;
    $('rec-overlay').style.display = 'flex';
    return;
  }

  if (type === 'gesture_recording') {
    $('rec-frames').textContent = `${data.frames} frame`;
    return;
  }

  if (type === 'gesture_saved') {
    $('gl-badge').textContent = `✓ Tersimpan`;
    $('rec-overlay').style.display = 'none';
    showGLStatus(`✓ Gesture "${data.label}" tersimpan (${data.frames} frame, ${data.count} rekaman)`);
    if (data.model_note) showGLStatus(data.model_note);
    loadGestureList();
    return;
  }

  if (type === 'gesture_list') {
    renderGestureList(data.labels);
    return;
  }

  if (type === 'sentence_cleared') {
    clearSentence();
    return;
  }
}

// ── Translation (Sign-to-Speech) ──────────────────────────────────────────────
function handleTranslation(data) {
  const word    = data.top_prediction || '—';
  const conf    = data.confidence ?? 0;
  const top5    = data.top5 || [];
  const lat     = data.latency_ms ?? 0;
  const wordBuf = data.word_buffer || [];
  const sentence= data.sentence || '';
  const sentComplete = data.sentence_complete || false;

  $('conf-card').style.display = 'block';
  const wordEl = $('current-word');
  wordEl.style.opacity = '1';   // reset dari candidate preview
  wordEl.classList.remove('is-empty', 'result-pop');
  void wordEl.offsetWidth;
  wordEl.classList.add('result-pop');
  wordEl.textContent = word;

  const pct = Math.round(conf * 100);
  $('conf-fill').style.width = pct + '%';
  $('conf-pct').textContent  = pct + '%';

  const lb = $('latency-badge');
  lb.textContent        = lat + 'ms';
  lb.style.background   = lat < 200 ? '#d1fae5' : '#fef3c7';
  lb.style.color        = lat < 200 ? '#065f46' : '#92400e';
  lb.style.borderRadius = '99px';
  lb.style.padding      = '.2rem .55rem';
  lb.style.fontSize     = '.72rem';
  lb.style.fontWeight   = '600';

  if (top5.length) {
    $('top5-output').innerHTML = top5.map((p, i) => `
      <div class="t5-item ${i===0?'first':''}">
        <span class="t5-rank">#${i+1}</span>
        <span class="t5-label">${p.label}</span>
        <div class="t5-bar"><div class="t5-bar-fill" style="width:${Math.round(p.confidence*100)}%"></div></div>
        <span class="t5-conf">${(p.confidence*100).toFixed(1)}%</span>
      </div>`).join('');
  }

  updateSentenceDisplay(wordBuf, sentence, 'sentence-output', 'sent-ph', 'word-chips');

  if (data.speak_word && ttsEnabled) speakText(data.speak_word);
  if (sentComplete && data.speak_sentence && ttsEnabled) {
    setTimeout(() => speakText(data.speak_sentence), 600);
  }

  // History tanpa region
  addHistory(word, pct);
}

// ── Gesture translation ────────────────────────────────────────────────────────
function handleGLTranslation(data) {
  const word     = data.top_prediction || '—';
  const wordBuf  = data.word_buffer || [];
  const sentence = data.sentence || '';
  const sentComplete = data.sentence_complete || false;

  updateSentenceDisplay(wordBuf, sentence, 'gl-sent-output', 'gl-ph', 'gl-word-chips');

  if (data.speak_word && ttsEnabled) speakText(data.speak_word);
  if (sentComplete && data.speak_sentence && ttsEnabled) {
    setTimeout(() => speakText(data.speak_sentence), 600);
  }
}

function updateSentenceDisplay(wordBuf, sentence, outputId, phId, chipsId) {
  const outputEl = $(outputId);
  const phEl     = $(phId);
  const chipsEl  = $(chipsId);

  if (sentence) {
    outputEl.textContent = sentence;
    if (phEl) phEl.style.display = 'none';
  }

  if (chipsEl && wordBuf.length) {
    chipsEl.innerHTML = wordBuf.map((w, i) =>
      `<span class="word-chip ${i === wordBuf.length-1 ? 'new' : ''}">${w}</span>`
    ).join('');
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. MEDIAPIPE — MAIN CAMERA
// ═══════════════════════════════════════════════════════════════════════════════
async function initMainCamera() {
  holisticMain = new Holistic({
    locateFile: f =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${f}`
  });
  holisticMain.setOptions({
    modelComplexity: 0,
    smoothLandmarks: true,
    enableSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  holisticMain.onResults(onMainResults);

  const video = $('camera-feed');
  // Kamera langsung aktif — scan dimulai segera tanpa tunggu Holistic
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' }, audio: false
  });
  video.srcObject = stream;

  // Hide overlay SEGERA — tidak perlu tunggu metadata
  $('cam-overlay').classList.add('hidden');
  $('cam-overlay').style.display = 'none';

  video.onloadedmetadata = () => {
    video.play().catch(() => {});
    $('camera-wrap').classList.add('is-scanning');
    startAutoScan();
  };

  // Fallback: jika onloadedmetadata lambat, tetap play setelah 1 detik
  setTimeout(() => {
    if (video.paused) video.play().catch(() => {});
    $('camera-wrap').classList.add('is-scanning');
    startAutoScan();
  }, 1000);

  // Holistic diinit secara paralel untuk drawing saja (opsional)
  // Jika gagal load CDN, scan tetap berjalan
  try {
    camMain = new Camera(video, {
      onFrame: async () => {
        if (holisticMain) await holisticMain.send({ image: video });
      },
      width: 640, height: 480
    });
    camMain.start().catch(() => {});  // silent fail — tidak blokir scan
  } catch (e) {}
}

let lastMainResults = null, mainHandDetected = false;
function onMainResults(results) {
  drawOnCanvas(results, $('output-canvas'), $('camera-feed'));
  lastMainResults = results;

  // Visual feedback: tangan terdeteksi atau tidak
  const hasHand = !!(results.leftHandLandmarks || results.rightHandLandmarks);
  if (hasHand !== mainHandDetected) {
    mainHandDetected = hasHand;
    const badge = $('hand-badge');
    if (badge) {
      badge.textContent = hasHand ? '✋ Tangan terdeteksi' : '⬜ Tidak ada tangan';
      badge.style.color = hasHand ? '#10b981' : '#6b7280';
    }
  }
}

let scanCount = 0, scanStarted = false;
function startAutoScan() {
  if (scanStarted) return;   // jangan double-start
  scanStarted = true;
  clearInterval(scanTimer);
  scanInterval = parseInt($('scan-interval').value || '1000');
  if (scanInterval <= 0) return;
  scanTimer = setInterval(() => {
    captureAndSend();
    scanCount++;
    // Update latency badge tiap 5 scan
    if (scanCount % 5 === 0) {
      const lb = $('latency-badge');
      if (lb && !lb.textContent.includes('ms'))
        lb.textContent = `scan #${scanCount}`;
    }
  }, scanInterval);
}

$('scan-interval').addEventListener('change', () => {
  scanInterval = parseInt($('scan-interval').value);
  clearInterval(scanTimer);
  if (scanInterval > 0) scanTimer = setInterval(() => captureAndSend(), scanInterval);
});

function captureAndSend() {
  if (!useWS) return;
  const video = $('camera-feed');
  // Kirim langsung dari video — tidak perlu tunggu Holistic onResults
  // Backend model berbasis IMAGE, bukan landmark
  if (!video || video.readyState < 2) return;   // video belum siap
  const b64 = captureFrame(video);
  if (!b64) return;

  // Sertakan landmark jika tersedia (untuk gesture learning), tapi opsional
  const getLM = lm => lm ? lm.map(p => ({ x: p.x, y: p.y, z: p.z ?? 0 })) : [];
  const landmarks = lastMainResults ? {
    pose:       getLM(lastMainResults.poseLandmarks),
    left_hand:  getLM(lastMainResults.leftHandLandmarks),
    right_hand: getLM(lastMainResults.rightHandLandmarks),
    face:       []
  } : { pose: [], left_hand: [], right_hand: [], face: [] };

  sendWS({ type: 'predict', image_frame: b64, landmarks });
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. GESTURE LEARNING CAMERA
// ═══════════════════════════════════════════════════════════════════════════════
async function initGLCamera() {
  holisticGL = new Holistic({
    locateFile: f =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${f}`
  });
  holisticGL.setOptions({
    modelComplexity: 0,
    smoothLandmarks: true,
    enableSegmentation: false,
    refineFaceLandmarks: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  holisticGL.onResults(onGLResults);

  const video = $('gl-feed');
  // Kamera GL langsung aktif
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }, audio: false
    });
    video.srcObject = stream;
    // Hide GL overlay SEGERA
    $('gl-overlay').classList.add('hidden');
    $('gl-overlay').style.display = 'none';

    video.onloadedmetadata = () => {
      video.play().catch(() => {});
      $('gl-camera-wrap').classList.add('is-scanning');
    };

    setTimeout(() => {
      if (video.paused) video.play().catch(() => {});
      $('gl-camera-wrap').classList.add('is-scanning');
    }, 1000);
  } catch (err) {
    $('gl-overlay').querySelector('p').textContent = 'Kamera GL: ' + err.message;
    return;
  }

  // Holistic untuk GL (drawing) — opsional
  try {
    camGL = new Camera(video, {
      onFrame: async () => { if (holisticGL) await holisticGL.send({ image: video }); },
      width: 640, height: 480
    });
    camGL.start().catch(() => {});
  } catch (e) {}
}

let lastGLResults = null, glFrameTick = 0, glFramesSent = 0;
function onGLResults(results) {
  drawOnCanvas(results, $('gl-canvas'), $('gl-feed'));
  lastGLResults = results;
  glFrameTick++;

  const getLM = lm => lm ? lm.map(p => ({ x: p.x, y: p.y, z: p.z ?? 0 })) : [];

  // Cek apakah ada tangan yang terdeteksi
  const hasHand = !!(results.leftHandLandmarks || results.rightHandLandmarks);

  const landmarks = {
    pose:       getLM(results.poseLandmarks),
    left_hand:  getLM(results.leftHandLandmarks),
    right_hand: getLM(results.rightHandLandmarks),
    face:       []
  };

  // Recording: kirim SETIAP frame (tidak perlu hasHand dari Holistic)
  if (glRecording) {
    const glVideo = $('gl-feed');
    const glB64   = (glVideo && glVideo.readyState >= 2) ? captureFrame(glVideo) : '';
    if (glB64) {
      glFramesSent++;
      sendWS({ type: 'gesture_frame', landmarks, image_frame: glB64 });
      const recFramesEl = $('rec-frames');
      if (recFramesEl) recFramesEl.textContent = glFramesSent + ' frame';
    }
  }

  // Recognizing: setiap 3 tick
  if (glRecognizing && glFrameTick % 3 === 0) {
    const glVideo = $('gl-feed');
    const glB64   = (glVideo && glVideo.readyState >= 2) ? captureFrame(glVideo) : '';
    if (glB64) sendWS({ type: 'gesture_recognize', landmarks, image_frame: glB64 });
  }
}

$('btn-rec-start').addEventListener('click', () => {
  const label = $('gl-label').value.trim();
  if (!label) { showGLStatus('⚠ Masukkan nama gesture terlebih dahulu'); return; }
  if (!useWS) { showGLStatus('⚠ Server belum terhubung'); return; }

  glRecording = true;
  glFramesSent = 0;
  $('btn-rec-start').style.display = 'none';
  $('btn-rec-stop').style.display  = 'flex';
  $('gl-badge').textContent = '⏺ Merekam — tunjukkan gesture ke kamera';
  sendWS({ type: 'gesture_record_start', label });

  // Fallback: jika Holistic tidak aktif, kirim frame via interval mandiri
  if (!holisticGL || !camGL) {
    window._glRecInterval = setInterval(() => {
      if (!glRecording) { clearInterval(window._glRecInterval); return; }
      const glVideo = $('gl-feed');
      if (!glVideo || glVideo.readyState < 2) return;
      glFramesSent++;
      const b64 = captureFrame(glVideo);
      sendWS({ type: 'gesture_frame', landmarks: { pose:[], left_hand:[], right_hand:[], face:[] }, image_frame: b64 });
      const el = $('rec-frames');
      if (el) el.textContent = glFramesSent + ' frame';
    }, 100);  // 10 fps
  }
});

$('btn-rec-stop').addEventListener('click', () => {
  const label = $('gl-label').value.trim();
  glRecording = false;
  $('btn-rec-start').style.display = 'flex';
  $('btn-rec-stop').style.display  = 'none';
  $('gl-badge').textContent = `Menyimpan ${glFramesSent} frame…`;
  if (glFramesSent < 5) {
    showGLStatus('⚠ Terlalu sedikit frame — tunjukkan tangan ke kamera saat merekam');
    return;
  }
  sendWS({ type: 'gesture_record_stop', label, image_frame: lastGLResults ? captureFrame($('gl-feed')) : '' });
});

$('btn-gl-recog').addEventListener('click', () => {
  glRecognizing = !glRecognizing;
  const btn = $('btn-gl-recog');
  if (glRecognizing) {
    btn.classList.add('active');
    btn.textContent = '⏹ Stop Kenali';
    $('gl-badge').textContent = '🎯 Mengenali…';
    showGLStatus('Pose gesture ke kamera untuk dikenali…');
  } else {
    btn.classList.remove('active');
    btn.textContent = '🎯 Kenali Gesture';
    $('gl-badge').textContent = 'Standby';
  }
});

function loadGestureList() { sendWS({ type: 'gesture_list' }); }

function renderGestureList(labels) {
  const el = $('gesture-list');
  if (!labels || !labels.length) {
    el.innerHTML = '<div class="list-empty">Belum ada gesture tersimpan</div>';
    return;
  }
  el.innerHTML = labels.map(label => `
    <div class="gl-item">
      <div class="gl-item-info">
        <span class="gl-label">${label}</span>
        <span class="gl-meta">Klik hapus untuk menghapus</span>
      </div>
      <button class="gl-del" onclick="deleteGesture('${label.replace(/'/g,"\\'")}')">✕ Hapus</button>
    </div>`).join('');
}

window.deleteGesture = label => {
  if (confirm(`Hapus gesture "${label}"?`)) {
    sendWS({ type: 'gesture_delete', label });
    setTimeout(loadGestureList, 400);
  }
};

$('btn-refresh-gl').addEventListener('click', loadGestureList);

$('btn-speak-gl').addEventListener('click', () => {
  speakText($('gl-sent-output').textContent.trim());
});
$('btn-clear-gl').addEventListener('click', () => {
  $('gl-sent-output').textContent = '';
  $('gl-word-chips').innerHTML = '';
  $('gl-ph').style.display = 'block';
});

function showGLStatus(msg) {
  const el = $('gl-badge');
  el.textContent = msg;
  setTimeout(() => { if (el.textContent === msg) el.textContent = 'Standby'; }, 3000);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. SENTENCE CONTROLS
// ═══════════════════════════════════════════════════════════════════════════════
$('btn-speak-sentence').addEventListener('click', () => {
  speakText($('sentence-output').textContent.trim());
});
$('btn-clear-sentence').addEventListener('click', () => {
  sendWS({ type: 'clear_sentence' });
  clearSentence();
});

function clearSentence() {
  $('sentence-output').textContent = '';
  $('word-chips').innerHTML = '';
  $('sent-ph').style.display = 'block';
}

$('btn-clear-history').addEventListener('click', () => {
  history = [];
  renderHistory();
  clearSentence();
  $('conf-card').style.display = 'none';
  $('top5-output').innerHTML = '';
});

// ═══════════════════════════════════════════════════════════════════════════════
// 6. TTS
// ═══════════════════════════════════════════════════════════════════════════════
function speakText(text) {
  if (!text || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang  = 'id-ID'; u.rate = 0.9; u.pitch = 1;
  const v = speechSynthesis.getVoices().find(v => v.lang.startsWith('id'));
  if (v) u.voice = v;
  speechSynthesis.speak(u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. STT
// ═══════════════════════════════════════════════════════════════════════════════
let sttInitialized = false;
function initSTT() {
  if (sttInitialized) return;
  sttInitialized = true;
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SR) startWebSpeech(SR);
  else startWhisperSTT();
}

function startWebSpeech(SR) {
  const rec = new SR();
  rec.lang = 'id-ID'; rec.continuous = true; rec.interimResults = true;
  rec.onresult = e => {
    let final = '';
    for (let i = e.resultIndex; i < e.results.length; i++)
      if (e.results[i].isFinal) final += e.results[i][0].transcript;
    if (final) appendTranscript(final);
  };
  rec.onstart = () => {
    $('stt-status').textContent = 'Web Speech API aktif · id-ID';
    if ($('stt-engine-badge')) $('stt-engine-badge').textContent = 'Web Speech API';
  };
  rec.onend = () => { setTimeout(() => { try { rec.start(); } catch {} }, 500); };
  rec.onerror = e => { $('stt-status').textContent = 'STT error: ' + e.error; };
  try { rec.start(); } catch {}
}

async function startWhisperSTT() {
  $('stt-status').textContent = 'Meminta akses mikrofon…';
  if ($('stt-engine-badge')) $('stt-engine-badge').textContent = 'Whisper';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    $('stt-status').textContent = 'Whisper · Mikrofon aktif';
    function chunk() {
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      const chunks = [];
      mr.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
      mr.onstop = async () => {
        if (!chunks.length) return;
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const b64  = await new Promise((res, rej) => {
          const r = new FileReader();
          r.onload  = () => res(r.result.split(',')[1]);
          r.onerror = rej;
          r.readAsDataURL(blob);
        });
        try {
          const d = await fetch(`${API_BASE}/api/stt`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audio: b64 })
          }).then(r => r.json());
          if (d.text?.trim()) appendTranscript(d.text.trim());
        } catch {}
      };
      mr.start();
      setTimeout(() => { if (mr.state === 'recording') mr.stop(); }, 4000);
    }
    chunk();
    setInterval(chunk, 4600);
  } catch {
    $('stt-status').textContent = 'Mikrofon tidak tersedia';
  }
}

function appendTranscript(text) {
  const out = $('transcript-output');
  const ph  = $('t-ph');
  if (ph) ph.style.display = 'none';
  out.textContent += (out.textContent ? ' ' : '') + text;
}

$('btn-speak-transcript').addEventListener('click', () => speakText($('transcript-output').textContent.trim()));
$('btn-clear-transcript').addEventListener('click', () => {
  $('transcript-output').textContent = '';
  if ($('t-ph')) $('t-ph').style.display = 'block';
});
$('btn-copy-transcript').addEventListener('click', () => {
  const text = $('transcript-output').textContent.trim();
  if (text) navigator.clipboard.writeText(text).catch(() => {});
});

// ═══════════════════════════════════════════════════════════════════════════════
// 8. MEDIAPIPE HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
function drawOnCanvas(results, canvas, video) {
  if (!canvas || typeof drawConnectors === 'undefined') return;
  const ctx = canvas.getContext('2d');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.poseLandmarks)
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: 'rgba(37,99,235,.3)', lineWidth: 1.5 });
  if (results.leftHandLandmarks) {
    drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: 'rgba(5,150,105,.6)', lineWidth: 2 });
    drawLandmarks(ctx, results.leftHandLandmarks, { color: 'rgba(52,211,153,.9)', lineWidth: 1, radius: 3 });
  }
  if (results.rightHandLandmarks) {
    drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: 'rgba(124,58,237,.6)', lineWidth: 2 });
    drawLandmarks(ctx, results.rightHandLandmarks, { color: 'rgba(167,139,250,.9)', lineWidth: 1, radius: 3 });
  }
}

function captureFrame(video) {
  const c = document.createElement('canvas');
  c.width = video.videoWidth || 640;
  c.height = video.videoHeight || 480;
  const ctx = c.getContext('2d');
  ctx.save(); ctx.scale(-1, 1); ctx.drawImage(video, -c.width, 0); ctx.restore();
  return c.toDataURL('image/jpeg', 0.75).split(',')[1];
}

// ═══════════════════════════════════════════════════════════════════════════════
// 9. HISTORY (tanpa region)
// ═══════════════════════════════════════════════════════════════════════════════
function addHistory(word, conf) {
  history.unshift({ word, conf });
  if (history.length > 30) history.pop();
  renderHistory();
}

function renderHistory() {
  const el = $('history-list');
  el.innerHTML = history.length
    ? history.map(h => `
        <div class="h-item" onclick="speakText('${h.word.replace(/'/g,"\\'")}')">
          <span class="h-word">${h.word}</span>
          <span class="h-conf">${h.conf}%</span>
        </div>`).join('')
    : '<div class="list-empty">Belum ada translasi</div>';
}

// ═══════════════════════════════════════════════════════════════════════════════
// 10. UI HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
function setChip(el, text, cls) {
  if (!el) return;
  el.textContent = text;
  el.className = 'chip chip--ws' + (cls ? ` ${cls}` : '');
}

function showHint(msg) {
  const chip = $('chip-ws');
  if (chip && msg.includes('⚠')) {
    const orig = chip.textContent;
    chip.textContent = msg.slice(0, 40);
    setTimeout(() => { if (chip.textContent === msg.slice(0,40)) chip.textContent = orig; }, 4000);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 11. MODEL INFO + VOCAB (tanpa region)
// ═══════════════════════════════════════════════════════════════════════════════
async function loadModelInfo() {
  try {
    const d = await fetch(`${API_BASE}/api/model-info`).then(r => r.json());
    if (d.test_accuracy)  { $('chip-acc').textContent   = (d.test_accuracy * 100).toFixed(1) + '%'; $('m-acc').textContent  = (d.test_accuracy * 100).toFixed(1) + '%'; }
    if (d.num_classes)    { $('chip-classes').textContent = d.num_classes + ' kelas';               $('m-kelas').textContent = d.num_classes; }
    if (d.test_macro_f1)  $('m-f1').textContent   = d.test_macro_f1.toFixed(4);
    if (d.top5_accuracy)  $('m-top5').textContent = (d.top5_accuracy * 100).toFixed(1) + '%';
    if (d.architecture)
      $('model-info-output').innerHTML =
        `<strong>Arsitektur:</strong> ${d.architecture}<br>` +
        `<strong>Dataset:</strong> BISINDO · ${d.train_samples||'—'} train / ${d.val_samples||'—'} val / ${d.test_samples||'—'} test<br>` +
        `<strong>Split:</strong> Signer-independent 70/15/15`;
  } catch { /* silent fail */ }
}

async function loadClasses() {
  try {
    const d = await fetch(`${API_BASE}/api/classes`).then(r => r.json());
    $('vocab-cloud').innerHTML = (d.classes || []).map(c =>
      `<span class="vocab-tag" onclick="speakText('${c.replace(/'/g,"\\'")}')">
         ${c}</span>`).join('');
  } catch {
    $('vocab-cloud').innerHTML = '<span style="color:var(--subtle);font-size:.8rem">—</span>';
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 12. INIT
// ═══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  if (IS_GH_PAGES && !API_OVERRIDE) {
    setChip($('chip-ws'), '⚠ Backend belum dikonfigurasi', 'error');
    showHint('⚠ Tambahkan ?api=https://backend-anda atau jalankan lokal');
    return;
  }
  checkBackend();
  connectWS();
});
