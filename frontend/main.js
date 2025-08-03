// frontend/main.js
document.addEventListener('DOMContentLoaded', () => {
  // ------- Upload card elements -------
  const dropArea   = document.getElementById('drop-area');
  const fileElem   = document.getElementById('fileElem');
  const imageWrap  = document.getElementById('image-wrap');
  const emotionOut = document.getElementById('emotion-output');
  const selectBtn  = document.getElementById('select-btn');

  // ------- Webcam card elements -------
  const webcamBtn      = document.getElementById('start-webcam');
  const stopWebcamBtn  = document.getElementById('stop-webcam');
  const webcamCont     = document.getElementById('webcam-container');
  const webcam         = document.getElementById('webcam');
  const webcamCanvas   = document.getElementById('webcam-canvas');
  const webcamCtx      = webcamCanvas.getContext('2d');
  const webcamEmotion  = document.getElementById('webcam-emotion');

  let stream = null, ws = null, sendInterval = null;

  // =======================
  // Image upload flow
  // =======================
  selectBtn.onclick = () => fileElem.click();

  dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.classList.add('dragover');
  });
  dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));
  dropArea.addEventListener('drop', e => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });

  fileElem.addEventListener('change', e => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  });

  function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      imageWrap.innerHTML =
        `<img id="preview-img" src="${evt.target.result}"
              style="max-width:420px; max-height:320px; border-radius:12px; box-shadow:0 4px 32px #0003;">`;
      emotionOut.textContent = '⏳ Detecting…';
      setTimeout(() => runPredict(file), 60);
    };
    reader.readAsDataURL(file);
  }

  async function runPredict(file) {
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/predict', { method: 'POST', body: form });
      if (!res.ok) throw new Error('Bad response from /predict');
      const data = await res.json();

      const img = document.getElementById('preview-img');
      // Draw boxes after image has dimensions
      const draw = () => drawBoxesOnImage(data, img);
      img.complete ? draw() : img.addEventListener('load', draw, { once: true });

      if (data.primary_emotion) {
        emotionOut.innerHTML = `<span style="color:#ffc542;font-size:1.1em;">
          Detected: <b>${data.primary_emotion}</b></span>`;
      } else if (data.num_faces === 0) {
        emotionOut.innerHTML = "<span style='color:#f44;'>No face detected.</span>";
      } else {
        emotionOut.textContent = '';
      }
    } catch (err) {
      console.error(err);
      emotionOut.innerHTML = "<span style='color:#f44;'>❌ Could not connect to API.</span>";
    }
  }

  function drawBoxesOnImage(data, img) {
    // clear old overlays
    imageWrap.querySelectorAll('.face-box, .emotion-tag').forEach(el => el.remove());
    if (!Array.isArray(data.faces) || data.faces.length === 0) return;

    // scale server boxes (original image size) to displayed size
    const scaleX = img.clientWidth  / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;

    data.faces.forEach(f => {
      const [x, y, w, h] = f.box;
      const left = x * scaleX, top = y * scaleY, width = w * scaleX, height = h * scaleY;

      const box = document.createElement('div');
      box.className = 'face-box';
      Object.assign(box.style, {
        position: 'absolute', left: `${left}px`, top: `${top}px`,
        width: `${width}px`, height: `${height}px`,
        border: '2.5px solid #60f098', borderRadius: '10px', pointerEvents: 'none'
      });

      const tag = document.createElement('div');
      tag.className = 'emotion-tag';
      Object.assign(tag.style, {
        position: 'absolute', left: `${left}px`, top: `${top - 28}px`,
        background: '#171f2e', color: '#60f098', padding: '3px 13px',
        borderRadius: '10px', fontSize: '1.0em', fontWeight: 'bold',
        pointerEvents: 'none', boxShadow: '0 2px 12px #0002'
      });
      tag.textContent = f.emotion;

      imageWrap.appendChild(box);
      imageWrap.appendChild(tag);
    });
  }

  // =======================
  // Webcam live flow
  // =======================
  webcamBtn.addEventListener('click', startWebcam);
  stopWebcamBtn.addEventListener('click', stopWebcam);

  async function startWebcam() {
    try {
      webcamCont.style.display = '';
      webcamBtn.style.display = 'none';
      stopWebcamBtn.style.display = '';
      webcamEmotion.textContent = 'Detecting…';

      // Request camera & start playback (play() returns a Promise)
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); // secure context required
      webcam.srcObject = stream;
      await new Promise((resolve) => {
        // ensure metadata present, then play
        const onMeta = async () => {
          try { await webcam.play(); resolve(); }
          catch (e) { console.error('play() failed', e); resolve(); }
          webcam.removeEventListener('loadedmetadata', onMeta);
        };
        webcam.addEventListener('loadedmetadata', onMeta);
      }); // MDN: play() returns a Promise you can await.  [oai_citation:2‡MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/play)

      // Set canvas capture size (send smaller frames for lower latency).
      const targetW = 640;
      const targetH = Math.round(targetW * (webcam.videoHeight / webcam.videoWidth)) || 480;
      webcamCanvas.width  = targetW;
      webcamCanvas.height = targetH;

      // Open WS to same origin
      const wsScheme = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${wsScheme}://${location.host}/ws`);

      ws.onopen = () => {
        sendInterval = setInterval(() => {
          // Draw current video frame to canvas, then send as JPEG blob
          webcamCtx.drawImage(webcam, 0, 0, webcamCanvas.width, webcamCanvas.height);
          webcamCanvas.toBlob((blob) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(blob);
          }, 'image/jpeg', 0.8);
        }, 150); // ~6–7 fps; adjust as you like
      };

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        // Always redraw base frame
        webcamCtx.drawImage(webcam, 0, 0, webcamCanvas.width, webcamCanvas.height);

        if (Array.isArray(data.faces)) {
          data.faces.forEach(f => {
            const [x, y, w, h] = f.box; // server returns boxes in the same size as the sent frame
            webcamCtx.strokeStyle = '#60f098';
            webcamCtx.lineWidth = 2;
            webcamCtx.strokeRect(x, y, w, h);

            const text = f.emotion;
            webcamCtx.font = '16px Inter, sans-serif';
            const tw = webcamCtx.measureText(text).width;
            webcamCtx.fillStyle = 'rgba(23,31,46,0.85)';
            webcamCtx.fillRect(x, y - 24, tw + 12, 20);
            webcamCtx.fillStyle = '#60f098';
            webcamCtx.fillText(text, x + 6, y - 9);
          });

          if (data.faces.length) {
            webcamEmotion.innerHTML = `Detected: <b>${data.faces[0].emotion}</b>`;
          } else {
            webcamEmotion.textContent = '';
          }
        }
      };

      ws.onclose = stopWebcam;
      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        webcamEmotion.textContent = 'WebSocket error.';
      };

    } catch (err) {
      console.error(err);
      // getUserMedia requires HTTPS or localhost and user permission.  [oai_citation:3‡MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
      webcamEmotion.textContent = 'Camera unavailable or permission denied.';
      stopWebcam();
    }
  }

  function stopWebcam() {
    stopWebcamBtn.style.display = 'none';
    webcamBtn.style.display = '';
    webcamCont.style.display = 'none';
    webcamEmotion.textContent = '';
    if (sendInterval) clearInterval(sendInterval);
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      try { ws.close(); } catch {}
    }
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
  }
});