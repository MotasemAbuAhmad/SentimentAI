# backend/main.py
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.face_detection import FaceDetector
from backend.inference import EmotionClassifier

EMOTION_CLASSES = ["Angry", "Disgusted", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]

app = FastAPI()

# CORS (adjust for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend paths
BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# --- API & WS (define BEFORE static mounts) ---
detector = FaceDetector()
classifier = EmotionClassifier()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    faces = detector.detect(frame)
    results = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        idx = classifier.predict(face)
        results.append({
            "emotion": EMOTION_CLASSES[idx],
            "box": [int(x), int(y), int(w), int(h)],
            "index": int(idx),
        })

    resp = {"num_faces": len(results), "faces": results}
    if results:
        resp["primary_emotion"] = results[0]["emotion"]
    return resp

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            faces = detector.detect(frame)
            results = []
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                idx = classifier.predict(face)
                results.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "emotion": EMOTION_CLASSES[idx],
                    "index": int(idx),
                })
            await ws.send_json({"num_faces": len(results), "faces": results})
    except WebSocketDisconnect:
        pass

# --- SPA HTML at "/" ---
@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# --- Static assets under /static ---
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)