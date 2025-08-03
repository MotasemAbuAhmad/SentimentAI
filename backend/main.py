# backend/main.py

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

# Allow CORS for all origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS, etc) at /static
app.mount("/static", StaticFiles(directory="frontend"), name="static")

detector = FaceDetector()
classifier = EmotionClassifier()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    POST an image file (jpg/png) and get detected faces + emotion predictions.
    """
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
        emotion = EMOTION_CLASSES[idx]
        results.append({
            "emotion": emotion,      # <---- Emotion FIRST!
            "box": [int(x), int(y), int(w), int(h)],
            "index": int(idx)
        })

    response = {
        "num_faces": len(results),
        "faces": results,
    }

    # Add 'primary_emotion' if there's one face or you want to always highlight one
    if len(results) == 1:
        response["primary_emotion"] = results[0]["emotion"]
    elif len(results) > 1:
        response["primary_emotion"] = results[0]["emotion"]  # You can pick the first, or implement confidence if available

    return response

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time video (clients send JPEG frames as bytes)
    """
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
                emotion = EMOTION_CLASSES[idx]
                results.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "emotion": emotion,
                    "index": int(idx)
                })
            await ws.send_json({"faces": results, "num_faces": len(results)})
    except WebSocketDisconnect:
        pass

@app.get("/")
def root():
    # Serve the main HTML file from the frontend directory
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)