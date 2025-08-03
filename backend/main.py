# main.py
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

from backend.face_detection import FaceDetector
from backend.inference import EmotionClassifier

# backend/main.py
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Allow CORS for all origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Emotion class labels (used for API responses)
EMOTION_CLASSES = ["Angry", "Disgusted", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]

app = FastAPI()
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
            "box": [int(x), int(y), int(w), int(h)],
            "emotion": emotion,
            "index": int(idx)
        })
    return {
        "faces": results,
        "num_faces": len(results)
    }

@app.websocket("/ws")
# backend/main.py
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            results = []
            for (x, y, w, h) in detector.detect(frame):
                face = frame[y:y+h, x:x+w]
                emotion_idx = classifier.predict(face)
                results.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "emotion": int(emotion_idx)
                })

            await ws.send_json({"faces": results})
    except WebSocketDisconnect:
        pass

@app.get("/")
def root():
    return {"message": "SentimentAI backend is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)