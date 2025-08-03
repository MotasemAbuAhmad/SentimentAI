# main.py
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

from face_detection import FaceDetector
from inference import EmotionClassifier

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
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time streaming—send raw JPEG bytes, receive emotions.
    """
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            emotions = []
            for (x, y, w, h) in detector.detect(frame):
                face = frame[y:y+h, x:x+w]
                idx = classifier.predict(face)
                emotions.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "emotion": EMOTION_CLASSES[idx],
                    "index": int(idx)
                })
            await ws.send_json({"faces": emotions})
    except WebSocketDisconnect:
        pass

@app.get("/")
def root():
    return {"message": "SentimentAI backend is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)