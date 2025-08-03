# main.py
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from face_detection import FaceDetector
from inference      import EmotionClassifier

app        = FastAPI()
detector   = FaceDetector()
classifier = EmotionClassifier()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Receive raw JPEG bytes from client
            data = await ws.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Detect faces & classify each
            emotions = []
            for (x, y, w, h) in detector.detect(frame):
                face = frame[y:y+h, x:x+w]
                emotions.append(classifier.predict(face))

            # Send back a list of emotion indices
            await ws.send_json({"emotions": emotions})

    except WebSocketDisconnect:
        # Client disconnected
        pass

if __name__ == "__main__":
    # Pass the app instance, not the "module:app" string
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)