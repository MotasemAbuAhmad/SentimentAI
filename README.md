Here’s an updated and improved README.md for your project, tailored to the actual code you’ve built.
This emphasizes the API, the modern frontend, and local-only privacy.

⸻


# SentimentAI

**SentimentAI** is a real-time facial emotion detection API and web app—runs **locally on your Mac** (Apple Silicon or Intel), with zero cloud required.

---

## ✨ Features

- **Real-time facial emotion detection** from images (webcam/video via WebSocket coming soon)
- **Modern, intuitive web UI** (drag-and-drop or click-to-upload)
- **ONNX-optimized deep learning model** (fast, private, offline)
- **No cloud, no tracking:** all computation happens on your machine
- **Modular backend:** add your own models, endpoints, or analytics

---

## 🚀 Quick Start

1. **Clone this repo**  
   ```sh
   git clone https://github.com/MotasemAbuAhmad/SentimentAI.git
   cd SentimentAI

2. **Install requirements**
(Recommended: use a virtualenv/venv)

pip install -r requirements.txt


3. **Start the API server**

uvicorn backend.main:app --reload


4. **Visit the web interface**
Go to http://localhost:8000 in your browser.

⸻

🤖 Model Details
	•	Filename: models/emotion_model.onnx
	•	Format: ONNX (Open Neural Network Exchange)
	•	Input: 48x48 grayscale face images
	•	Output Classes: Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral
	•	Source: Exported from the Keras CNN in atulapra/Emotion-detection using tf2onnx
	•	Purpose: Powers real-time emotion recognition in the API & UI

Want to train your own? See atulapra/Emotion-detection or use any model that exports to ONNX.

⸻

📦 API Usage

/predict (POST)

Upload a JPG/PNG image.
Returns: List of faces with bounding boxes and detected emotions.

/ws (WebSocket)

Send video frames as JPEG bytes for real-time streaming detection.
(See code for usage example.)

⸻

🖥️ Frontend
	•	Clean drag-and-drop interface (frontend/index.html, style.css, main.js)
	•	Results: Detected faces and emotions overlaid on your image
	•	Fully local—no internet required!

⸻

🛠️ For Developers
	•	Add your own models: swap out models/emotion_model.onnx
	•	Extend endpoints in backend/main.py
	•	Retrain or export: atulapra/Emotion-detection or your own dataset

⸻

⚡️ Why use SentimentAI?
	•	Private: never sends data to the cloud
	•	Fast: runs on Apple Silicon (M1/M2/M3) or Intel Macs
	•	Open: MIT-licensed, easy to fork or extend

⸻

License

MIT

---
