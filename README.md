# SentimentAI

**SentimentAI** is a real-time facial emotion detection application for macOS, running locally on your apple sillicon devices.
---

## Features

- Real-time facial emotion detection via webcam
- ONNX-optimized deep learning model (no cloud required)
- Modular codebase: add your own models or dashboard extensions

---

### Emotion Recognition Model

- **Filename:** `emotion_model.onnx`
- **Format:** ONNX ([Open Neural Network Exchange](https://onnx.ai/))
- **Input:** 48x48 grayscale face images
- **Classes:** Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral
- **Origin:** Exported from the Keras CNN in [atulapra/Emotion-detection](https://github.com/atulapra/Emotion-detection) using `tf2onnx`
- **Purpose:** Powers real-time emotion inference in the GUI

> Want to retrain or fine-tune?  
> See [atulapra/Emotion-detection](https://github.com/atulapra/Emotion-detection) or use your own dataset/model—just export to ONNX for local inference!
