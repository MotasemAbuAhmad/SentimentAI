#inference.py
import cv2
import numpy as np
import onnxruntime as ort

class EmotionClassifier:
    def __init__(self, model_path="models/emotion_model.onnx"):
        # Prefer CoreML (Metal + Neural Engine), then CPU
        providers = [
            ('CoreMLExecutionProvider', {
                'ModelFormat': 'MLProgram',
                'MLComputeUnits': 'ALL'
            }),
            ('CPUExecutionProvider', {})
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)

    def predict(self, face_img):
        # Preprocess: resize, normalize, NCHW
        img = cv2.resize(face_img, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # shape (1,3,224,224)
        outputs = self.session.run(None, {'input': img})
        return int(np.argmax(outputs[0]))