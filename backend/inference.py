#inference.py
import cv2
import numpy as np
import onnxruntime as ort

class EmotionClassifier:
    def __init__(self, model_path="models/emotion_model.onnx"):
        # Use only CPUExecutionProvider for maximum compatibility
        providers = [
            # ('CoreMLExecutionProvider', {
            #     'ModelFormat': 'MLProgram',
            #     'MLComputeUnits': 'ALL'
            # }),
            ('CPUExecutionProvider', {})
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)

    def predict(self, face_img):
        # Preprocess: resize to 48x48, convert to grayscale, normalize, shape (1, 48, 48, 1)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0
        img = img.reshape(1, 48, 48, 1)  # batch size 1, HWC1
        outputs = self.session.run(None, {'input': img})
        return int(np.argmax(outputs[0]))