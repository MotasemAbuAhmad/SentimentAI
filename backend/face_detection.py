import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing       = mp.solutions.drawing_utils

class FaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        self.detector = mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):
        # Convert BGR→RGB and run detection
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.detector.process(image_rgb)

        boxes = []
        if results.detections:
            h, w, _ = frame.shape
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                boxes.append((
                    int(bbox.xmin * w),
                    int(bbox.ymin * h),
                    int(bbox.width * w),
                    int(bbox.height * h)
                ))
        return boxes