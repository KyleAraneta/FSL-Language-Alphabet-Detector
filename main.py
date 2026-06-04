import os
import time
import urllib.request
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
CLASSIFIER_PATH = "fsl_model.joblib"

CAMERA_INDEX = 0

if not os.path.exists(CLASSIFIER_PATH):
    print("No trained model found. Run collect_data.py first, then train_model.py.")
    exit()

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

classifier = joblib.load(CLASSIFIER_PATH)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

prediction_history = deque(maxlen=10)

def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks[0]

    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []
    for lm in hand_landmarks:
        features.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])

    return features

def get_stable_prediction():
    if not prediction_history:
        return ""

    most_common = Counter(prediction_history).most_common(1)
    return most_common[0][0]

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read camera.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = frame_count * 33
            frame_count += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            predicted_letter = ""
            confidence = 0

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                points = []

                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

                features = normalize_landmarks(hand_landmarks)
                features = np.array(features).reshape(1, -1)

                probabilities = classifier.predict_proba(features)[0]
                max_index = np.argmax(probabilities)

                predicted_letter = classifier.classes_[max_index]
                confidence = probabilities[max_index]

                if confidence >= 0.60:
                    prediction_history.append(predicted_letter)

            stable_letter = get_stable_prediction()

            cv2.rectangle(frame, (20, 20), (420, 150), (0, 0, 0), -1)

            cv2.putText(
                frame,
                f"Detected Letter: {stable_letter}",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (40, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("FSL Alphabet Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()