import os
import time
import urllib.request
from collections import deque, Counter
import warnings

import cv2
import joblib
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

CLASSIFIER_PATH = "fsl_model.joblib"
MOTION_MODEL_PATH = "fsl_motion_model.joblib"

CAMERA_INDEX = 0

SEQUENCE_LENGTH = 30
STATIC_CONFIDENCE_THRESHOLD = 0.60
MOTION_CONFIDENCE_THRESHOLD = 0.75
MOTION_MOVEMENT_THRESHOLD = 0.15

# How long J/Z stays on screen after detection
MOTION_HOLD_SECONDS = 3.0

if not os.path.exists(CLASSIFIER_PATH):
    print("No trained static model found. Run collect_data.py first, then train_model.py.")
    exit()

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

classifier = joblib.load(CLASSIFIER_PATH)

motion_classifier = None
if os.path.exists(MOTION_MODEL_PATH):
    motion_classifier = joblib.load(MOTION_MODEL_PATH)
    print("Motion model loaded. J and Z detection enabled.")
else:
    print("No motion model found. J and Z movement detection disabled.")

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
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)


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


def raw_landmarks(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def sequence_to_features(sequence):
    first_frame = sequence[0]
    wrist0 = first_frame[0]

    xs = [p[0] for p in first_frame]
    ys = [p[1] for p in first_frame]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []

    for frame in sequence:
        for x, y, z in frame:
            features.extend([
                (x - wrist0[0]) / scale,
                (y - wrist0[1]) / scale,
                (z - wrist0[2]) / scale
            ])

    return features


def calculate_movement(sequence):
    if len(sequence) < 2:
        return 0

    first_frame = sequence[0]

    xs = [p[0] for p in first_frame]
    ys = [p[1] for p in first_frame]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    important_points = [0, 4, 8, 12, 16, 20]
    max_dist = 0

    for frame in sequence:
        for point_id in important_points:
            dx = (frame[point_id][0] - first_frame[point_id][0]) / scale
            dy = (frame[point_id][1] - first_frame[point_id][1]) / scale
            dist = (dx * dx + dy * dy) ** 0.5
            max_dist = max(max_dist, dist)

    return max_dist


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

    last_motion_letter = ""
    last_motion_time = 0

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

            static_letter = ""
            static_confidence = 0.0

            motion_letter = ""
            motion_confidence = 0.0
            movement_score = 0.0

            final_letter = ""

            now = time.time()
            motion_hold_active = (
                last_motion_letter != ""
                and now - last_motion_time <= MOTION_HOLD_SECONDS
            )

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]

                motion_buffer.append(raw_landmarks(hand_landmarks))

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

                static_letter = classifier.classes_[max_index]
                static_confidence = probabilities[max_index]

                if static_confidence >= STATIC_CONFIDENCE_THRESHOLD:
                    prediction_history.append(static_letter)

                stable_letter = get_stable_prediction()
                final_letter = stable_letter

                if not motion_hold_active:
                    if motion_classifier is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                        sequence = list(motion_buffer)
                        movement_score = calculate_movement(sequence)

                        if movement_score >= MOTION_MOVEMENT_THRESHOLD:
                            motion_features = sequence_to_features(sequence)
                            motion_features = np.array(motion_features).reshape(1, -1)

                            motion_probabilities = motion_classifier.predict_proba(motion_features)[0]
                            motion_index = np.argmax(motion_probabilities)

                            motion_letter = motion_classifier.classes_[motion_index]
                            motion_confidence = motion_probabilities[motion_index]

                            if motion_letter != "NONE" and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                                last_motion_letter = motion_letter
                                last_motion_time = time.time()

                                final_letter = motion_letter

                                prediction_history.clear()
                                motion_buffer.clear()

                else:
                    final_letter = last_motion_letter

            else:
                prediction_history.clear()
                motion_buffer.clear()
                final_letter = ""

            cv2.rectangle(frame, (20, 20), (620, 190), (0, 0, 0), -1)

            cv2.putText(
                frame,
                f"Detected Letter: {final_letter}",
                (40, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Static Confidence: {static_confidence:.2f}",
                (40, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Motion: {motion_letter} {motion_confidence:.2f} | Move: {movement_score:.2f}",
                (40, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (40, 178),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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