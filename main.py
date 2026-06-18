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

# ============================================================
# PATHS
# ============================================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

ALPHABET_MODEL_PATH = os.path.join("models", "alphabet", "fsl_alphabet_model.joblib")
MOTION_MODEL_PATH = os.path.join("models", "alphabet", "fsl_alphabet_motion_model.joblib")

NUMBER_MODEL_PATH = os.path.join("models", "numbers", "fsl_number_model.joblib")
PHRASE_MODEL_PATH = os.path.join("models", "phrases", "fsl_phrase_model.joblib")

# ============================================================
# CAMERA / MODEL SETTINGS
# ============================================================
CAMERA_INDEX = 0

SEQUENCE_LENGTH = 30
PHRASE_MAX_HANDS = 2

STATIC_CONFIDENCE_THRESHOLD = 0.60
MOTION_CONFIDENCE_THRESHOLD = 0.75
MOTION_MOVEMENT_THRESHOLD = 0.15

NUMBER_CONFIDENCE_THRESHOLD = 0.60

PHRASE_CONFIDENCE_THRESHOLD = 0.70
PHRASE_MOVEMENT_THRESHOLD = 0.00

PHRASE_DETECTION_COOLDOWN_SECONDS = 0.35
PHRASE_DISPLAY_HOLD_SECONDS = 2.50

# Main.py phrase detection capture duration.
# Change this to 4.0 if you want exactly 4 seconds.
PHRASE_CAPTURE_SECONDS = 3.5

PHRASE_READY_MESSAGE = "READY: Perform next phrase now"
PHRASE_WAIT_MESSAGE = "WAIT: Resetting detector..."
PHRASE_CAPTURE_MESSAGE = "CAPTURING: Keep signing..."
PHRASE_ANALYZE_MESSAGE = "ANALYZING SIGN..."

MOTION_HOLD_SECONDS = 2.0
MENU_HOLD_SECONDS = 1.2

MODE_MENU = "MENU"
MODE_ALPHABET = "ALPHABET"
MODE_NUMBERS = "NUMBERS"
MODE_PHRASES = "PHRASES"

PHRASE_EXPECTED_FEATURES = SEQUENCE_LENGTH * PHRASE_MAX_HANDS * 21 * 3

PHRASE_DISPLAY_RENAME_MAP = {
    "AYOS_LANG_ITS_OKAY": "AYOS_LANG",
    "SORRY_PASENSYA": "PASENSYA",
}

# ============================================================
# OUTPUT SENTENCE BUILDER SETTINGS
# ============================================================
AUTO_ADD_PHRASES_TO_OUTPUT = True
AUTO_ADD_ALPHABET_TO_OUTPUT = True
AUTO_ADD_NUMBERS_TO_OUTPUT = True

PHRASE_ADD_COOLDOWN_SECONDS = 1.20

MIN_LETTER_STABLE_COUNT = 6
MIN_NUMBER_STABLE_COUNT = 6

# ============================================================
# DASHBOARD UI SETTINGS
# ============================================================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

HEADER_H = 70

CAM_X = 20
CAM_Y = 90
CAM_W = 860
CAM_H = 484

SIDE_X = 900
SIDE_Y = 90
SIDE_W = 360
SIDE_H = 484

OUTPUT_X = 20
OUTPUT_Y = 595
OUTPUT_W = 1240
OUTPUT_H = 105

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_BG = (20, 20, 24)
COLOR_PANEL = (8, 10, 14)
COLOR_PANEL_2 = (28, 30, 36)
COLOR_BORDER = (70, 70, 80)
COLOR_GREEN = (0, 255, 80)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 80, 255)
COLOR_WHITE = (245, 245, 245)
COLOR_MUTED = (160, 160, 160)

# ============================================================
# DOWNLOAD MEDIAPIPE MODEL IF NEEDED
# ============================================================
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

# ============================================================
# LOAD CLASSIFIERS
# ============================================================
if not os.path.exists(ALPHABET_MODEL_PATH):
    print("No trained alphabet model found.")
    print("Run collect_alphabet_data.py first, then train_alphabet_model.py.")
    exit()

alphabet_classifier = joblib.load(ALPHABET_MODEL_PATH)
print("Alphabet model loaded.")

motion_classifier = None

if os.path.exists(MOTION_MODEL_PATH):
    motion_classifier = joblib.load(MOTION_MODEL_PATH)
    print("Alphabet motion model loaded. J and Z detection enabled.")
else:
    print("No alphabet motion model found. J and Z movement detection disabled.")

number_classifier = None

if os.path.exists(NUMBER_MODEL_PATH):
    number_classifier = joblib.load(NUMBER_MODEL_PATH)
    print("Number model loaded. Number detection enabled.")
else:
    print("No number model found. Option 2 will open but detection is disabled.")

phrase_classifier = None
phrase_model_compatible = False

if os.path.exists(PHRASE_MODEL_PATH):
    phrase_classifier = joblib.load(PHRASE_MODEL_PATH)
    print("Phrase model loaded.")

    try:
        phrase_feature_count = phrase_classifier.n_features_in_
    except Exception:
        try:
            phrase_feature_count = phrase_classifier.named_steps["scaler"].n_features_in_
        except Exception:
            phrase_feature_count = None

    if phrase_feature_count is None:
        phrase_model_compatible = True
        print("Phrase model feature count could not be checked, but it will be used.")
    elif phrase_feature_count == PHRASE_EXPECTED_FEATURES:
        phrase_model_compatible = True
        print("Phrase model is compatible with 2-hand phrase detection.")
    else:
        phrase_model_compatible = False
        print("")
        print("WARNING: Phrase model is not compatible with this 2-hand main.py.")
        print(f"Expected features: {PHRASE_EXPECTED_FEATURES}")
        print(f"Model features: {phrase_feature_count}")
        print("This usually means the phrase model was trained using the old 1-hand phrase collector.")
        print("Recollect phrase data using the 2-hand collect_phrase_data.py, then run train_phrase_model.py.")
        print("")
else:
    print("No phrase model found. Option 3 will open but detection is disabled.")

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
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
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# PREDICTION BUFFERS
# ============================================================
alphabet_history = deque(maxlen=10)
number_history = deque(maxlen=10)
phrase_history = deque(maxlen=5)

# Keep this frame-based for alphabet J/Z motion only.
motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

# ============================================================
# LANDMARK HELPERS
# ============================================================
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


def get_hand_center_x(hand_points):
    return sum(point[0] for point in hand_points) / len(hand_points)


def get_two_hand_frame(all_hand_landmarks):
    hands = []

    for hand_landmarks in all_hand_landmarks:
        hands.append(raw_landmarks(hand_landmarks))

    hands.sort(key=get_hand_center_x)

    while len(hands) < PHRASE_MAX_HANDS:
        hands.append(None)

    return hands[:PHRASE_MAX_HANDS]


def resample_phrase_sequence(sequence, target_length=SEQUENCE_LENGTH):
    """
    Main.py can now capture a phrase for 3.5 or 4 seconds.
    This function compresses/expands those captured frames back to 30 frames
    so it remains compatible with your trained phrase model.
    """
    if not sequence:
        return []

    if len(sequence) == target_length:
        return sequence

    if len(sequence) == 1:
        return [sequence[0] for _ in range(target_length)]

    indices = np.linspace(0, len(sequence) - 1, target_length)
    resampled = []

    for idx in indices:
        resampled.append(sequence[int(round(idx))])

    return resampled


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


def sequence_to_phrase_features(sequence):
    first_frame = sequence[0]

    visible_points = []

    for hand in first_frame:
        if hand is not None:
            visible_points.extend(hand)

    if not visible_points:
        return [0.0] * PHRASE_EXPECTED_FEATURES

    anchor = visible_points[0]

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []

    for frame in sequence:
        for hand in frame:
            if hand is None:
                for _ in range(21):
                    features.extend([0.0, 0.0, 0.0])
            else:
                for x, y, z in hand:
                    features.extend([
                        (x - anchor[0]) / scale,
                        (y - anchor[1]) / scale,
                        (z - anchor[2]) / scale
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


def calculate_phrase_movement(sequence):
    if len(sequence) < 2:
        return 0

    first_frame = sequence[0]

    visible_points = []

    for hand in first_frame:
        if hand is not None:
            visible_points.extend(hand)

    if not visible_points:
        return 0

    xs = [p[0] for p in visible_points]
    ys = [p[1] for p in visible_points]

    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    important_points = [0, 4, 8, 12, 16, 20]
    max_dist = 0

    for frame in sequence:
        for hand_index, hand in enumerate(frame):
            first_hand = first_frame[hand_index]

            if hand is None or first_hand is None:
                continue

            for point_id in important_points:
                dx = (hand[point_id][0] - first_hand[point_id][0]) / scale
                dy = (hand[point_id][1] - first_hand[point_id][1]) / scale
                dist = (dx * dx + dy * dy) ** 0.5
                max_dist = max(max_dist, dist)

    return max_dist

# ============================================================
# PREDICTION HELPERS
# ============================================================
def get_stable_prediction(history):
    if not history:
        return ""

    most_common = Counter(history).most_common(1)
    return most_common[0][0]


def predict_static(classifier, hand_landmarks):
    features = normalize_landmarks(hand_landmarks)
    features = np.array(features).reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    max_index = np.argmax(probabilities)

    label = classifier.classes_[max_index]
    confidence = probabilities[max_index]

    return label, confidence


def predict_motion(classifier, sequence):
    features = sequence_to_features(sequence)
    features = np.array(features).reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    max_index = np.argmax(probabilities)

    label = classifier.classes_[max_index]
    confidence = probabilities[max_index]

    return label, confidence


def predict_phrase_motion(classifier, sequence):
    features = sequence_to_phrase_features(sequence)
    features = np.array(features).reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    max_index = np.argmax(probabilities)

    label = classifier.classes_[max_index]
    confidence = probabilities[max_index]

    return label, confidence

# ============================================================
# OUTPUT SENTENCE BUILDER HELPERS
# ============================================================
def append_token(output_tokens, output_types, token, token_type="word"):
    token = str(token).upper().strip()

    if not token:
        return

    token = token.replace("_", " ")

    if token_type == "letter":
        if len(token) != 1 or not token.isalpha():
            return

        if output_tokens and output_types[-1] == "letter":
            output_tokens[-1] += token
        else:
            output_tokens.append(token)
            output_types.append("letter")

        return

    if token_type == "number":
        if len(token) != 1 or not token.isdigit():
            return

        if output_tokens and output_types[-1] == "number":
            output_tokens[-1] += token
        else:
            output_tokens.append(token)
            output_types.append("number")

        return

    output_tokens.append(token)
    output_types.append("word")


def get_output_text(output_tokens):
    return " ".join(output_tokens).strip()

# ============================================================
# MENU HELPERS
# ============================================================
def is_finger_up(hand_landmarks, tip_id, pip_id):
    return hand_landmarks[tip_id].y < hand_landmarks[pip_id].y


def detect_menu_option(hand_landmarks):
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    ring_up = is_finger_up(hand_landmarks, 16, 14)
    pinky_up = is_finger_up(hand_landmarks, 20, 18)

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "1"

    if index_up and middle_up and not ring_up and not pinky_up:
        return "2"

    if index_up and middle_up and ring_up and not pinky_up:
        return "3"

    return ""


def detect_menu_option_from_hands(all_hand_landmarks):
    for hand_landmarks in all_hand_landmarks:
        option = detect_menu_option(hand_landmarks)

        if option:
            return option

    return ""


def clear_all_buffers():
    alphabet_history.clear()
    number_history.clear()
    phrase_history.clear()
    motion_buffer.clear()

# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_hand(frame, hand_landmarks, width, height):
    points = []

    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def make_base_canvas():
    canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), COLOR_BG, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, HEADER_H), COLOR_PANEL, -1)
    cv2.line(canvas, (0, HEADER_H), (WINDOW_WIDTH, HEADER_H), COLOR_RED, 3)

    cv2.putText(
        canvas,
        "FILIPINO SIGN LANGUAGE DETECTOR",
        (25, 43),
        FONT,
        0.8,
        COLOR_GREEN,
        2
    )

    cv2.putText(
        canvas,
        "M = menu   |   C = clear sentence   |   Backspace = undo   |   Q = quit",
        (620, 43),
        FONT,
        0.48,
        COLOR_YELLOW,
        1
    )

    return canvas


def paste_camera(canvas, camera_frame):
    camera_view = cv2.resize(camera_frame, (CAM_W, CAM_H))

    cv2.rectangle(
        canvas,
        (CAM_X - 2, CAM_Y - 2),
        (CAM_X + CAM_W + 2, CAM_Y + CAM_H + 2),
        COLOR_BORDER,
        2
    )

    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = camera_view

    cv2.rectangle(
        canvas,
        (CAM_X + 12, CAM_Y + CAM_H - 38),
        (CAM_X + 112, CAM_Y + CAM_H - 12),
        COLOR_PANEL,
        -1
    )

    cv2.putText(
        canvas,
        "LIVE",
        (CAM_X + 30, CAM_Y + CAM_H - 20),
        FONT,
        0.48,
        COLOR_GREEN,
        1
    )


def draw_panel(canvas, x, y, w, h, title):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 2)

    cv2.rectangle(canvas, (x, y), (x + w, y + 42), COLOR_PANEL_2, -1)

    cv2.putText(
        canvas,
        title,
        (x + 16, y + 28),
        FONT,
        0.55,
        COLOR_WHITE,
        2
    )


def draw_wrapped_text(canvas, text, x, y, max_chars, font_scale, color, thickness, line_gap):
    text = str(text)

    if not text:
        return y

    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = word if current == "" else current + " " + word

        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    for line in lines:
        cv2.putText(canvas, line, (x, y), FONT, font_scale, color, thickness)
        y += line_gap

    return y


def draw_output_panel(canvas, output_text):
    draw_panel(canvas, OUTPUT_X, OUTPUT_Y, OUTPUT_W, OUTPUT_H, "OUTPUT SENTENCE")

    if output_text:
        display_text = output_text[-90:]
        color = COLOR_GREEN
    else:
        display_text = "Start signing to build a sentence..."
        color = COLOR_MUTED

    draw_wrapped_text(
        canvas,
        display_text,
        OUTPUT_X + 18,
        OUTPUT_Y + 72,
        65,
        0.75,
        color,
        2,
        32
    )


def draw_menu_dashboard(camera_frame, selected_option, progress, output_text):
    canvas = make_base_canvas()
    paste_camera(canvas, camera_frame)

    draw_panel(canvas, SIDE_X, SIDE_Y, SIDE_W, SIDE_H, "MODE SELECTION")

    cv2.putText(canvas, "Show 1, 2, or 3", (SIDE_X + 22, SIDE_Y + 85), FONT, 0.72, COLOR_YELLOW, 2)
    cv2.putText(canvas, "to choose a mode.", (SIDE_X + 22, SIDE_Y + 120), FONT, 0.62, COLOR_YELLOW, 2)

    options_text = [
        ("1", "FSL Alphabet"),
        ("2", "FSL Numbers 0-9"),
        ("3", "FSL Words / Phrases"),
    ]

    y = SIDE_Y + 185

    for number, label in options_text:
        color = COLOR_GREEN

        if selected_option == number:
            color = COLOR_YELLOW

        cv2.putText(canvas, f"Option {number}", (SIDE_X + 28, y), FONT, 0.60, color, 2)
        cv2.putText(canvas, label, (SIDE_X + 28, y + 32), FONT, 0.56, color, 2)

        y += 82

    if selected_option:
        bar_x = SIDE_X + 25
        bar_y = SIDE_Y + SIDE_H - 62
        bar_w = SIDE_W - 50
        bar_h = 22

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_PANEL_2, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * progress / 100), bar_y + bar_h), COLOR_GREEN, -1)

        cv2.putText(
            canvas,
            f"Selecting Option {selected_option}... {progress:.0f}%",
            (bar_x, bar_y - 14),
            FONT,
            0.48,
            COLOR_YELLOW,
            1
        )

    draw_output_panel(canvas, output_text)
    return canvas


def draw_detection_dashboard(
    camera_frame,
    mode_name,
    detected_label,
    raw_label,
    confidence,
    movement_score,
    detected_hands,
    status_text,
    status_color,
    output_text
):
    canvas = make_base_canvas()
    paste_camera(canvas, camera_frame)

    draw_panel(canvas, SIDE_X, SIDE_Y, SIDE_W, SIDE_H, mode_name)

    cv2.putText(
        canvas,
        "DETECTED",
        (SIDE_X + 22, SIDE_Y + 82),
        FONT,
        0.55,
        COLOR_MUTED,
        2
    )

    if detected_label:
        draw_wrapped_text(
            canvas,
            detected_label,
            SIDE_X + 22,
            SIDE_Y + 132,
            16,
            0.85,
            COLOR_GREEN,
            3,
            40
        )
    else:
        cv2.putText(
            canvas,
            "---",
            (SIDE_X + 22, SIDE_Y + 132),
            FONT,
            1.0,
            COLOR_MUTED,
            2
        )

    metric_y = SIDE_Y + 245

    cv2.putText(canvas, f"Raw: {raw_label}", (SIDE_X + 22, metric_y), FONT, 0.48, COLOR_WHITE, 1)
    cv2.putText(canvas, f"Confidence: {confidence:.2f}", (SIDE_X + 22, metric_y + 34), FONT, 0.48, COLOR_WHITE, 1)
    cv2.putText(canvas, f"Movement: {movement_score:.2f}", (SIDE_X + 22, metric_y + 68), FONT, 0.48, COLOR_WHITE, 1)
    cv2.putText(canvas, f"Hands: {detected_hands}", (SIDE_X + 22, metric_y + 102), FONT, 0.48, COLOR_WHITE, 1)

    cv2.rectangle(
        canvas,
        (SIDE_X + 18, SIDE_Y + SIDE_H - 112),
        (SIDE_X + SIDE_W - 18, SIDE_Y + SIDE_H - 20),
        COLOR_PANEL_2,
        -1
    )

    draw_wrapped_text(
        canvas,
        status_text,
        SIDE_X + 32,
        SIDE_Y + SIDE_H - 72,
        26,
        0.50,
        status_color,
        2,
        26
    )

    draw_output_panel(canvas, output_text)
    return canvas

# ============================================================
# MAIN APP
# ============================================================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0
    current_mode = MODE_MENU

    menu_candidate = ""
    menu_candidate_start = 0

    last_motion_letter = ""
    last_motion_time = 0

    displayed_phrase_label = ""
    displayed_phrase_time = 0
    phrase_cooldown_until = 0

    # New phrase time-based capture state.
    phrase_capture_start_time = 0
    phrase_capture_sequence = []

    output_tokens = []
    output_types = []

    last_added_phrase = ""
    last_added_phrase_time = 0

    last_added_letter = ""
    last_added_number = ""

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, camera_frame = cap.read()

            if not ret:
                print("Failed to read camera.")
                break

            camera_frame = cv2.flip(camera_frame, 1)
            h, w, _ = camera_frame.shape

            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = frame_count * 33
            frame_count += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_landmarks = None
            all_hand_landmarks = []
            detected_hands = 0
            current_phrase_frame = None

            if result.hand_landmarks:
                all_hand_landmarks = result.hand_landmarks
                detected_hands = len(all_hand_landmarks)
                hand_landmarks = all_hand_landmarks[0]

                for one_hand_landmarks in all_hand_landmarks:
                    draw_hand(camera_frame, one_hand_landmarks, w, h)

                if current_mode == MODE_ALPHABET:
                    motion_buffer.append(raw_landmarks(hand_landmarks))

                elif current_mode == MODE_PHRASES:
                    current_phrase_frame = get_two_hand_frame(all_hand_landmarks)

            output_text = get_output_text(output_tokens)
            dashboard_frame = None

            # ====================================================
            # MENU MODE
            # ====================================================
            if current_mode == MODE_MENU:
                selected_option = ""
                progress = 0

                if all_hand_landmarks:
                    detected_option = detect_menu_option_from_hands(all_hand_landmarks)

                    if detected_option:
                        if detected_option != menu_candidate:
                            menu_candidate = detected_option
                            menu_candidate_start = time.time()
                        else:
                            elapsed = time.time() - menu_candidate_start
                            progress = min((elapsed / MENU_HOLD_SECONDS) * 100, 100)

                            if elapsed >= MENU_HOLD_SECONDS:
                                if detected_option == "1":
                                    current_mode = MODE_ALPHABET
                                    print("Alphabet mode selected.")

                                elif detected_option == "2":
                                    current_mode = MODE_NUMBERS
                                    print("Numbers mode selected.")

                                elif detected_option == "3":
                                    current_mode = MODE_PHRASES
                                    print("Phrases mode selected.")

                                clear_all_buffers()
                                menu_candidate = ""
                                menu_candidate_start = 0
                                displayed_phrase_label = ""
                                displayed_phrase_time = 0
                                phrase_cooldown_until = 0
                                phrase_capture_start_time = 0
                                phrase_capture_sequence = []
                                last_added_letter = ""
                                last_added_number = ""
                    else:
                        menu_candidate = ""
                        menu_candidate_start = 0
                else:
                    menu_candidate = ""
                    menu_candidate_start = 0

                selected_option = menu_candidate
                dashboard_frame = draw_menu_dashboard(camera_frame, selected_option, progress, output_text)

            # ====================================================
            # ALPHABET MODE
            # ====================================================
            elif current_mode == MODE_ALPHABET:
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

                status_text = "Show an alphabet hand sign."
                status_color = COLOR_YELLOW

                if hand_landmarks is not None:
                    static_letter, static_confidence = predict_static(
                        alphabet_classifier,
                        hand_landmarks
                    )

                    static_letter = str(static_letter).upper().strip()

                    if static_confidence >= STATIC_CONFIDENCE_THRESHOLD:
                        alphabet_history.append(static_letter)

                    stable_letter = get_stable_prediction(alphabet_history)
                    final_letter = stable_letter

                    if AUTO_ADD_ALPHABET_TO_OUTPUT:
                        letter_count = alphabet_history.count(stable_letter)
                        can_add_letter = (
                            stable_letter
                            and letter_count >= MIN_LETTER_STABLE_COUNT
                            and stable_letter != last_added_letter
                        )

                        if can_add_letter:
                            append_token(output_tokens, output_types, stable_letter, token_type="letter")
                            last_added_letter = stable_letter
                            print(f"Added letter to output: {stable_letter}")

                    if motion_hold_active:
                        final_letter = last_motion_letter
                    else:
                        if motion_classifier is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                            sequence = list(motion_buffer)
                            movement_score = calculate_movement(sequence)

                            if movement_score >= MOTION_MOVEMENT_THRESHOLD:
                                motion_letter, motion_confidence = predict_motion(
                                    motion_classifier,
                                    sequence
                                )

                                motion_letter = str(motion_letter).upper().strip()

                                if motion_letter != "NONE" and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                                    last_motion_letter = motion_letter
                                    last_motion_time = time.time()
                                    final_letter = motion_letter

                                    if AUTO_ADD_ALPHABET_TO_OUTPUT and motion_letter != last_added_letter:
                                        append_token(output_tokens, output_types, motion_letter, token_type="letter")
                                        last_added_letter = motion_letter
                                        print(f"Added motion letter to output: {motion_letter}")

                                    alphabet_history.clear()
                                    motion_buffer.clear()

                    if final_letter:
                        status_text = "Letter captured. Remove hand before repeating same letter."
                        status_color = COLOR_GREEN
                else:
                    alphabet_history.clear()
                    motion_buffer.clear()
                    final_letter = ""
                    last_added_letter = ""
                    status_text = "READY: Show a letter."
                    status_color = COLOR_GREEN

                output_text = get_output_text(output_tokens)

                dashboard_frame = draw_detection_dashboard(
                    camera_frame=camera_frame,
                    mode_name="MODE: FSL ALPHABET",
                    detected_label=final_letter,
                    raw_label=static_letter,
                    confidence=static_confidence,
                    movement_score=movement_score,
                    detected_hands=detected_hands,
                    status_text=status_text,
                    status_color=status_color,
                    output_text=output_text
                )

            # ====================================================
            # NUMBERS MODE
            # ====================================================
            elif current_mode == MODE_NUMBERS:
                detected_number = ""
                number_confidence = 0.0

                status_text = "Show a number hand sign."
                status_color = COLOR_YELLOW

                if number_classifier is None:
                    status_text = "Number model not found."
                    status_color = COLOR_RED
                else:
                    if hand_landmarks is not None:
                        number_label, number_confidence = predict_static(
                            number_classifier,
                            hand_landmarks
                        )

                        number_label = str(number_label).upper().strip()

                        if number_confidence >= NUMBER_CONFIDENCE_THRESHOLD:
                            number_history.append(number_label)

                        detected_number = get_stable_prediction(number_history)

                        if AUTO_ADD_NUMBERS_TO_OUTPUT:
                            number_count = number_history.count(detected_number)
                            can_add_number = (
                                detected_number
                                and number_count >= MIN_NUMBER_STABLE_COUNT
                                and detected_number != last_added_number
                            )

                            if can_add_number:
                                append_token(output_tokens, output_types, detected_number, token_type="number")
                                last_added_number = detected_number
                                print(f"Added number to output: {detected_number}")

                        if detected_number:
                            status_text = "Number captured. Remove hand before repeating same number."
                            status_color = COLOR_GREEN
                    else:
                        number_history.clear()
                        last_added_number = ""
                        status_text = "READY: Show a number."
                        status_color = COLOR_GREEN

                output_text = get_output_text(output_tokens)

                dashboard_frame = draw_detection_dashboard(
                    camera_frame=camera_frame,
                    mode_name="MODE: FSL NUMBERS",
                    detected_label=detected_number,
                    raw_label=detected_number,
                    confidence=number_confidence,
                    movement_score=0.0,
                    detected_hands=detected_hands,
                    status_text=status_text,
                    status_color=status_color,
                    output_text=output_text
                )

            # ====================================================
            # PHRASES MODE
            # ====================================================
            elif current_mode == MODE_PHRASES:
                phrase_confidence = 0.0
                movement_score = 0.0
                raw_phrase_label = ""
                display_phrase = ""

                now = time.time()
                phrase_status = PHRASE_READY_MESSAGE
                phrase_status_color = COLOR_GREEN

                if phrase_classifier is None:
                    phrase_status = "Phrase model not found."
                    phrase_status_color = COLOR_RED
                    phrase_capture_start_time = 0
                    phrase_capture_sequence = []

                elif not phrase_model_compatible:
                    phrase_status = "Phrase model is old/incompatible. Retrain phrase model."
                    phrase_status_color = COLOR_RED
                    phrase_capture_start_time = 0
                    phrase_capture_sequence = []

                else:
                    if now < phrase_cooldown_until:
                        phrase_status = PHRASE_WAIT_MESSAGE
                        phrase_status_color = COLOR_YELLOW
                        motion_buffer.clear()
                        phrase_capture_start_time = 0
                        phrase_capture_sequence = []

                    elif hand_landmarks is None:
                        phrase_status = "READY: Show your hand to start."
                        phrase_status_color = COLOR_GREEN
                        motion_buffer.clear()
                        phrase_history.clear()
                        phrase_capture_start_time = 0
                        phrase_capture_sequence = []

                    else:
                        if phrase_capture_start_time == 0:
                            phrase_capture_start_time = now
                            phrase_capture_sequence = []

                        if current_phrase_frame is not None:
                            phrase_capture_sequence.append(current_phrase_frame)

                        capture_elapsed = now - phrase_capture_start_time

                        if capture_elapsed < PHRASE_CAPTURE_SECONDS:
                            phrase_status = (
                                f"{PHRASE_CAPTURE_MESSAGE} "
                                f"{capture_elapsed:.1f}s/{PHRASE_CAPTURE_SECONDS:.1f}s"
                            )
                            phrase_status_color = COLOR_YELLOW

                        else:
                            phrase_status = PHRASE_ANALYZE_MESSAGE
                            phrase_status_color = COLOR_YELLOW

                            sequence = resample_phrase_sequence(phrase_capture_sequence, SEQUENCE_LENGTH)

                            if sequence:
                                movement_score = calculate_phrase_movement(sequence)

                                if movement_score >= PHRASE_MOVEMENT_THRESHOLD:
                                    raw_phrase_label, phrase_confidence = predict_phrase_motion(
                                        phrase_classifier,
                                        sequence
                                    )

                                    raw_phrase_label = str(raw_phrase_label).upper().strip()

                                    if phrase_confidence >= PHRASE_CONFIDENCE_THRESHOLD:
                                        if raw_phrase_label == "NONE":
                                            phrase_history.clear()
                                        else:
                                            displayed_phrase_label = raw_phrase_label
                                            displayed_phrase_time = now
                                            phrase_cooldown_until = now + PHRASE_DETECTION_COOLDOWN_SECONDS
                                            phrase_history.clear()

                                            if AUTO_ADD_PHRASES_TO_OUTPUT:
                                                can_add_phrase = (
                                                    raw_phrase_label
                                                    and raw_phrase_label != "NONE"
                                                    and (
                                                        raw_phrase_label != last_added_phrase
                                                        or now - last_added_phrase_time >= PHRASE_ADD_COOLDOWN_SECONDS
                                                    )
                                                )

                                                if can_add_phrase:
                                                    display_label_for_output = PHRASE_DISPLAY_RENAME_MAP.get(
                                                        raw_phrase_label,
                                                        raw_phrase_label
                                                    )

                                                    append_token(
                                                        output_tokens,
                                                        output_types,
                                                        display_label_for_output,
                                                        token_type="word"
                                                    )

                                                    last_added_phrase = raw_phrase_label
                                                    last_added_phrase_time = now
                                                    last_added_letter = ""
                                                    last_added_number = ""
                                                    print(f"Added phrase to output: {display_label_for_output}")

                            motion_buffer.clear()
                            phrase_capture_start_time = 0
                            phrase_capture_sequence = []

                    if displayed_phrase_label and now - displayed_phrase_time <= PHRASE_DISPLAY_HOLD_SECONDS:
                        display_label = PHRASE_DISPLAY_RENAME_MAP.get(displayed_phrase_label, displayed_phrase_label)
                        display_phrase = display_label.replace("_", " ")
                    else:
                        display_phrase = ""

                output_text = get_output_text(output_tokens)

                dashboard_frame = draw_detection_dashboard(
                    camera_frame=camera_frame,
                    mode_name="MODE: FSL PHRASES",
                    detected_label=display_phrase,
                    raw_label=raw_phrase_label,
                    confidence=phrase_confidence,
                    movement_score=movement_score,
                    detected_hands=detected_hands,
                    status_text=phrase_status,
                    status_color=phrase_status_color,
                    output_text=output_text
                )

            cv2.imshow("FSL Detector Menu System", dashboard_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("m"):
                current_mode = MODE_MENU
                clear_all_buffers()
                menu_candidate = ""
                menu_candidate_start = 0

                displayed_phrase_label = ""
                displayed_phrase_time = 0
                phrase_cooldown_until = 0
                phrase_capture_start_time = 0
                phrase_capture_sequence = []

                last_added_letter = ""
                last_added_number = ""

                print("Returned to main menu.")

            if key == ord("c"):
                output_tokens.clear()
                output_types.clear()

                last_added_phrase = ""
                last_added_letter = ""
                last_added_number = ""

                print("Output sentence cleared.")

            if key == 8:
                if output_tokens:
                    removed = output_tokens.pop()
                    output_types.pop()
                    print(f"Removed from output: {removed}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
