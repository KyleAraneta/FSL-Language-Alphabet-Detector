import os
import csv
import time
import urllib.request
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

PHRASE_DATA_FILE = os.path.join("data", "phrases", "fsl_phrase_data.csv")

CAMERA_INDEX = 0
SEQUENCE_LENGTH = 30
MAX_HANDS = 2

COUNTDOWN_SECONDS = 3.0
PAGE_SIZE = 10

# Pause between repeated captures.
# Example: save sample -> wait 0.8s -> countdown again -> record again.
REPEAT_PAUSE_SECONDS = 0.8

PHRASE_LABELS = [
    "SALAMAT",
    "KAMUSTA",
    "MAGANDANG_UMAGA",
    "MAHAL_KITA",
    "PASENSYA",
    "OO",
    "HINDI",
    "ANG_PANGALAN_KO_AY_SI",
    "KAMUSTA_KA",
    "AYOS_LANG",
    "NAIINTINDIHAN_KO",
    "MAGANDANG_HAPON",
    "MAGANDANG_GABI",
    "MALIGAYANG_PAGDATING",
    "KAMI_AY_MGA",
    "AKO",
    "IKAW",
    "ANO",
    "KAILAN",
    "SINO",
    "BAKIT",
    "SIGE",
    "TEKA",
    "HINDI_KO_ALAM",

    # Add more phrases here later.
    # Example:
    # "PAARALAN",
    # "BAHAY",
    # "PAMILYA",
    # "KAIBIGAN",
    # "TULONG",

    "NONE"
]

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def clean_label(label):
    label = str(label).strip().upper()
    label = label.replace("?", "")
    label = label.replace("/", "_")
    label = label.replace("-", "_")
    label = "_".join(label.split())
    return label


def prepare_labels():
    labels = [clean_label(label) for label in PHRASE_LABELS]
    labels = [label for label in labels if label]

    if "NONE" not in labels:
        labels.append("NONE")

    seen = set()
    unique_labels = []

    for label in labels:
        if label not in seen:
            unique_labels.append(label)
            seen.add(label)

    return unique_labels


def create_csv_if_needed():
    os.makedirs(os.path.dirname(PHRASE_DATA_FILE), exist_ok=True)

    if not os.path.exists(PHRASE_DATA_FILE):
        header = ["label"]

        for i in range(SEQUENCE_LENGTH * MAX_HANDS * 21 * 3):
            header.append(f"f{i}")

        with open(PHRASE_DATA_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)


def load_existing_counts():
    counts = Counter()

    if not os.path.exists(PHRASE_DATA_FILE):
        return counts

    try:
        with open(PHRASE_DATA_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                label = clean_label(row.get("label", ""))

                if label:
                    counts[label] += 1
    except Exception:
        pass

    return counts


def landmarks_to_points(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def get_hand_center_x(hand_points):
    return sum(point[0] for point in hand_points) / len(hand_points)


def get_two_hand_frame(all_hand_landmarks):
    hands = []

    for hand_landmarks in all_hand_landmarks:
        hands.append(landmarks_to_points(hand_landmarks))

    # Stable left-to-right order.
    hands.sort(key=get_hand_center_x)

    while len(hands) < MAX_HANDS:
        hands.append(None)

    return hands[:MAX_HANDS]


def sequence_to_features(sequence):
    first_frame = sequence[0]

    visible_points = []

    for hand in first_frame:
        if hand is not None:
            visible_points.extend(hand)

    if not visible_points:
        return [0.0] * (SEQUENCE_LENGTH * MAX_HANDS * 21 * 3)

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


def draw_hand(frame, hand_landmarks, width, height):
    points = []

    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)


def draw_ui(
    frame,
    labels,
    saved_counts,
    current_index,
    detected_hands,
    armed_label,
    countdown_remaining,
    recording_label,
    sequence_length,
    last_index,
    jump_text,
    auto_advance,
    repeat_last_mode
):
    total_labels = len(labels)
    current_label = labels[current_index]

    # ============================================================
    # COMPACT COUNTDOWN UI
    # Hide the long instruction/phrase list while preparing.
    # ============================================================
    if armed_label is not None:
        cv2.rectangle(frame, (20, 20), (960, 180), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"GET READY: {armed_label}",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Recording starts in: {countdown_remaining:.1f}s",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Auto-repeat: {'ON' if repeat_last_mode else 'OFF'} | Press R to stop auto-repeat",
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        return

    # ============================================================
    # COMPACT RECORDING UI
    # Hide instructions and phrase list while recording.
    # ============================================================
    if recording_label is not None:
        cv2.rectangle(frame, (20, 20), (960, 180), (0, 0, 0), -1)

        cv2.putText(
            frame,
            f"RECORDING: {recording_label}",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Frames: {sequence_length}/{SEQUENCE_LENGTH}",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Auto-repeat: {'ON' if repeat_last_mode else 'OFF'} | Press R to stop after this capture",
            (40, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        return

    # ============================================================
    # NORMAL FULL UI
    # This returns after countdown/recording is done.
    # ============================================================
    cv2.rectangle(frame, (20, 20), (1210, 390), (0, 0, 0), -1)

    current_page = current_index // PAGE_SIZE
    total_pages = (total_labels + PAGE_SIZE - 1) // PAGE_SIZE

    start = current_page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total_labels)

    cv2.putText(
        frame,
        "FSL Words / Phrases Data Collection - Index Mode",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "SPACE = record once | R = auto-repeat ON/OFF | N/B = next/back | ]/[ = page | A = auto advance | ESC = quit",
        (35, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Type phrase number then ENTER to jump. Example: 57 ENTER",
        (35, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Selected: {current_index + 1}/{total_labels} - {current_label} | Count: {saved_counts[current_label]}",
        (35, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Detected hands: {detected_hands} | Page: {current_page + 1}/{total_pages} | Jump: {jump_text} | Auto advance: {'ON' if auto_advance else 'OFF'} | Auto-repeat: {'ON' if repeat_last_mode else 'OFF'}",
        (35, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        2
    )

    y = 225

    for i in range(start, end):
        label = labels[i]
        marker = ">>" if i == current_index else "  "
        text = f"{marker} {i + 1:03d}. {label} ({saved_counts[label]})"

        color = (0, 255, 255) if i == current_index else (0, 255, 0)

        cv2.putText(
            frame,
            text,
            (45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        y += 28

    if last_index is not None:
        last_label = labels[last_index]

        cv2.putText(
            frame,
            f"Last recorded/repeated: {last_index + 1:03d}. {last_label}",
            (620, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


def main():
    labels = prepare_labels()

    if len(labels) < 2:
        print("Add at least 2 phrase labels in PHRASE_LABELS.")
        return

    create_csv_if_needed()
    saved_counts = load_existing_counts()

    print("")
    print("Index-based phrase collector is active.")
    print("")
    print("Controls:")
    print("SPACE = record selected phrase once")
    print("R = auto-repeat last/selected phrase ON/OFF")
    print("N = next phrase")
    print("B = previous phrase")
    print("] = next page")
    print("[ = previous page")
    print("A = toggle auto advance")
    print("Type number + ENTER = jump to phrase number")
    print("ESC = quit")
    print("")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not found. Try changing CAMERA_INDEX to 1 or 2.")
        return

    frame_count = 0

    current_index = 0
    last_index = None

    armed_label = None
    armed_start_time = 0

    recording_label = None
    recording_index = None
    sequence = []

    jump_text = ""
    auto_advance = False
    repeat_last_mode = False

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

            detected_hands = 0
            current_two_hand_frame = None

            if result.hand_landmarks:
                detected_hands = len(result.hand_landmarks)
                current_two_hand_frame = get_two_hand_frame(result.hand_landmarks)

                for hand_landmarks in result.hand_landmarks:
                    draw_hand(frame, hand_landmarks, w, h)

            countdown_remaining = 0

            # Countdown phase
            if armed_label is not None and recording_label is None:
                elapsed = time.time() - armed_start_time
                countdown_remaining = max(COUNTDOWN_SECONDS - elapsed, 0)

                if elapsed >= COUNTDOWN_SECONDS:
                    recording_label = armed_label
                    recording_index = current_index
                    armed_label = None
                    sequence = []
                    print(f"Recording phrase: {recording_label}")

            # Recording phase
            if recording_label is not None:
                if current_two_hand_frame is not None:
                    sequence.append(current_two_hand_frame)

                if len(sequence) >= SEQUENCE_LENGTH:
                    features = sequence_to_features(sequence)

                    with open(PHRASE_DATA_FILE, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([recording_label] + features)

                    saved_counts[recording_label] += 1
                    last_index = recording_index

                    saved_label = recording_label
                    saved_index = recording_index

                    print(
                        f"Saved phrase sample: {saved_label} | "
                        f"Total: {saved_counts[saved_label]}"
                    )

                    recording_label = None
                    recording_index = None
                    sequence = []

                    if repeat_last_mode:
                        current_index = saved_index
                        armed_label = saved_label

                        # Small pause before the next countdown starts.
                        # This gives you time to reset your hands.
                        armed_start_time = time.time() + REPEAT_PAUSE_SECONDS

                        print(f"Auto-repeat ON. Next capture for: {armed_label}")

                    elif auto_advance:
                        current_index = min(current_index + 1, len(labels) - 1)

            draw_ui(
                frame=frame,
                labels=labels,
                saved_counts=saved_counts,
                current_index=current_index,
                detected_hands=detected_hands,
                armed_label=armed_label,
                countdown_remaining=countdown_remaining,
                recording_label=recording_label,
                sequence_length=len(sequence),
                last_index=last_index,
                jump_text=jump_text,
                auto_advance=auto_advance,
                repeat_last_mode=repeat_last_mode
            )

            cv2.imshow("Collect FSL Phrase Data", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 255:
                continue

            if key == 27:
                break

            pressed_key = ""

            if key not in [10, 13, 8]:
                try:
                    pressed_key = chr(key).lower()
                except ValueError:
                    pressed_key = ""

            # Allow R to stop auto-repeat even during countdown/recording.
            # If recording is already in progress, it finishes the current sample
            # but will not automatically start another one.
            if pressed_key == "r" and repeat_last_mode:
                repeat_last_mode = False
                auto_advance = False

                if armed_label is not None and recording_label is None:
                    armed_label = None

                print("Auto-repeat OFF.")
                continue

            # Do not accept navigation while counting down or recording.
            if armed_label is not None or recording_label is not None:
                continue

            # ENTER confirms numeric jump.
            if key in [10, 13]:
                if jump_text:
                    target = int(jump_text)

                    if 1 <= target <= len(labels):
                        current_index = target - 1
                        print(f"Selected phrase {target}: {labels[current_index]}")
                    else:
                        print(f"Invalid phrase number: {target}")

                    jump_text = ""

                continue

            # BACKSPACE clears typed jump number.
            if key == 8:
                jump_text = jump_text[:-1]
                continue

            if pressed_key.isdigit():
                jump_text += pressed_key

                if len(jump_text) > 3:
                    jump_text = jump_text[-3:]

            elif pressed_key == " ":
                repeat_last_mode = False
                armed_label = labels[current_index]
                armed_start_time = time.time()
                print(f"Get ready for: {armed_label}")

            elif pressed_key == "n":
                current_index = min(current_index + 1, len(labels) - 1)
                jump_text = ""

            elif pressed_key == "b":
                current_index = max(current_index - 1, 0)
                jump_text = ""

            elif pressed_key == "]":
                current_index = min(current_index + PAGE_SIZE, len(labels) - 1)
                jump_text = ""

            elif pressed_key == "[":
                current_index = max(current_index - PAGE_SIZE, 0)
                jump_text = ""

            elif pressed_key == "r":
                # Start auto-repeat.
                # If there is a last recorded phrase, repeat that.
                # Otherwise, repeat the currently selected phrase.
                if last_index is not None:
                    current_index = last_index

                repeat_last_mode = True
                auto_advance = False

                armed_label = labels[current_index]
                armed_start_time = time.time()

                print(f"Auto-repeat ON for: {armed_label}")

            elif pressed_key == "a":
                auto_advance = not auto_advance

                if auto_advance:
                    repeat_last_mode = False

                print(f"Auto advance: {'ON' if auto_advance else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()