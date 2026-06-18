import os
import csv
import time
import urllib.request
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# PATHS / SETTINGS
# ============================================================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

PHRASE_DATA_FILE = os.path.join("data", "phrases", "fsl_phrase_data.csv")

CAMERA_INDEX = 0
SEQUENCE_LENGTH = 30
MAX_HANDS = 2

# Ready countdown before recording starts.
COUNTDOWN_SECONDS = 1.0

# Actual recording duration.
# The collector records for 3.5 seconds, then resamples the captured frames
# back to SEQUENCE_LENGTH so your CSV/model format stays compatible.
RECORDING_SECONDS = 3.5

REPEAT_PAUSE_SECONDS = 0.8
PAGE_SIZE = 8

# Add all your FSL words/phrases here.
# Index mode supports 100+ phrases.
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
    "KAILAN",
    "ANO",
    "SINO",
    "BAKIT",
    "SIGE",
    "TEKA",
    "HINDI_KO_ALAM",
    "MASAYA_AKONG",
    "MAKILALA_KA",
    "MGA_TAO",
    "BINGI",
    "MAHINA_ANG_PANDINIG",
    "PANDINIG",
    "ANONG_PANGALAN_MO?"
]

# ============================================================
# DASHBOARD UI SETTINGS
# ============================================================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

HEADER_H = 64

CAM_X = 20
CAM_Y = 84
CAM_W = 760
CAM_H = 456

SIDE_X = 800
SIDE_Y = 84
SIDE_W = 460
SIDE_H = 456

BOTTOM_X = 20
BOTTOM_Y = 560
BOTTOM_W = 1240
BOTTOM_H = 140

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLOR_BG = (20, 20, 24)
COLOR_PANEL = (8, 10, 14)
COLOR_PANEL_2 = (28, 30, 36)
COLOR_BORDER = (75, 75, 85)
COLOR_GREEN = (0, 255, 80)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 70, 255)
COLOR_WHITE = (245, 245, 245)
COLOR_MUTED = (155, 155, 155)
COLOR_BLUE = (255, 140, 0)

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
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

# ============================================================
# DATA HELPERS
# ============================================================
def clean_label(label):
    label = str(label).strip().upper()
    label = label.replace("?", "")
    label = label.replace("/", "_")
    label = label.replace("-", "_")
    label = "_".join(label.split())

    while "__" in label:
        label = label.replace("__", "_")

    return label.strip("_")


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


def save_sample(label, features):
    with open(PHRASE_DATA_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([label] + features)


# ============================================================
# LANDMARK HELPERS
# ============================================================
def landmarks_to_points(hand_landmarks):
    return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]


def get_hand_center_x(hand_points):
    return sum(point[0] for point in hand_points) / len(hand_points)


def get_two_hand_frame(all_hand_landmarks):
    hands = []

    for hand_landmarks in all_hand_landmarks:
        hands.append(landmarks_to_points(hand_landmarks))

    # Stable left-to-right ordering.
    hands.sort(key=get_hand_center_x)

    while len(hands) < MAX_HANDS:
        hands.append(None)

    return hands[:MAX_HANDS]


def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
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


# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_hand(frame, hand_landmarks, width, height):
    points = []

    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, COLOR_GREEN, -1)

    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], COLOR_BLUE, 2)


def make_canvas():
    canvas = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), COLOR_BG, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, HEADER_H), COLOR_PANEL, -1)
    cv2.line(canvas, (0, HEADER_H), (WINDOW_WIDTH, HEADER_H), COLOR_RED, 3)

    cv2.putText(
        canvas,
        "FSL PHRASE DATA COLLECTOR",
        (24, 40),
        FONT,
        0.82,
        COLOR_GREEN,
        2
    )

    cv2.putText(
        canvas,
        "SPACE = record once   |   R = auto-repeat   |   N/B = next/back   |   ]/[ = page   |   A = auto advance   |   ESC = quit",
        (450, 38),
        FONT,
        0.42,
        COLOR_YELLOW,
        1
    )

    return canvas


def paste_camera(canvas, frame):
    camera_view = cv2.resize(frame, (CAM_W, CAM_H))

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
        (CAM_X + 14, CAM_Y + CAM_H - 38),
        (CAM_X + 116, CAM_Y + CAM_H - 12),
        COLOR_PANEL,
        -1
    )

    cv2.putText(
        canvas,
        "LIVE",
        (CAM_X + 35, CAM_Y + CAM_H - 20),
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
        (x + 15, y + 28),
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


def draw_progress_bar(canvas, x, y, w, h, progress, color):
    progress = max(0.0, min(progress, 1.0))

    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL_2, -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * progress), y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)


def draw_phrase_list(
    canvas,
    labels,
    saved_counts,
    current_index,
    jump_text,
    auto_advance,
    repeat_mode,
    last_index
):
    draw_panel(canvas, SIDE_X, SIDE_Y, SIDE_W, SIDE_H, "PHRASE INDEX")

    total_labels = len(labels)
    current_label = labels[current_index]

    current_page = current_index // PAGE_SIZE
    total_pages = (total_labels + PAGE_SIZE - 1) // PAGE_SIZE

    start = current_page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total_labels)

    cv2.putText(
        canvas,
        f"Selected: {current_index + 1}/{total_labels}",
        (SIDE_X + 18, SIDE_Y + 72),
        FONT,
        0.52,
        COLOR_YELLOW,
        1
    )

    draw_wrapped_text(
        canvas,
        current_label.replace("_", " "),
        SIDE_X + 18,
        SIDE_Y + 105,
        24,
        0.62,
        COLOR_GREEN,
        2,
        28
    )

    cv2.putText(
        canvas,
        f"Count: {saved_counts[current_label]}",
        (SIDE_X + 18, SIDE_Y + 162),
        FONT,
        0.50,
        COLOR_WHITE,
        1
    )

    cv2.putText(
        canvas,
        f"Page: {current_page + 1}/{total_pages}  Jump: {jump_text}",
        (SIDE_X + 18, SIDE_Y + 190),
        FONT,
        0.45,
        COLOR_MUTED,
        1
    )

    cv2.putText(
        canvas,
        f"Auto advance: {'ON' if auto_advance else 'OFF'}",
        (SIDE_X + 18, SIDE_Y + 218),
        FONT,
        0.45,
        COLOR_GREEN if auto_advance else COLOR_MUTED,
        1
    )

    cv2.putText(
        canvas,
        f"Auto-repeat: {'ON' if repeat_mode else 'OFF'}",
        (SIDE_X + 230, SIDE_Y + 218),
        FONT,
        0.45,
        COLOR_GREEN if repeat_mode else COLOR_MUTED,
        1
    )

    y = SIDE_Y + 260

    for i in range(start, end):
        label = labels[i]
        marker = ">" if i == current_index else " "
        color = COLOR_YELLOW if i == current_index else COLOR_GREEN

        text = f"{marker} {i + 1:03d}. {label[:26]} ({saved_counts[label]})"

        cv2.putText(
            canvas,
            text,
            (SIDE_X + 18, y),
            FONT,
            0.45,
            color,
            1
        )

        y += 24

    if last_index is not None:
        last_label = labels[last_index]
        cv2.putText(
            canvas,
            f"Last: {last_index + 1:03d}. {last_label[:22]}",
            (SIDE_X + 18, SIDE_Y + SIDE_H - 24),
            FONT,
            0.43,
            COLOR_YELLOW,
            1
        )


def draw_bottom_panel(
    canvas,
    detected_hands,
    current_label,
    saved_counts,
    armed_label,
    countdown_remaining,
    recording_label,
    recording_elapsed,
    sequence_length,
    repeat_mode,
    repeat_delay_remaining,
    message
):
    draw_panel(canvas, BOTTOM_X, BOTTOM_Y, BOTTOM_W, BOTTOM_H, "COLLECTION STATUS")

    left_x = BOTTOM_X + 18
    mid_x = BOTTOM_X + 470
    right_x = BOTTOM_X + 900
    top_y = BOTTOM_Y + 62

    cv2.putText(
        canvas,
        f"Detected hands: {detected_hands}",
        (left_x, top_y),
        FONT,
        0.55,
        COLOR_WHITE,
        1
    )

    cv2.putText(
        canvas,
        f"Selected count: {saved_counts[current_label]}",
        (left_x, top_y + 32),
        FONT,
        0.55,
        COLOR_WHITE,
        1
    )

    if recording_label is not None:
        status = f"RECORDING: {recording_label.replace('_', ' ')}"
        color = COLOR_RED
        progress = recording_elapsed / RECORDING_SECONDS
        remaining = max(RECORDING_SECONDS - recording_elapsed, 0)
        progress_label = f"Recording: {recording_elapsed:.1f}s / {RECORDING_SECONDS:.1f}s | Remaining: {remaining:.1f}s | Frames: {sequence_length}"

    elif armed_label is not None:
        status = f"GET READY: {armed_label.replace('_', ' ')}"
        color = COLOR_YELLOW

        if repeat_delay_remaining > 0:
            progress = 0.0
            progress_label = f"Next repeat starts in: {repeat_delay_remaining:.1f}s"
        else:
            progress = 1.0 - (countdown_remaining / COUNTDOWN_SECONDS)
            progress_label = f"Recording starts in: {countdown_remaining:.1f}s"

    else:
        status = "READY TO COLLECT"
        color = COLOR_GREEN
        progress = 0.0
        progress_label = "Press SPACE once or R for auto-repeat."

    draw_wrapped_text(
        canvas,
        status,
        mid_x,
        top_y,
        27,
        0.62,
        color,
        2,
        27
    )

    draw_progress_bar(
        canvas,
        mid_x,
        top_y + 52,
        350,
        20,
        progress,
        color
    )

    cv2.putText(
        canvas,
        progress_label,
        (mid_x, top_y + 95),
        FONT,
        0.45,
        COLOR_WHITE,
        1
    )

    cv2.putText(
        canvas,
        f"Repeat: {'ON' if repeat_mode else 'OFF'}",
        (right_x, top_y),
        FONT,
        0.55,
        COLOR_GREEN if repeat_mode else COLOR_MUTED,
        1
    )

    draw_wrapped_text(
        canvas,
        message,
        right_x,
        top_y + 34,
        28,
        0.45,
        COLOR_YELLOW,
        1,
        22
    )


def draw_dashboard(
    frame,
    labels,
    saved_counts,
    current_index,
    detected_hands,
    armed_label,
    countdown_remaining,
    recording_label,
    recording_elapsed,
    sequence_length,
    last_index,
    jump_text,
    auto_advance,
    repeat_mode,
    repeat_delay_remaining,
    message
):
    canvas = make_canvas()
    paste_camera(canvas, frame)

    draw_phrase_list(
        canvas=canvas,
        labels=labels,
        saved_counts=saved_counts,
        current_index=current_index,
        jump_text=jump_text,
        auto_advance=auto_advance,
        repeat_mode=repeat_mode,
        last_index=last_index
    )

    draw_bottom_panel(
        canvas=canvas,
        detected_hands=detected_hands,
        current_label=labels[current_index],
        saved_counts=saved_counts,
        armed_label=armed_label,
        countdown_remaining=countdown_remaining,
        recording_label=recording_label,
        recording_elapsed=recording_elapsed,
        sequence_length=sequence_length,
        repeat_mode=repeat_mode,
        repeat_delay_remaining=repeat_delay_remaining,
        message=message
    )

    return canvas


# ============================================================
# MAIN APP
# ============================================================
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
    print("R = auto-repeat selected/last phrase ON/OFF")
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
    recording_start_time = 0
    sequence = []

    jump_text = ""
    auto_advance = False
    repeat_mode = False

    message = "Select a phrase, then press SPACE to collect once or R for auto-repeat."

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
            repeat_delay_remaining = 0
            recording_elapsed = 0

            # Countdown phase
            if armed_label is not None and recording_label is None:
                elapsed = time.time() - armed_start_time

                if elapsed < 0:
                    repeat_delay_remaining = abs(elapsed)
                    countdown_remaining = COUNTDOWN_SECONDS
                else:
                    countdown_remaining = max(COUNTDOWN_SECONDS - elapsed, 0)

                    if elapsed >= COUNTDOWN_SECONDS:
                        recording_label = armed_label
                        recording_index = current_index
                        recording_start_time = time.time()
                        armed_label = None
                        sequence = []
                        message = f"Recording phrase: {recording_label}"
                        print(f"Recording phrase: {recording_label}")

            # Recording phase
            if recording_label is not None:
                recording_elapsed = time.time() - recording_start_time

                if current_two_hand_frame is not None:
                    sequence.append(current_two_hand_frame)

                if recording_elapsed >= RECORDING_SECONDS:
                    if not sequence:
                        print(f"No hand frames captured for: {recording_label}. Sample not saved.")
                        message = f"No hand frames captured for {recording_label}. Try again."

                        recording_label = None
                        recording_index = None
                        recording_start_time = 0
                        sequence = []

                    else:
                        fixed_sequence = resample_sequence(sequence, SEQUENCE_LENGTH)
                        features = sequence_to_features(fixed_sequence)
                        save_sample(recording_label, features)

                        saved_counts[recording_label] += 1
                        last_index = recording_index

                        saved_label = recording_label
                        saved_index = recording_index

                        print(
                            f"Saved phrase sample: {saved_label} | "
                            f"Total: {saved_counts[saved_label]} | "
                            f"Captured frames: {len(sequence)}"
                        )

                        message = f"Saved {saved_label}. Total: {saved_counts[saved_label]} | Captured frames: {len(sequence)}"

                        recording_label = None
                        recording_index = None
                        recording_start_time = 0
                        sequence = []

                        if repeat_mode:
                            current_index = saved_index
                            armed_label = saved_label
                            armed_start_time = time.time() + REPEAT_PAUSE_SECONDS
                            message = f"Auto-repeat ON. Next capture for {armed_label}."
                            print(f"Auto-repeat ON. Next capture for: {armed_label}")

                        elif auto_advance:
                            current_index = min(current_index + 1, len(labels) - 1)

            dashboard = draw_dashboard(
                frame=frame,
                labels=labels,
                saved_counts=saved_counts,
                current_index=current_index,
                detected_hands=detected_hands,
                armed_label=armed_label,
                countdown_remaining=countdown_remaining,
                recording_label=recording_label,
                recording_elapsed=recording_elapsed,
                sequence_length=len(sequence),
                last_index=last_index,
                jump_text=jump_text,
                auto_advance=auto_advance,
                repeat_mode=repeat_mode,
                repeat_delay_remaining=repeat_delay_remaining,
                message=message
            )

            cv2.imshow("Collect FSL Phrase Data", dashboard)

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
            if pressed_key == "r" and repeat_mode:
                repeat_mode = False
                auto_advance = False

                if armed_label is not None and recording_label is None:
                    armed_label = None

                message = "Auto-repeat OFF."
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
                        message = f"Selected phrase {target}: {labels[current_index]}"
                        print(message)
                    else:
                        message = f"Invalid phrase number: {target}"
                        print(message)

                    jump_text = ""

                continue

            # BACKSPACE clears typed jump number.
            if key == 8:
                jump_text = jump_text[:-1]
                message = f"Jump input: {jump_text}"
                continue

            if pressed_key.isdigit():
                jump_text += pressed_key

                if len(jump_text) > 3:
                    jump_text = jump_text[-3:]

                message = f"Jump input: {jump_text}. Press ENTER."

            elif pressed_key == " ":
                repeat_mode = False
                armed_label = labels[current_index]
                armed_start_time = time.time()
                message = f"Get ready for: {armed_label}"
                print(f"Get ready for: {armed_label}")

            elif pressed_key == "n":
                current_index = min(current_index + 1, len(labels) - 1)
                jump_text = ""
                message = f"Selected: {labels[current_index]}"

            elif pressed_key == "b":
                current_index = max(current_index - 1, 0)
                jump_text = ""
                message = f"Selected: {labels[current_index]}"

            elif pressed_key == "]":
                current_index = min(current_index + PAGE_SIZE, len(labels) - 1)
                jump_text = ""
                message = f"Page moved. Selected: {labels[current_index]}"

            elif pressed_key == "[":
                current_index = max(current_index - PAGE_SIZE, 0)
                jump_text = ""
                message = f"Page moved. Selected: {labels[current_index]}"

            elif pressed_key == "r":
                if last_index is not None:
                    current_index = last_index

                repeat_mode = True
                auto_advance = False

                armed_label = labels[current_index]
                armed_start_time = time.time()

                message = f"Auto-repeat ON for: {armed_label}"
                print(message)

            elif pressed_key == "a":
                auto_advance = not auto_advance

                if auto_advance:
                    repeat_mode = False

                message = f"Auto advance: {'ON' if auto_advance else 'OFF'}"
                print(message)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
