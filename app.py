import time
import threading
from collections import deque, Counter

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, render_template_string

# Import your existing detector logic from main.py.
# Keep app.py and main.py in the same HAND-TRACKING folder.
import main as detector


app = Flask(__name__)

CAMERA_INDEX = detector.CAMERA_INDEX
SEQUENCE_LENGTH = detector.SEQUENCE_LENGTH

MODE_ALPHABET = detector.MODE_ALPHABET
MODE_NUMBERS = detector.MODE_NUMBERS
MODE_PHRASES = detector.MODE_PHRASES

STATIC_CONFIDENCE_THRESHOLD = detector.STATIC_CONFIDENCE_THRESHOLD
MOTION_CONFIDENCE_THRESHOLD = detector.MOTION_CONFIDENCE_THRESHOLD
MOTION_MOVEMENT_THRESHOLD = detector.MOTION_MOVEMENT_THRESHOLD
NUMBER_CONFIDENCE_THRESHOLD = detector.NUMBER_CONFIDENCE_THRESHOLD
PHRASE_CONFIDENCE_THRESHOLD = detector.PHRASE_CONFIDENCE_THRESHOLD
PHRASE_MOVEMENT_THRESHOLD = detector.PHRASE_MOVEMENT_THRESHOLD

PHRASE_DETECTION_COOLDOWN_SECONDS = detector.PHRASE_DETECTION_COOLDOWN_SECONDS
PHRASE_DISPLAY_HOLD_SECONDS = detector.PHRASE_DISPLAY_HOLD_SECONDS

PHRASE_READY_MESSAGE = getattr(detector, "PHRASE_READY_MESSAGE", "READY: Perform next phrase now")
PHRASE_WAIT_MESSAGE = getattr(detector, "PHRASE_WAIT_MESSAGE", "WAIT: Resetting detector...")
PHRASE_CAPTURE_MESSAGE = getattr(detector, "PHRASE_CAPTURE_MESSAGE", "CAPTURING: Keep signing...")
PHRASE_ANALYZE_MESSAGE = getattr(detector, "PHRASE_ANALYZE_MESSAGE", "ANALYZING SIGN...")

PHRASE_DISPLAY_RENAME_MAP = getattr(
    detector,
    "PHRASE_DISPLAY_RENAME_MAP",
    {
        "AYOS_LANG_ITS_OKAY": "AYOS_LANG",
        "SORRY_PASENSYA": "PASENSYA",
    }
)

MOTION_HOLD_SECONDS = detector.MOTION_HOLD_SECONDS

state_lock = threading.Lock()
latest_jpeg = None
reset_generation = 0

app_state = {
    "mode": MODE_PHRASES,
    "detected": "",
    "raw": "",
    "confidence": 0.0,
    "movement": 0.0,
    "hands": 0,
    "status": "Starting camera...",
    "is_ready": False,
    "log": ["Web detector starting..."],
}


def log_message(message):
    timestamp = time.strftime("%I:%M:%S %p")
    line = f"[{timestamp}] {message}"

    with state_lock:
        app_state["log"].append(line)

        if len(app_state["log"]) > 80:
            app_state["log"] = app_state["log"][-80:]


def update_state(**kwargs):
    with state_lock:
        app_state.update(kwargs)


def get_state_copy():
    with state_lock:
        return dict(app_state)


def display_phrase_label(label):
    label = str(label).upper().strip()
    label = PHRASE_DISPLAY_RENAME_MAP.get(label, label)
    return label.replace("_", " ")


def get_stable_prediction(history):
    if not history:
        return ""

    return Counter(history).most_common(1)[0][0]


def draw_web_overlay(frame, mode, status, detected, hands):
    cv2.rectangle(frame, (20, 20), (850, 120), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Mode: {mode}",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        status,
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Hands: {hands}",
        (690, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    if detected:
        cv2.rectangle(frame, (20, 130), (850, 205), (0, 0, 0), -1)
        cv2.putText(
            frame,
            detected,
            (40, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (0, 255, 0),
            3
        )


def reset_local_buffers(
    alphabet_history,
    number_history,
    phrase_history,
    motion_buffer
):
    alphabet_history.clear()
    number_history.clear()
    phrase_history.clear()
    motion_buffer.clear()


def detector_worker():
    global latest_jpeg, reset_generation

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        update_state(status="Camera not found. Try CAMERA_INDEX 1 or 2.", is_ready=False)
        log_message("Camera not found.")
        return

    alphabet_history = deque(maxlen=10)
    number_history = deque(maxlen=10)
    phrase_history = deque(maxlen=5)
    motion_buffer = deque(maxlen=SEQUENCE_LENGTH)

    last_motion_letter = ""
    last_motion_time = 0

    displayed_phrase_label = ""
    displayed_phrase_time = 0
    phrase_cooldown_until = 0

    frame_count = 0
    seen_reset_generation = -1

    log_message("Camera started.")
    log_message("Detector worker running.")

    with detector.HandLandmarker.create_from_options(detector.options) as landmarker:
        while True:
            current = get_state_copy()
            current_mode = current["mode"]

            with state_lock:
                current_reset_generation = reset_generation

            if current_reset_generation != seen_reset_generation:
                reset_local_buffers(
                    alphabet_history,
                    number_history,
                    phrase_history,
                    motion_buffer
                )

                last_motion_letter = ""
                last_motion_time = 0

                displayed_phrase_label = ""
                displayed_phrase_time = 0
                phrase_cooldown_until = 0

                seen_reset_generation = current_reset_generation
                log_message(f"Mode switched to {current_mode}.")

            ret, frame = cap.read()

            if not ret:
                update_state(status="Failed to read camera.", is_ready=False)
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)

            mp_image = detector.mp.Image(
                image_format=detector.mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = frame_count * 33
            frame_count += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_landmarks = None
            all_hand_landmarks = []
            detected_hands = 0

            if result.hand_landmarks:
                all_hand_landmarks = result.hand_landmarks
                detected_hands = len(all_hand_landmarks)
                hand_landmarks = all_hand_landmarks[0]

                for one_hand_landmarks in all_hand_landmarks:
                    detector.draw_hand(frame, one_hand_landmarks, w, h)

                if current_mode == MODE_ALPHABET:
                    motion_buffer.append(detector.raw_landmarks(hand_landmarks))

                elif current_mode == MODE_PHRASES:
                    motion_buffer.append(detector.get_two_hand_frame(all_hand_landmarks))

            detected_text = ""
            raw_label = ""
            confidence = 0.0
            movement_score = 0.0
            status = ""
            is_ready = False

            # ====================================================
            # ALPHABET MODE
            # ====================================================
            if current_mode == MODE_ALPHABET:
                status = "Show alphabet hand sign."

                static_letter = ""
                static_confidence = 0.0

                motion_letter = ""
                motion_confidence = 0.0

                now = time.time()
                motion_hold_active = (
                    last_motion_letter != ""
                    and now - last_motion_time <= MOTION_HOLD_SECONDS
                )

                if hand_landmarks is not None:
                    static_letter, static_confidence = detector.predict_static(
                        detector.alphabet_classifier,
                        hand_landmarks
                    )

                    if static_confidence >= STATIC_CONFIDENCE_THRESHOLD:
                        alphabet_history.append(static_letter)

                    detected_text = get_stable_prediction(alphabet_history)
                    confidence = static_confidence
                    raw_label = str(static_letter)

                    if motion_hold_active:
                        detected_text = last_motion_letter
                    else:
                        if detector.motion_classifier is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                            sequence = list(motion_buffer)
                            movement_score = detector.calculate_movement(sequence)

                            if movement_score >= MOTION_MOVEMENT_THRESHOLD:
                                motion_letter, motion_confidence = detector.predict_motion(
                                    detector.motion_classifier,
                                    sequence
                                )

                                motion_letter = str(motion_letter).upper().strip()

                                if motion_letter != "NONE" and motion_confidence >= MOTION_CONFIDENCE_THRESHOLD:
                                    last_motion_letter = motion_letter
                                    last_motion_time = time.time()
                                    detected_text = motion_letter
                                    confidence = motion_confidence
                                    raw_label = motion_letter

                                    alphabet_history.clear()
                                    motion_buffer.clear()
                else:
                    alphabet_history.clear()
                    motion_buffer.clear()
                    detected_text = ""

            # ====================================================
            # NUMBERS MODE
            # ====================================================
            elif current_mode == MODE_NUMBERS:
                status = "Show number hand sign."

                if detector.number_classifier is None:
                    status = "Number model not found."
                    detected_text = ""
                else:
                    if hand_landmarks is not None:
                        number_label, number_confidence = detector.predict_static(
                            detector.number_classifier,
                            hand_landmarks
                        )

                        if number_confidence >= NUMBER_CONFIDENCE_THRESHOLD:
                            number_history.append(number_label)

                        detected_text = get_stable_prediction(number_history)
                        raw_label = str(number_label)
                        confidence = number_confidence
                    else:
                        number_history.clear()
                        detected_text = ""

            # ====================================================
            # PHRASES MODE
            # ====================================================
            elif current_mode == MODE_PHRASES:
                now = time.time()
                status = PHRASE_READY_MESSAGE
                is_ready = True

                if detector.phrase_classifier is None:
                    status = "Phrase model not found."
                    is_ready = False
                    detected_text = ""

                elif not detector.phrase_model_compatible:
                    status = "Phrase model is incompatible. Retrain phrase model."
                    is_ready = False
                    detected_text = ""

                else:
                    if now < phrase_cooldown_until:
                        status = PHRASE_WAIT_MESSAGE
                        is_ready = False
                        motion_buffer.clear()

                    elif hand_landmarks is None:
                        status = "READY: Show your hand to start"
                        is_ready = True
                        motion_buffer.clear()
                        phrase_history.clear()

                    elif len(motion_buffer) < SEQUENCE_LENGTH:
                        status = f"{PHRASE_CAPTURE_MESSAGE} {len(motion_buffer)}/{SEQUENCE_LENGTH}"
                        is_ready = False

                    elif hand_landmarks is not None and len(motion_buffer) == SEQUENCE_LENGTH:
                        status = PHRASE_ANALYZE_MESSAGE
                        is_ready = False

                        sequence = list(motion_buffer)
                        movement_score = detector.calculate_phrase_movement(sequence)

                        if movement_score >= PHRASE_MOVEMENT_THRESHOLD:
                            raw_label, confidence = detector.predict_phrase_motion(
                                detector.phrase_classifier,
                                sequence
                            )

                            raw_label = str(raw_label).upper().strip()

                            if confidence >= PHRASE_CONFIDENCE_THRESHOLD:
                                if raw_label == "NONE":
                                    phrase_history.clear()
                                else:
                                    displayed_phrase_label = raw_label
                                    displayed_phrase_time = now
                                    phrase_cooldown_until = now + PHRASE_DETECTION_COOLDOWN_SECONDS
                                    phrase_history.clear()

                        motion_buffer.clear()

                    if displayed_phrase_label and now - displayed_phrase_time <= PHRASE_DISPLAY_HOLD_SECONDS:
                        detected_text = display_phrase_label(displayed_phrase_label)
                    else:
                        detected_text = ""

            draw_web_overlay(
                frame,
                current_mode,
                status,
                detected_text,
                detected_hands
            )

            ok, encoded = cv2.imencode(".jpg", frame)

            if ok:
                with state_lock:
                    latest_jpeg = encoded.tobytes()

            update_state(
                detected=detected_text,
                raw=raw_label,
                confidence=float(confidence),
                movement=float(movement_score),
                hands=int(detected_hands),
                status=status,
                is_ready=bool(is_ready),
            )

            time.sleep(0.005)

    cap.release()


def generate_frames():
    while True:
        with state_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sign Language Bridge</title>
    <style>
        body {
            margin: 0;
            background: #f4f3ee;
            color: #111;
            font-family: Arial, sans-serif;
        }

        header {
            background: #101014;
            color: white;
            padding: 18px 28px;
            font-size: 24px;
            font-weight: bold;
            border-bottom: 4px solid #e74c3c;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        header span {
            font-size: 14px;
            color: #aaa;
            letter-spacing: 1px;
        }

        .container {
            max-width: 1400px;
            margin: 28px auto;
            display: grid;
            grid-template-columns: 1.35fr 0.95fr;
            gap: 28px;
        }

        .card {
            background: white;
            border: 1px solid #222;
            border-radius: 6px;
            overflow: hidden;
        }

        .card-header {
            padding: 12px 18px;
            font-weight: bold;
            border-bottom: 1px solid #222;
            display: flex;
            justify-content: space-between;
        }

        .camera-feed {
            width: 100%;
            display: block;
            background: #111;
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 12px;
            padding: 14px;
            background: #fafafa;
        }

        button {
            padding: 13px 24px;
            border: 1px solid #222;
            background: white;
            cursor: pointer;
            font-weight: bold;
            border-radius: 6px;
        }

        button.active {
            background: #e74c3c;
            color: white;
            border-color: #e74c3c;
        }

        .detected-box {
            background: #101014;
            color: white;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .detected-text {
            font-size: 54px;
            font-weight: bold;
            letter-spacing: 2px;
            text-align: center;
        }

        .status {
            margin-top: 20px;
            font-size: 18px;
            color: #f1c40f;
            text-align: center;
        }

        .status.ready {
            color: #2ecc71;
        }

        .details {
            padding: 16px 20px;
            font-family: Consolas, monospace;
            font-size: 15px;
            line-height: 1.7;
        }

        .log {
            max-width: 1400px;
            margin: 0 auto 40px auto;
            background: #101014;
            color: #9cff00;
            font-family: Consolas, monospace;
            height: 150px;
            overflow-y: auto;
            padding: 14px;
            border: 1px solid #222;
            border-radius: 6px;
            white-space: pre-line;
        }
    </style>
</head>
<body>
    <header>
        <div>Sign Language Bridge</div>
        <span id="header-mode">FSL: PHRASES</span>
    </header>

    <div class="container">
        <div class="card">
            <div class="card-header">
                <div>📷 FSL Camera</div>
                <div>live detection</div>
            </div>

            <img class="camera-feed" src="/video_feed">

            <div class="controls">
                <button id="btn-ALPHABET" onclick="setMode('ALPHABET')">ALPHABET</button>
                <button id="btn-NUMBERS" onclick="setMode('NUMBERS')">NUMBERS</button>
                <button id="btn-PHRASES" onclick="setMode('PHRASES')">PHRASES</button>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div>Detected Sign</div>
                <div id="mode-label">PHRASES</div>
            </div>

            <div class="detected-box">
                <div id="detected-text" class="detected-text">---</div>
                <div id="status-text" class="status">Starting...</div>
            </div>

            <div class="details">
                <div>Raw: <span id="raw-label"></span></div>
                <div>Confidence: <span id="confidence">0.00</span></div>
                <div>Movement: <span id="movement">0.00</span></div>
                <div>Hands: <span id="hands">0</span></div>
            </div>
        </div>
    </div>

    <div class="log" id="activity-log"></div>

    <script>
        async function setMode(mode) {
            await fetch("/api/mode", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({mode})
            });

            await refreshState();
        }

        function setActiveButton(mode) {
            for (const name of ["ALPHABET", "NUMBERS", "PHRASES"]) {
                const button = document.getElementById("btn-" + name);

                if (button) {
                    button.classList.toggle("active", name === mode);
                }
            }
        }

        async function refreshState() {
            try {
                const response = await fetch("/api/state");
                const state = await response.json();

                document.getElementById("header-mode").textContent = "FSL: " + state.mode;
                document.getElementById("mode-label").textContent = state.mode;

                document.getElementById("detected-text").textContent = state.detected || "---";
                document.getElementById("status-text").textContent = state.status || "";
                document.getElementById("status-text").classList.toggle("ready", !!state.is_ready);

                document.getElementById("raw-label").textContent = state.raw || "";
                document.getElementById("confidence").textContent = Number(state.confidence || 0).toFixed(2);
                document.getElementById("movement").textContent = Number(state.movement || 0).toFixed(2);
                document.getElementById("hands").textContent = state.hands || 0;

                document.getElementById("activity-log").textContent = (state.log || []).join("\\n");

                setActiveButton(state.mode);
            } catch (error) {
                document.getElementById("status-text").textContent = "Connection lost. Retrying...";
            }
        }

        setInterval(refreshState, 200);
        refreshState();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/state")
def api_state():
    return jsonify(get_state_copy())


@app.route("/api/mode", methods=["POST"])
def api_mode():
    global reset_generation

    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "")).upper().strip()

    if mode not in [MODE_ALPHABET, MODE_NUMBERS, MODE_PHRASES]:
        return jsonify({"error": "Invalid mode"}), 400

    with state_lock:
        app_state["mode"] = mode
        app_state["detected"] = ""
        app_state["raw"] = ""
        app_state["confidence"] = 0.0
        app_state["movement"] = 0.0
        app_state["hands"] = 0
        app_state["status"] = f"Switched to {mode}"
        app_state["is_ready"] = False
        reset_generation += 1

    log_message(f"Mode changed to {mode}.")
    return jsonify({"ok": True, "mode": mode})


if __name__ == "__main__":
    print("Starting Sign Language Bridge web app...")

    worker = threading.Thread(target=detector_worker, daemon=True)
    worker.start()

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)