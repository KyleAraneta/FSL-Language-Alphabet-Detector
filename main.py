import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)

def main():
    global prev_time
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            max_attempts = 5
            attempt = 0
            while not ret and attempt < max_attempts:
                attempt += 1
                time.sleep(0.1)
                ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame after {max_attempts} attempts.")
                break
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    fingertips = {
                        "thumb": hand_landmarks.landmark[4],
                        "index": hand_landmarks.landmark[8],
                        "middle": hand_landmarks.landmark[12],
                        "ring": hand_landmarks.landmark[16],
                        "pinky": hand_landmarks.landmark[20],
                    }

                    for name, landmark in fingertips.items():
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.putText(
                            frame,
                            name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                        cv2.circle(
                            frame,
                            (x, y),
                            6,
                            (0, 255, 0),
                            -1
                        )

            cv2.imshow("Hand + Finger Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()