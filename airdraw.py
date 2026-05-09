import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Canvas for drawing
canvas = None

# Previous coordinates
prev_x = 0
prev_y = 0

while True:

    success, img = cap.read()

    # Flip image horizontally
    img = cv2.flip(img, 1)

    # Create blank canvas
    if canvas is None:
        canvas = np.zeros_like(img)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = img.shape

            # Index finger tip
            index_tip = hand_landmarks.landmark[8]

            # Middle finger tip
            middle_tip = hand_landmarks.landmark[12]

            # Convert coordinates
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            middle_y = int(middle_tip.y * h)

            # Draw fingertip circle
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

            # ==========================
            # DRAW MODE
            # ==========================
            if y < middle_y:

                cv2.putText(
                    img,
                    "DRAW MODE",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    (255, 0, 255),
                    5
                )

                prev_x, prev_y = x, y

            # ==========================
            # ERASER MODE
            # ==========================
            else:

                cv2.putText(
                    img,
                    "ERASER MODE",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                cv2.circle(
                    canvas,
                    (x, y),
                    30,
                    (0, 0, 0),
                    -1
                )

                prev_x = 0
                prev_y = 0

    else:
        prev_x = 0
        prev_y = 0

    # Merge canvas with webcam
    img = cv2.add(img, canvas)

    # Show window
    cv2.imshow("Air Drawing with Eraser", img)

    key = cv2.waitKey(1)

    # ESC to exit
    if key == 27:
        break

    # Press C to clear canvas
    if key == ord('c'):
        canvas = np.zeros_like(img)

# Release resources
cap.release()
cv2.destroyAllWindows()