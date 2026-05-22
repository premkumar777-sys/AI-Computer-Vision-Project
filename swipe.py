import cv2
import mediapipe as mp
import pyautogui
import math

# Webcam
cap = cv2.VideoCapture(0)

# MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Click delay
click_delay = 0

# Swipe gesture variables
prev_hand_x = 0
gesture_delay = 0

while True:

    success, img = cap.read()

    # Flip image
    img = cv2.flip(img, 1)

    h, w, c = img.shape

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hand detection
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            index_finger = hand_landmarks.landmark[8]

            # Thumb tip
            thumb = hand_landmarks.landmark[4]

            # Convert coordinates
            index_x = int(index_finger.x * w)
            index_y = int(index_finger.y * h)

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            # Draw circles
            cv2.circle(img, (index_x, index_y), 10, (0, 255, 0), -1)
            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), -1)

            # =========================
            # MOVE MOUSE
            # =========================

            screen_x = int(index_finger.x * screen_width)
            screen_y = int(index_finger.y * screen_height)

            pyautogui.moveTo(screen_x, screen_y)

            # =========================
            # LEFT CLICK
            # =========================

            distance = math.hypot(
                thumb_x - index_x,
                thumb_y - index_y
            )

            cv2.putText(
                img,
                f'Distance: {int(distance)}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            if distance < 40 and click_delay == 0:

                pyautogui.click()

                cv2.putText(
                    img,
                    "LEFT CLICK",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

                click_delay = 1

            # =========================
            # SWIPE TAB CONTROL
            # =========================

            if prev_hand_x != 0:

                movement = index_x - prev_hand_x

                # Swipe Right → Next Tab
                if movement > 120 and gesture_delay == 0:

                    pyautogui.hotkey('ctrl', 'tab')

                    cv2.putText(
                        img,
                        "NEXT TAB",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3
                    )

                    gesture_delay = 1

                # Swipe Left → Previous Tab
                elif movement < -120 and gesture_delay == 0:

                    pyautogui.hotkey('ctrl', 'shift', 'tab')

                    cv2.putText(
                        img,
                        "PREVIOUS TAB",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    gesture_delay = 1

            prev_hand_x = index_x

    # =========================
    # RESET CLICK DELAY
    # =========================

    if click_delay > 0:
        click_delay += 1

    if click_delay > 10:
        click_delay = 0

    # =========================
    # RESET GESTURE DELAY
    # =========================

    if gesture_delay > 0:
        gesture_delay += 1

    if gesture_delay > 15:
        gesture_delay = 0

    # Show webcam
    cv2.imshow("Gesture Control System", img)

    # ESC to exit
    key = cv2.waitKey(1)

    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()