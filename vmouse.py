import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

click_delay = 0

while True:

    success, img = cap.read()

    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            index_finger = hand_landmarks.landmark[8]

            x = int(index_finger.x * screen_width)
            y = int(index_finger.y * screen_height)

            pyautogui.moveTo(x, y)

            # Count fingers (thumb + 4 fingers)
            finger_tips = [4, 8, 12, 16, 20]
            fingers = []

            # Thumb: compare x coordinates (image is flipped)
            if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other fingers: tip.y < pip.y means finger is up
            for tip in (8, 12, 16, 20):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            fingers_up = sum(fingers)

            # If all 5 fingers are up, perform a click (debounced)
            if fingers_up == 5 and click_delay == 0:
                pyautogui.click()
                cv2.putText(img, 'CLICK', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                click_delay = 1

    cv2.imshow("Virtual Mouse", img)

    # Simple debounce for the click action
    if click_delay > 0:
        click_delay += 1

    if click_delay > 10:
        click_delay = 0

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()