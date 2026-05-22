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

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()