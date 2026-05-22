import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

click_delay = 0

# Create fullscreen window
window_name = "Virtual Mouse"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:

    success, img = cap.read()

    img = cv2.flip(img, 1)

    h, w, c = img.shape

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

            # Move mouse
            screen_x = int(index_finger.x * screen_width)
            screen_y = int(index_finger.y * screen_height)

            pyautogui.moveTo(screen_x, screen_y)

            # Calculate distance
            distance = math.hypot(
                thumb_x - index_x,
                thumb_y - index_y
            )

            # Show distance
            cv2.putText(
                img,
                f'Distance: {int(distance)}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            # LEFT CLICK
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

    # Reset click delay
    if click_delay > 0:
        click_delay += 1

    if click_delay > 10:
        click_delay = 0

    cv2.imshow(window_name, img)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()