import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    success, img = cap.read()

    # Flip image horizontally
    img = cv2.flip(img, 1)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process pose detection
    results = pose.process(img_rgb)

    # Draw pose landmarks
    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(
                color=(0, 255, 0),
                thickness=3,
                circle_radius=3
            ),
            mp_draw.DrawingSpec(
                color=(255, 0, 255),
                thickness=3
            )
        )

    # Display title
    cv2.putText(
        img,
        "Body Pose Detection",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    # Show webcam
    cv2.imshow("Pose Detection", img)

    # Press ESC to exit
    key = cv2.waitKey(1)

    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()