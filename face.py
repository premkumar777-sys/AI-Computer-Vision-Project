import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(img, detection)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break