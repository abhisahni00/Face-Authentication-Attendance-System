import cv2
import time

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def is_live(camera_index=0, duration=2):
    cam = cv2.VideoCapture(camera_index)
    start = time.time()
    detected_frames = 0

    while time.time() - start < duration:
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            detected_frames += 1

    cam.release()

    # If face detected multiple times → live
    return detected_frames >= 5
