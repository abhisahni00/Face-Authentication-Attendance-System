import cv2
import os

FACE_SIZE = (200, 200)

def register_user(name):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    save_path = f"faces/{name}"
    os.makedirs(save_path, exist_ok=True)

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("❌ Camera not opening")
        return

    print("📸 Registration started. Look at the camera...")

    count = 0
    while count < 25:   # 🔴 capture MORE images
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, FACE_SIZE)   # 🔴 force same size
            cv2.imwrite(f"{save_path}/{count}.jpg", face)
            count += 1
            print(f"✅ Image {count} saved")

        cv2.imshow("Register Face (ESC to quit)", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cam.release()
    cv2.destroyAllWindows()
    print("🎉 Registration complete!")

# ---------- RUN FROM TERMINAL ----------
if __name__ == "__main__":
    name = input("Enter your name: ").strip()
    if not name:
        print("❌ Name cannot be empty")
    else:
        register_user(name)
