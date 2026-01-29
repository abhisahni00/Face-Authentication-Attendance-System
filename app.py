from flask import Flask, render_template, Response, request
import cv2
import os
import atexit
from recognize import recognize, train_model
from attendance import mark_attendance

app = Flask(__name__)

# ---------------- VIDEO STREAM (HOME PAGE ONLY) ----------------
def gen_frames():
    cam = cv2.VideoCapture(0)   # 🔴 separate camera
    while True:
        success, frame = cam.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cam.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        save_path = f"faces/{name}"
        os.makedirs(save_path, exist_ok=True)

        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        cam = cv2.VideoCapture(0)   # 🔴 NEW camera
        count = 0

        while count < 20:
            success, frame = cam.read()
            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, 1.1, 3, minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                cv2.imwrite(f"{save_path}/{count}.jpg", face)
                count += 1

        cam.release()
        cv2.destroyAllWindows()

        trained = train_model()
        if not trained:
            return render_template(
                "register.html",
                message="❌ Training failed. Try again."
            )

        return render_template(
            "index.html",
            message=f"✅ {name} registered successfully"
        )

    return render_template("register.html")

# ---------------- LOGIN / ATTENDANCE ----------------
@app.route("/login")
def login():
    cam = cv2.VideoCapture(0)   # 🔴 NEW camera
    recognized_name = "Unknown"

    for _ in range(15):  # try multiple frames
        success, frame = cam.read()
        if not success:
            continue

        name = recognize(frame)
        if name != "Unknown":
            recognized_name = name
            break

    cam.release()
    cv2.destroyAllWindows()

    if recognized_name != "Unknown":
        mark_attendance(recognized_name)
        return render_template(
            "index.html",
            message=f"✅ Attendance marked for {recognized_name}"
        )

    return render_template(
        "index.html",
        message="❌ Face not recognized. Try again."
    )

# ---------------- CLEANUP ----------------
@atexit.register
def cleanup():
    cv2.destroyAllWindows()

# ---------------- START APP ----------------
if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(host="127.0.0.1", port=5000, debug=True)
