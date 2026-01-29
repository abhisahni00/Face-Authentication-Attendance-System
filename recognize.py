import cv2
import os
import numpy as np

FACE_SIZE = (200, 200)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def train_model():
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    print("🚀 train_model() CALLED")

    if not os.path.exists("faces"):
        print("❌ faces folder missing")
        return False

    for name in os.listdir("faces"):
        person_path = f"faces/{name}"
        if not os.path.isdir(person_path):
            continue

        print(f"📂 Loading faces for {name}")
        label_map[label_id] = name

        for img_name in os.listdir(person_path):
            img_path = f"{person_path}/{img_name}"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, FACE_SIZE)
            faces.append(img)
            labels.append(label_id)

        label_id += 1

    if len(faces) == 0:
        print("❌ NO FACES FOUND. TRAINING ABORTED.")
        return False

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(faces, np.array(labels))

    os.makedirs("model", exist_ok=True)
    model.save("model/lbph.yml")

    with open("model/labels.txt", "w") as f:
        for k, v in label_map.items():
            f.write(f"{k},{v}\n")

    print("✅ TRAINING COMPLETE:", len(faces), "faces")
    return True


def recognize(frame):
    if not os.path.exists("model/lbph.yml"):
        print("❌ Model file not found")
        return "Unknown"

    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("model/lbph.yml")

    # ✅ FIXED: label_map INSIDE function
    label_map = {}
    with open("model/labels.txt", "r") as f:
        for line in f:
            key, value = line.strip().split(",")
            label_map[int(key)] = value

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(80, 80)
    )

    # for (x, y, w, h) in faces:
    #     face = gray[y:y+h, x:x+w]
    #     face = cv2.resize(face, FACE_SIZE)

    #     label, confidence = model.predict(face)
    #     print("Prediction confidence:", confidence)

    #     # LBPH: lower confidence = better match
    #     if confidence < 120:   # relaxed for reliability
    #         return label_map[label]

    return "Unknown"
