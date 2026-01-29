import streamlit as st
import cv2
import os
import numpy as np
from recognize import train_model, recognize
from attendance import mark_attendance

st.set_page_config(page_title="Face Attendance", layout="centered")
st.title("📸 Face Authentication Attendance System")

menu = st.sidebar.selectbox(
    "Choose Action",
    ["Register", "Login / Mark Attendance"]
)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FRAME_WINDOW = st.image([])

# ---------------- REGISTER ----------------
if menu == "Register":
    st.subheader("🆕 Register New User")
    name = st.text_input("Enter Name")

    start = st.button("Start Camera & Register")

    if start and name:
        cam = cv2.VideoCapture(0)
        count = 0
        save_path = f"faces/{name}"
        os.makedirs(save_path, exist_ok=True)

        st.info("Look at the camera...")

        while count < 25:
            ret, frame = cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                cv2.imwrite(f"{save_path}/{count}.jpg", face)
                count += 1

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cam.release()
        FRAME_WINDOW.empty()

        trained = train_model()
        if trained:
            st.success(f"✅ {name} registered & model trained")
        else:
            st.error("❌ Training failed")

# ---------------- LOGIN ----------------
if menu == "Login / Mark Attendance":
    st.subheader("🔐 Face Login")

    if st.button("Start Camera & Login"):
        cam = cv2.VideoCapture(0)
        recognized_name = "Unknown"

        for _ in range(15):
            ret, frame = cam.read()
            if not ret:
                continue

            name = recognize(frame)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if name != "Unknown":
                recognized_name = name
                break

        cam.release()
        FRAME_WINDOW.empty()

        if recognized_name != "Unknown":
            mark_attendance(recognized_name)
            st.success(f"✅ Attendance marked for {recognized_name}")
        else:
            st.error("❌ Face not recognized")

# ---------------- SHOW ATTENDANCE ----------------
if os.path.exists("attendance.csv"):
    st.subheader("📊 Attendance Records")
    st.dataframe(
        st.read_csv("attendance.csv"),
        use_container_width=True
    )
