# 👤 Face Authentication Attendance System

A modern, AI-powered face authentication system for attendance management. Built using state-of-the-art face recognition technology with anti-spoofing capabilities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![DeepFace](https://img.shields.io/badge/DeepFace-ArcFace-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## ✨ Features

- **👤 Face Registration**: Register users with multiple face images for accurate recognition
- **🔍 Face Identification**: Real-time face verification using ArcFace embeddings
- **⏰ Attendance Tracking**: Automatic punch-in/punch-out with timestamps
- **🛡️ Spoof Detection**: Basic anti-spoofing using texture analysis and movement detection
- **📊 History & Reports**: View attendance history with filtering options
- **🎨 Modern UI**: Beautiful, responsive Streamlit interface

---

## 🏗️ Architecture

```
Camera Input
     ↓
Face Detection (OpenCV/RetinaFace)
     ↓
Face Alignment
     ↓
Embedding Extraction (ArcFace 512D)
     ↓
Cosine Similarity Matching
     ↓
Spoof Detection Check
     ↓
Attendance Recording (SQLite)
```

---

## 📁 Project Structure

```
Attendace_Face_system/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation (this file)
├── src/
│   ├── __init__.py
│   ├── database.py           # SQLite database operations
│   ├── face_recognition.py   # Face detection & embedding
│   └── spoof_detection.py    # Anti-spoofing module
├── data/
│   ├── attendance.db         # SQLite database (auto-created)
│   └── faces/                # Face images storage
└── static/                   # Static assets
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- 4GB+ RAM recommended

### Installation

1. **Clone/Navigate to the project**
   ```bash
   cd ~/Desktop/Attendace_Face_system
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`

---

## 🧠 Model & Approach

### Face Embedding Model: ArcFace

We use **ArcFace** (Additive Angular Margin Loss) for face recognition:

- **Why ArcFace?**
  - State-of-the-art accuracy (~99.8% on LFW benchmark)
  - Robust to lighting variations
  - 512-dimensional embeddings
  - Pre-trained on millions of faces

- **Embedding Process**:
  1. Face detection using OpenCV/RetinaFace
  2. Face alignment and normalization
  3. Feature extraction through deep CNN
  4. 512D embedding vector output

### Face Detection

- **Primary**: OpenCV Haar Cascades (fast, reliable)
- **Alternative**: RetinaFace (more accurate, slower)
- **Fallback**: MTCNN (balanced option)

### Matching Algorithm

```python
# Cosine similarity between embeddings
similarity = 1 - cosine_distance(embedding1, embedding2)

# Match if similarity > threshold (default: 0.45)
if similarity > 0.45:
    return MATCH
```

---

## 📚 Training Process

### No Custom Training Required!

This system uses **transfer learning** with pre-trained models:

1. **ArcFace Model**: Pre-trained on MS-Celeb-1M dataset (~1M identities)
2. **Fine-tuning**: Not required - embeddings are discriminative enough
3. **User Registration**: Simply store embeddings in database

### Registration Process

1. Capture 3-5 face images per user
2. Extract embeddings from each image
3. Average embeddings for robustness
4. Store in SQLite database

```python
# Simplified registration flow
embeddings = [model.get_embedding(img) for img in captured_images]
final_embedding = np.mean(embeddings, axis=0)
database.store(user_id, final_embedding)
```

---

## 📊 Accuracy Expectations

| Condition | Expected Accuracy |
|-----------|------------------|
| Controlled lighting, frontal face | ~98-99% |
| Variable lighting | ~95-97% |
| Slight angle (±15°) | ~93-95% |
| Significant angle (±30°) | ~85-90% |
| Low light conditions | ~80-90% |

### Factors Affecting Accuracy

- ✅ Good lighting → Higher accuracy
- ✅ Frontal face → Higher accuracy
- ✅ Multiple registration images → Higher accuracy
- ⚠️ Glasses/masks → May reduce accuracy
- ⚠️ Significant pose changes → Lower accuracy

---

## ⚠️ Known Failure Cases

### 1. **Identical Twins**
- Face embeddings may be very similar
- **Mitigation**: Lower similarity threshold or add secondary verification

### 2. **Extreme Lighting**
- Very dark or overexposed images
- **Mitigation**: Implement exposure normalization

### 3. **Significant Pose Variation**
- Profile views (>45° from frontal)
- **Mitigation**: Register multiple angles during enrollment

### 4. **Occlusion**
- Masks, sunglasses, heavy makeup
- **Mitigation**: Register with/without common occlusions

### 5. **Photo/Video Attacks (Spoofing)**
- Printed photos or video replay
- **Mitigation**: Basic spoof detection implemented (texture + movement)

### 6. **Age Progression**
- Significant changes over time
- **Mitigation**: Periodic re-registration

### 7. **Camera Quality**
- Low resolution webcams
- **Mitigation**: Minimum face size requirements

---

## 🛡️ Spoof Detection

### Implemented Techniques

1. **Texture Analysis**
   - Real faces have more texture variation
   - Printed photos appear smoother
   - Uses Laplacian variance measurement

2. **Movement Detection**
   - Compares consecutive frames
   - Static images (photos) show no movement
   - Threshold: >2% pixel change

3. **Eye Blink Detection**
   - Optional liveness check
   - Detects eye state changes over time

### Limitations

- Not foolproof against high-quality video replay
- May fail with very still subjects
- Texture analysis affected by camera quality

---

## 🔧 Configuration

### Adjustable Parameters

```python
# src/face_recognition.py
MODEL_NAME = "ArcFace"           # Options: ArcFace, Facenet512, VGG-Face
DETECTOR_BACKEND = "opencv"      # Options: opencv, retinaface, mtcnn
SIMILARITY_THRESHOLD = 0.55      # Lower = stricter matching
MIN_FACE_SIZE = (80, 80)         # Minimum face dimensions

# src/spoof_detection.py
TEXTURE_THRESHOLD = 100          # Laplacian variance threshold
MOVEMENT_THRESHOLD = 0.02        # Frame difference threshold
```

---

## 🖥️ API Reference

### Database Functions

```python
# Register user
register_user(name: str, embedding: np.ndarray, email: str = None)

# Get all users with embeddings
get_all_users() -> List[dict]

# Record attendance
record_attendance(user_id: int, punch_type: str, confidence: float)

# Get attendance history
get_attendance_history(user_id: int = None, limit: int = 50)
```

### Face Recognition Functions

```python
# Initialize
fr = FaceRecognition(model_name="ArcFace", detector="opencv")

# Detect faces
faces = fr.detect_faces(image)

# Get embedding
embedding = fr.get_embedding(image)

# Identify face
user, confidence = fr.identify_face(embedding, registered_users)
```

---

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'deepface'"**
   ```bash
   pip install deepface
   ```

2. **Camera not working**
   - Check camera permissions
   - Try different camera index: `cv2.VideoCapture(1)`

3. **Slow model loading**
   - First run downloads models (~500MB)
   - Subsequent runs use cached models

4. **Low accuracy**
   - Ensure good lighting
   - Register more face images
   - Lower the similarity threshold

5. **TensorFlow warnings**
   - Set environment variable: `TF_CPP_MIN_LOG_LEVEL=3`

---

## 📈 Performance Optimization

1. **Use GPU acceleration** (if available)
   ```bash
   pip install tensorflow-gpu
   ```

2. **Reduce detection frequency** for real-time video

3. **Use OpenCV detector** for faster processing

4. **Cache embeddings** in memory for frequent users

---

## 🔮 Future Improvements

- [ ] Multi-face detection support
- [ ] Advanced liveness detection (3D depth)
- [ ] Face mask detection and handling
- [ ] Export attendance reports (CSV/PDF)
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP)

---

## 📄 License

This project is created for educational purposes as part of an AI/ML internship assignment.

---

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) - Face recognition library
- [ArcFace](https://arxiv.org/abs/1801.07698) - Face embedding model
- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenCV](https://opencv.org/) - Computer vision library

---

## 👨‍💻 Author

AI/ML Intern Assignment - Face Authentication Attendance System

---

**Built with ❤️ using Python, DeepFace, and Streamlit**
