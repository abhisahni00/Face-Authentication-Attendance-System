"""
Face Recognition Module using DeepFace
Handles face detection, embedding extraction, and face matching
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from scipy.spatial.distance import cosine
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace

# Configuration
MODEL_NAME = "ArcFace"  # Best accuracy: ArcFace, Facenet512, VGG-Face
DETECTOR_BACKEND = "opencv"  # Options: opencv, retinaface, mtcnn, ssd
SIMILARITY_THRESHOLD = 0.55  # Lower = stricter matching (cosine distance)
MIN_FACE_SIZE = (80, 80)  # Minimum face size in pixels


class FaceRecognition:
    """
    Face Recognition class using DeepFace with ArcFace embeddings
    """
    
    def __init__(self, model_name: str = MODEL_NAME, detector: str = DETECTOR_BACKEND):
        """
        Initialize face recognition system
        
        Args:
            model_name: Face embedding model (ArcFace, Facenet512, VGG-Face)
            detector: Face detector backend (opencv, retinaface, mtcnn)
        """
        self.model_name = model_name
        self.detector = detector
        self.threshold = SIMILARITY_THRESHOLD
        
        # Warm up the model (first call is slow)
        print(f"🔄 Loading {model_name} model...")
        self._warmup()
        print(f"✅ Face recognition model loaded!")
    
    def _warmup(self):
        """Warm up the model with a dummy image"""
        try:
            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
            dummy[60:100, 60:100] = 255  # Add some variation
            DeepFace.represent(
                dummy, 
                model_name=self.model_name, 
                detector_backend="skip",
                enforce_detection=False
            )
        except Exception:
            pass  # Warmup may fail, that's okay
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image
        
        Args:
            image: BGR image (OpenCV format)
        
        Returns:
            List of detected faces with bounding boxes
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            faces = DeepFace.extract_faces(
                rgb_image,
                detector_backend=self.detector,
                enforce_detection=False,
                align=True
            )
            
            result = []
            for face in faces:
                # Filter out low confidence detections
                if face.get("confidence", 0) > 0.9:
                    facial_area = face.get("facial_area", {})
                    result.append({
                        "face": face.get("face"),
                        "box": (
                            facial_area.get("x", 0),
                            facial_area.get("y", 0),
                            facial_area.get("w", 0),
                            facial_area.get("h", 0)
                        ),
                        "confidence": face.get("confidence", 0)
                    })
            
            return result
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image
        
        Args:
            image: BGR image (OpenCV format)
        
        Returns:
            512D embedding vector or None if no face detected
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            embeddings = DeepFace.represent(
                rgb_image,
                model_name=self.model_name,
                detector_backend=self.detector,
                enforce_detection=False
            )
            
            if embeddings and len(embeddings) > 0:
                return np.array(embeddings[0]["embedding"])
            
            return None
            
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None
    
    def get_multiple_embeddings(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Get averaged embedding from multiple face images
        
        Args:
            images: List of BGR images
        
        Returns:
            Averaged embedding vector
        """
        embeddings = []
        
        for img in images:
            emb = self.get_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            return None
        
        # Average the embeddings for better representation
        return np.mean(embeddings, axis=0)
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[bool, float]:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
        
        Returns:
            Tuple of (is_match: bool, similarity_score: float)
        """
        # Calculate cosine distance (0 = identical, 2 = opposite)
        distance = cosine(embedding1, embedding2)
        
        # Convert to similarity (1 = identical, 0 = completely different)
        similarity = 1 - distance
        
        # Check against threshold
        is_match = distance < self.threshold
        
        return is_match, similarity
    
    def identify_face(self, 
                     embedding: np.ndarray, 
                     registered_users: List[Dict]) -> Tuple[Optional[Dict], float]:
        """
        Identify a face against registered users
        
        Args:
            embedding: Face embedding to identify
            registered_users: List of user dicts with 'embedding' key
        
        Returns:
            Tuple of (matched_user or None, confidence)
        """
        best_match = None
        best_similarity = 0
        
        for user in registered_users:
            user_embedding = user.get("embedding")
            if user_embedding is None:
                continue
            
            is_match, similarity = self.compare_faces(embedding, user_embedding)
            
            if is_match and similarity > best_similarity:
                best_match = user
                best_similarity = similarity
        
        return best_match, best_similarity
    
    def draw_face_box(self, 
                     image: np.ndarray, 
                     box: Tuple[int, int, int, int],
                     name: str = None,
                     confidence: float = None,
                     color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw face bounding box on image
        
        Args:
            image: BGR image
            box: (x, y, w, h) bounding box
            name: Optional name to display
            confidence: Optional confidence score
            color: BGR color tuple
        
        Returns:
            Image with drawn box
        """
        img = image.copy()
        x, y, w, h = box
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Draw name and confidence
        if name:
            label = name
            if confidence:
                label += f" ({confidence:.1%})"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 5, y), color, -1)
            
            # Text
            cv2.putText(img, label, (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return img


def capture_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Capture a single frame from camera"""
    ret, frame = cap.read()
    if ret:
        return frame
    return None


def init_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Initialize camera capture"""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


# Singleton instance for use across the app
_face_recognition_instance = None

def get_face_recognition() -> FaceRecognition:
    """Get or create FaceRecognition singleton"""
    global _face_recognition_instance
    if _face_recognition_instance is None:
        _face_recognition_instance = FaceRecognition()
    return _face_recognition_instance
