"""
Basic Spoof Detection Module
Implements simple anti-spoofing techniques:
1. Eye blink detection
2. Head movement detection
3. Frame difference analysis (photo vs real face)
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from collections import deque
import time


class SpoofDetector:
    """
    Basic spoof detection using multiple techniques
    """
    
    def __init__(self):
        """Initialize spoof detector with OpenCV cascades"""
        # Load Haar cascades for eye detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # Blink detection state
        self.blink_history = deque(maxlen=30)  # Last 30 frames
        self.blink_count = 0
        self.last_blink_time = 0
        
        # Movement detection state
        self.prev_frame = None
        self.movement_history = deque(maxlen=20)
        
        # Frame analysis
        self.texture_history = deque(maxlen=10)
        
    def detect_eyes(self, face_roi_gray: np.ndarray) -> List[Tuple]:
        """
        Detect eyes in a face region
        
        Args:
            face_roi_gray: Grayscale image of face region
        
        Returns:
            List of eye bounding boxes
        """
        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return list(eyes)
    
    def calculate_eye_aspect_ratio(self, eye_region: np.ndarray) -> float:
        """
        Calculate a simple eye aspect ratio based on pixel intensity
        Higher value = eye more open
        """
        if eye_region.size == 0:
            return 0.5
        
        # Simple method: analyze the center region intensity
        h, w = eye_region.shape[:2]
        center_region = eye_region[h//3:2*h//3, w//4:3*w//4]
        
        if center_region.size == 0:
            return 0.5
        
        # Calculate mean intensity (darker = eye more open/pupil visible)
        mean_intensity = np.mean(center_region)
        
        # Normalize to 0-1 range (inverted: lower intensity = higher ratio)
        return 1 - (mean_intensity / 255)
    
    def detect_blink(self, frame: np.ndarray, face_box: Tuple) -> Tuple[bool, int]:
        """
        Detect eye blink in frame
        
        Args:
            frame: BGR image
            face_box: (x, y, w, h) face bounding box
        
        Returns:
            Tuple of (blink_detected, total_blink_count)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face_box
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = self.detect_eyes(face_roi)
        
        # Track eye state
        eyes_detected = len(eyes) >= 2
        self.blink_history.append(eyes_detected)
        
        # Detect blink: eyes visible -> not visible -> visible
        blink_detected = False
        if len(self.blink_history) >= 10:
            recent = list(self.blink_history)[-10:]
            
            # Pattern: True, True, False/True(few), True, True
            # Simple: count transitions
            transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
            
            if transitions >= 2 and time.time() - self.last_blink_time > 0.3:
                # Check if current state shows eyes
                if eyes_detected:
                    blink_detected = True
                    self.blink_count += 1
                    self.last_blink_time = time.time()
        
        return blink_detected, self.blink_count
    
    def detect_movement(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect face/head movement between frames
        
        Args:
            frame: BGR image
        
        Returns:
            Tuple of (movement_detected, movement_score)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate movement score
        movement_score = np.sum(thresh) / (frame.shape[0] * frame.shape[1] * 255)
        self.movement_history.append(movement_score)
        
        self.prev_frame = gray
        
        # Detect significant movement
        movement_detected = movement_score > 0.02  # 2% of pixels changed
        
        return movement_detected, movement_score
    
    def analyze_texture(self, face_region: np.ndarray) -> Tuple[bool, float]:
        """
        Analyze face texture to detect printed photos
        Real faces have more texture variation than printed photos
        
        Args:
            face_region: BGR image of face
        
        Returns:
            Tuple of (is_real_texture, texture_score)
        """
        if face_region.size == 0:
            return False, 0.0
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (texture measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Calculate local binary pattern-like texture
        # High variance = more texture = likely real face
        
        self.texture_history.append(variance)
        
        # Threshold for real face (calibrated value)
        # Printed photos typically have lower texture variance
        is_real = variance > 100
        
        # Normalize score
        texture_score = min(variance / 500, 1.0)
        
        return is_real, texture_score
    
    def check_liveness(self, 
                       frame: np.ndarray, 
                       face_box: Tuple,
                       require_blink: bool = False,
                       require_movement: bool = True) -> Tuple[bool, dict]:
        """
        Comprehensive liveness check
        
        Args:
            frame: BGR image
            face_box: (x, y, w, h) face bounding box
            require_blink: Whether to require blink detection
            require_movement: Whether to require movement detection
        
        Returns:
            Tuple of (is_live, details_dict)
        """
        x, y, w, h = face_box
        face_region = frame[y:y+h, x:x+w]
        
        results = {
            "blink_detected": False,
            "blink_count": 0,
            "movement_detected": False,
            "movement_score": 0.0,
            "texture_real": False,
            "texture_score": 0.0,
            "is_live": False
        }
        
        # Blink detection
        blink_detected, blink_count = self.detect_blink(frame, face_box)
        results["blink_detected"] = blink_detected
        results["blink_count"] = blink_count
        
        # Movement detection
        movement_detected, movement_score = self.detect_movement(frame)
        results["movement_detected"] = movement_detected
        results["movement_score"] = movement_score
        
        # Texture analysis
        texture_real, texture_score = self.analyze_texture(face_region)
        results["texture_real"] = texture_real
        results["texture_score"] = texture_score
        
        # Determine liveness
        checks_passed = 0
        checks_required = 1  # At minimum, texture check
        
        if texture_real:
            checks_passed += 1
        
        if require_movement:
            checks_required += 1
            # Check if there's been movement in recent history
            if len(self.movement_history) > 5:
                avg_movement = np.mean(list(self.movement_history))
                if avg_movement > 0.005:  # Some movement detected
                    checks_passed += 1
        
        if require_blink:
            checks_required += 1
            if blink_count > 0:
                checks_passed += 1
        
        results["is_live"] = checks_passed >= checks_required
        
        return results["is_live"], results
    
    def reset(self):
        """Reset all detection state"""
        self.blink_history.clear()
        self.blink_count = 0
        self.last_blink_time = 0
        self.prev_frame = None
        self.movement_history.clear()
        self.texture_history.clear()


class LivenessChallenge:
    """
    Interactive liveness challenge system
    Asks user to perform specific actions
    """
    
    CHALLENGES = [
        "Please blink your eyes",
        "Please turn your head slightly left",
        "Please turn your head slightly right",
        "Please nod your head",
    ]
    
    def __init__(self):
        self.current_challenge = None
        self.challenge_start_time = None
        self.challenge_timeout = 5.0  # seconds
        self.spoof_detector = SpoofDetector()
        
    def start_challenge(self, challenge_type: str = "blink") -> str:
        """Start a liveness challenge"""
        if challenge_type == "blink":
            self.current_challenge = "blink"
            self.spoof_detector.reset()
            self.challenge_start_time = time.time()
            return "Please blink your eyes 2 times"
        elif challenge_type == "movement":
            self.current_challenge = "movement"
            self.spoof_detector.reset()
            self.challenge_start_time = time.time()
            return "Please move your head slowly"
        else:
            self.current_challenge = "blink"
            self.challenge_start_time = time.time()
            return "Please blink your eyes"
    
    def check_challenge(self, frame: np.ndarray, face_box: Tuple) -> Tuple[bool, str]:
        """
        Check if challenge is completed
        
        Returns:
            Tuple of (challenge_passed, message)
        """
        if self.current_challenge is None:
            return False, "No active challenge"
        
        # Check timeout
        elapsed = time.time() - self.challenge_start_time
        if elapsed > self.challenge_timeout:
            return False, "Challenge timed out"
        
        is_live, details = self.spoof_detector.check_liveness(
            frame, face_box,
            require_blink=(self.current_challenge == "blink"),
            require_movement=(self.current_challenge == "movement")
        )
        
        if self.current_challenge == "blink":
            if details["blink_count"] >= 2:
                return True, "✅ Liveness verified (blink detected)"
            return False, f"Blinks detected: {details['blink_count']}/2"
        
        elif self.current_challenge == "movement":
            if details["movement_score"] > 0.03:
                return True, "✅ Liveness verified (movement detected)"
            return False, f"Please move your head... ({details['movement_score']:.1%})"
        
        return False, "Checking..."
    
    def reset(self):
        """Reset challenge state"""
        self.current_challenge = None
        self.challenge_start_time = None
        self.spoof_detector.reset()


# Singleton instance
_spoof_detector = None

def get_spoof_detector() -> SpoofDetector:
    """Get or create SpoofDetector singleton"""
    global _spoof_detector
    if _spoof_detector is None:
        _spoof_detector = SpoofDetector()
    return _spoof_detector
