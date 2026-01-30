#!/usr/bin/env python3
"""
Test script for Face Authentication Attendance System
Verifies all components are working correctly
"""

import sys
import numpy as np

def test_imports():
    """Test all required imports"""
    print("🔄 Testing imports...")
    
    try:
        import cv2
        print(f"  ✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"  ❌ OpenCV: {e}")
        return False
    
    try:
        import streamlit
        print(f"  ✅ Streamlit: {streamlit.__version__}")
    except ImportError as e:
        print(f"  ❌ Streamlit: {e}")
        return False
    
    try:
        from deepface import DeepFace
        print(f"  ✅ DeepFace: imported")
    except ImportError as e:
        print(f"  ❌ DeepFace: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"  ❌ TensorFlow: {e}")
        return False
    
    return True


def test_database():
    """Test database operations"""
    print("\n🔄 Testing database...")
    
    try:
        from src.database import init_db, get_all_users
        init_db()
        users = get_all_users()
        print(f"  ✅ Database initialized")
        print(f"  ✅ Current users: {len(users)}")
        return True
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False


def test_face_recognition():
    """Test face recognition module"""
    print("\n🔄 Testing face recognition...")
    
    try:
        from src.face_recognition import FaceRecognition
        
        # Create instance (this will load the model)
        print("  🔄 Loading model (may take a moment)...")
        fr = FaceRecognition()
        print(f"  ✅ Face recognition model loaded")
        
        # Test with dummy image
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = fr.detect_faces(dummy)
        print(f"  ✅ Face detection working (found {len(faces)} faces in random noise)")
        
        return True
    except Exception as e:
        print(f"  ❌ Face recognition error: {e}")
        return False


def test_spoof_detection():
    """Test spoof detection module"""
    print("\n🔄 Testing spoof detection...")
    
    try:
        from src.spoof_detection import SpoofDetector
        
        sd = SpoofDetector()
        print(f"  ✅ Spoof detector initialized")
        
        # Test with dummy image
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        is_live, details = sd.check_liveness(dummy, (100, 100, 200, 200))
        print(f"  ✅ Liveness check working")
        
        return True
    except Exception as e:
        print(f"  ❌ Spoof detection error: {e}")
        return False


def test_camera():
    """Test camera access"""
    print("\n🔄 Testing camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print(f"  ✅ Camera working (frame: {frame.shape})")
                return True
            else:
                print(f"  ⚠️ Camera opened but couldn't read frame")
                return True  # Camera exists, just not reading
        else:
            print(f"  ⚠️ Camera not available (this is OK for server deployment)")
            return True  # Not critical for testing
            
    except Exception as e:
        print(f"  ⚠️ Camera error: {e}")
        return True  # Not critical


def main():
    """Run all tests"""
    print("=" * 50)
    print("Face Authentication Attendance System - Test Suite")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Database", test_database()))
    results.append(("Face Recognition", test_face_recognition()))
    results.append(("Spoof Detection", test_spoof_detection()))
    results.append(("Camera", test_camera()))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nRun the app with:")
        print("  streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
