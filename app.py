"""
Face Authentication Attendance System
Main Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
from PIL import Image
import pandas as pd

# Import custom modules
from src.database import (
    init_db, register_user, get_all_users, get_user_by_name,
    delete_user, record_attendance, get_last_punch,
    get_attendance_history, get_today_attendance
)
from src.face_recognition import FaceRecognition, get_face_recognition
from src.spoof_detection import SpoofDetector, LivenessChallenge, get_spoof_detector

# Page configuration
st.set_page_config(
    page_title="Face Auth Attendance",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6C63FF;
        --secondary: #3D3B8E;
        --accent: #FF6B6B;
        --success: #4CAF50;
        --warning: #FFC107;
        --background: #0E1117;
        --card-bg: #1A1D24;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1A1D24 50%, #0E1117 100%);
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(90deg, #6C63FF 0%, #3D3B8E 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.3);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Card styles */
    .metric-card {
        background: linear-gradient(145deg, #1A1D24, #252932);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(108, 99, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6C63FF;
        margin: 0;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-in {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
    }
    
    .status-out {
        background: linear-gradient(90deg, #FF6B6B, #ee5a5a);
        color: white;
    }
    
    /* Camera preview */
    .camera-container {
        border: 3px solid #6C63FF;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 0 30px rgba(108, 99, 255, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #6C63FF 0%, #3D3B8E 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.5);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #1A1D24;
    }
    
    /* Success/Error messages */
    .success-msg {
        background: linear-gradient(90deg, rgba(76, 175, 80, 0.2), rgba(76, 175, 80, 0.1));
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    .error-msg {
        background: linear-gradient(90deg, rgba(255, 107, 107, 0.2), rgba(255, 107, 107, 0.1));
        border-left: 4px solid #FF6B6B;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Table styles */
    .dataframe {
        background: #1A1D24 !important;
        border-radius: 10px;
    }
    
    /* Animation */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(108, 99, 255, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(108, 99, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(108, 99, 255, 0); }
    }
    
    .recording {
        animation: pulse 2s infinite;
    }
    
    /* Photo card styling */
    .photo-card {
        background: #1A1D24;
        border-radius: 10px;
        padding: 8px;
        border: 2px solid #6C63FF;
        margin-bottom: 5px;
    }
    
    .pending-photo {
        border: 3px dashed #FFC107;
        background: rgba(255, 193, 7, 0.1);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'face_recognition' not in st.session_state:
        st.session_state.face_recognition = None
    if 'spoof_detector' not in st.session_state:
        st.session_state.spoof_detector = None
    if 'captured_frames' not in st.session_state:
        st.session_state.captured_frames = []
    if 'pending_frame' not in st.session_state:
        st.session_state.pending_frame = None
    if 'pending_face_box' not in st.session_state:
        st.session_state.pending_face_box = None
    if 'registration_step' not in st.session_state:
        st.session_state.registration_step = 0
    if 'last_action' not in st.session_state:
        st.session_state.last_action = None
    if 'camera_key' not in st.session_state:
        st.session_state.camera_key = 0


@st.cache_resource
def load_models():
    """Load face recognition and spoof detection models"""
    fr = FaceRecognition()
    sd = SpoofDetector()
    return fr, sd


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>👤 Face Authentication Attendance System</h1>
        <p>Secure biometric attendance using AI-powered face recognition</p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics():
    """Render dashboard metrics"""
    users = get_all_users()
    today_records = get_today_attendance()
    
    # Count unique users present today
    present_today = len(set(r['name'] for r in today_records if r['punch_type'] == 'IN'))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(users)}</p>
            <p class="metric-label">👥 Registered Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{present_today}</p>
            <p class="metric-label">✅ Present Today</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{len(today_records)}</p>
            <p class="metric-label">📊 Today's Punches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{current_time}</p>
            <p class="metric-label">🕐 Current Time</p>
        </div>
        """, unsafe_allow_html=True)


def page_register():
    """User registration page"""
    st.markdown("## 📝 Register New User")
    st.markdown("---")
    
    # Load models
    fr, sd = load_models()
    
    # Initialize captured frames if not exists
    if 'captured_frames' not in st.session_state:
        st.session_state.captured_frames = []
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### User Details")
        name = st.text_input("Full Name", placeholder="Enter your full name", key="reg_name")
        email = st.text_input("Email (optional)", placeholder="user@example.com", key="reg_email")
        
        st.markdown("### 📸 Capture Face Images")
        
        frames_count = len(st.session_state.captured_frames)
        remaining = 5 - frames_count
        
        if frames_count >= 5:
            st.success("✅ All 5 images captured! You can now register.")
        else:
            st.info(f"📷 Capture {remaining} more image(s). Take a photo, then click **Confirm** to save it.")
        
        # Camera input with dynamic key to allow retake
        camera_image = st.camera_input(
            f"Take photo {frames_count + 1}/5", 
            key=f"register_camera_{st.session_state.camera_key}"
        )
        
        if camera_image is not None and frames_count < 5:
            # Convert to OpenCV format
            image = Image.open(camera_image)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect face
            faces = fr.detect_faces(frame)
            
            if len(faces) > 0:
                face = faces[0]
                box = face['box']
                
                # Store as pending (not yet confirmed)
                st.session_state.pending_frame = frame
                st.session_state.pending_face_box = box
                
                # Show preview with face box
                preview = fr.draw_face_box(frame, box, "Face Detected", face['confidence'])
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                
                st.markdown("---")
                st.markdown("#### 📋 Pending Photo Preview")
                st.markdown('<div class="pending-photo">', unsafe_allow_html=True)
                st.image(preview_rgb, channels="RGB", use_container_width=True)
                
                # Confirm and Retake buttons
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    if st.button("✅ Confirm & Save", use_container_width=True, type="primary"):
                        if st.session_state.pending_frame is not None:
                            st.session_state.captured_frames.append(st.session_state.pending_frame)
                            st.session_state.pending_frame = None
                            st.session_state.pending_face_box = None
                            st.session_state.camera_key += 1  # Reset camera for next photo
                            st.rerun()
                
                with btn_col2:
                    if st.button("🔄 Retake Photo", use_container_width=True):
                        st.session_state.pending_frame = None
                        st.session_state.pending_face_box = None
                        st.session_state.camera_key += 1
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ No face detected. Please ensure your face is clearly visible and try again.")
                if st.button("🔄 Try Again"):
                    st.session_state.camera_key += 1
                    st.rerun()
    
    with col2:
        st.markdown("### 📊 Registration Status")
        
        frames_count = len(st.session_state.captured_frames)
        progress = frames_count / 5
        st.progress(progress)
        st.markdown(f"**{frames_count}/5** images captured")
        
        if frames_count > 0:
            st.markdown("#### ✅ Saved Images")
            st.markdown("*Click 🗑️ to delete any image*")
            
            # Display captured images with delete buttons
            for i, frame in enumerate(st.session_state.captured_frames):
                img_col, btn_col = st.columns([3, 1])
                
                with img_col:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(rgb, caption=f"Photo {i+1}", width=150)
                
                with btn_col:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"🗑️", key=f"del_photo_{i}", help=f"Delete photo {i+1}"):
                        st.session_state.captured_frames.pop(i)
                        st.rerun()
        
        st.markdown("---")
        
        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🗑️ Clear All", use_container_width=True, disabled=frames_count == 0):
                st.session_state.captured_frames = []
                st.session_state.pending_frame = None
                st.session_state.camera_key += 1
                st.rerun()
        
        with col_btn2:
            register_disabled = frames_count < 3 or not name
            if st.button("✅ Register User", use_container_width=True, type="primary", disabled=register_disabled):
                with st.spinner("Processing face embeddings..."):
                    # Get averaged embedding from all captured frames
                    embedding = fr.get_multiple_embeddings(st.session_state.captured_frames)
                    
                    if embedding is not None:
                        success, message = register_user(name, embedding, email)
                        if success:
                            st.success(f"🎉 {message}")
                            st.session_state.captured_frames = []
                            st.session_state.pending_frame = None
                            st.session_state.camera_key += 1
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    else:
                        st.error("❌ Could not extract face embeddings. Please try again with clearer photos.")
        
        # Show requirements
        if frames_count < 3:
            st.warning(f"⚠️ Need at least 3 images (have {frames_count})")
        if not name:
            st.warning("⚠️ Please enter a name")


def page_attendance():
    """Mark attendance page"""
    st.markdown("## ⏰ Mark Attendance")
    st.markdown("---")
    
    # Load models
    fr, sd = load_models()
    
    users = get_all_users()
    
    if len(users) == 0:
        st.warning("⚠️ No registered users found. Please register first!")
        if st.button("📝 Go to Registration"):
            st.query_params["page"] = "register"
            st.rerun()
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Face Verification")
        st.info("Look at the camera to mark your attendance. The system will verify your identity.")
        
        # Spoof detection toggle
        enable_spoof = st.checkbox("🛡️ Enable Spoof Detection", value=True)
        
        # Camera input
        camera_image = st.camera_input("Scan your face", key="attendance_camera")
        
        if camera_image is not None:
            # Convert to OpenCV format
            image = Image.open(camera_image)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detect face
            faces = fr.detect_faces(frame)
            
            if len(faces) > 0:
                face = faces[0]
                box = face['box']
                
                # Spoof detection
                is_live = True
                spoof_details = {}
                
                if enable_spoof:
                    is_live, spoof_details = sd.check_liveness(
                        frame, box, 
                        require_blink=False, 
                        require_movement=False
                    )
                
                if not is_live and enable_spoof:
                    st.error("⚠️ Spoof attempt detected! Please use a real face.")
                    with col2:
                        st.markdown("### ⚠️ Spoof Detection")
                        st.json(spoof_details)
                else:
                    # Get embedding and identify
                    embedding = fr.get_embedding(frame)
                    
                    if embedding is not None:
                        matched_user, confidence = fr.identify_face(embedding, users)
                        
                        if matched_user:
                            # Draw success box
                            preview = fr.draw_face_box(
                                frame, box, 
                                matched_user['name'], 
                                confidence,
                                color=(0, 255, 0)
                            )
                            
                            with col2:
                                st.markdown("### ✅ User Identified")
                                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                                st.image(preview_rgb, channels="RGB", use_container_width=True)
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>👤 {matched_user['name']}</h3>
                                    <p>Confidence: <strong>{confidence:.1%}</strong></p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Determine punch type
                                last_punch = get_last_punch(matched_user['id'])
                                next_punch = "OUT" if last_punch == "IN" else "IN"
                                
                                st.markdown(f"**Next action:** Punch **{next_punch}**")
                                
                                if st.button(f"⏱️ Punch {next_punch}", type="primary", use_container_width=True):
                                    success, msg = record_attendance(
                                        matched_user['id'], 
                                        next_punch, 
                                        confidence
                                    )
                                    if success:
                                        st.success(f"🎉 {msg}")
                                        st.balloons()
                                    else:
                                        st.error(f"❌ {msg}")
                        else:
                            # Unknown face
                            preview = fr.draw_face_box(
                                frame, box, 
                                "Unknown", 
                                None,
                                color=(0, 0, 255)
                            )
                            
                            with col2:
                                st.markdown("### ❌ User Not Found")
                                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                                st.image(preview_rgb, channels="RGB", use_container_width=True)
                                st.error("Face not recognized. Please register first!")
                    else:
                        st.error("❌ Could not process face. Please try again.")
            else:
                st.warning("⚠️ No face detected. Please look at the camera.")


def page_history():
    """View attendance history"""
    st.markdown("## 📊 Attendance History")
    st.markdown("---")
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        filter_option = st.selectbox(
            "Filter by",
            ["Today", "All Time", "By User"]
        )
        
        user_filter = None
        if filter_option == "By User":
            users = get_all_users()
            user_names = [u['name'] for u in users]
            if user_names:
                selected_user = st.selectbox("Select User", user_names)
                user_filter = next((u['id'] for u in users if u['name'] == selected_user), None)
    
    # Get attendance data
    if filter_option == "Today":
        records = get_today_attendance()
    elif filter_option == "By User" and user_filter:
        records = get_attendance_history(user_id=user_filter)
    else:
        records = get_attendance_history()
    
    with col2:
        if records:
            # Convert to DataFrame
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['Time'] = df['timestamp'].dt.strftime('%H:%M:%S')
            df['Date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
            df['Confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}" if x else "N/A")
            
            # Style the punch type
            def style_punch(val):
                if val == "IN":
                    return "🟢 IN"
                return "🔴 OUT"
            
            df['Status'] = df['punch_type'].apply(style_punch)
            
            # Display table
            st.dataframe(
                df[['name', 'Date', 'Time', 'Status', 'Confidence']].rename(columns={
                    'name': 'Name'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Summary stats
            st.markdown("### 📈 Summary")
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                total_in = len(df[df['punch_type'] == 'IN'])
                st.metric("Total Punch INs", total_in)
            
            with col_s2:
                total_out = len(df[df['punch_type'] == 'OUT'])
                st.metric("Total Punch OUTs", total_out)
            
            with col_s3:
                unique_users = df['name'].nunique()
                st.metric("Unique Users", unique_users)
        else:
            st.info("📭 No attendance records found.")


def page_manage():
    """Manage users page"""
    st.markdown("## ⚙️ Manage Users")
    st.markdown("---")
    
    users = get_all_users()
    
    if not users:
        st.info("No registered users yet.")
        return
    
    st.markdown(f"### 👥 Registered Users ({len(users)})")
    
    for user in users:
        with st.expander(f"👤 {user['name']}", expanded=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"**Email:** {user['email'] or 'N/A'}")
                st.markdown(f"**Registered:** {user['created_at']}")
            
            with col2:
                # Get last attendance
                last_punch = get_last_punch(user['id'], today_only=False)
                st.markdown(f"**Last Punch:** {last_punch or 'Never'}")
            
            with col3:
                if st.button("🗑️ Delete", key=f"del_{user['id']}"):
                    success, msg = delete_user(user['name'])
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


def main():
    """Main application"""
    init_session_state()
    init_db()
    
    render_header()
    
    # Define pages
    pages = ["🏠 Dashboard", "📝 Register", "⏰ Attendance", "📊 History", "⚙️ Manage"]
    page_map = {"dashboard": 0, "register": 1, "attendance": 2, "history": 3, "manage": 4}
    
    # Check query params for navigation
    query_page = st.query_params.get("page", "").lower()
    default_index = page_map.get(query_page, 0)
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        page = st.radio(
            "Select Page",
            pages,
            index=default_index,
            label_visibility="collapsed"
        )
        
        # Clear query params after navigation
        if query_page:
            st.query_params.clear()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Face Auth Attendance v1.0**
        
        AI-powered attendance system using:
        - 🧠 ArcFace embeddings
        - 🔍 DeepFace detection
        - 🛡️ Basic spoof prevention
        
        Built for AI/ML Internship
        """)
    
    # Render selected page
    current_page = page
    
    if "Dashboard" in current_page:
        render_metrics()
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Today's Activity")
            records = get_today_attendance()
            if records:
                for r in records[:10]:
                    status = "🟢" if r['punch_type'] == "IN" else "🔴"
                    time_str = datetime.fromisoformat(r['timestamp']).strftime("%H:%M")
                    st.markdown(f"{status} **{r['name']}** - {r['punch_type']} at {time_str}")
            else:
                st.info("No activity today yet.")
        
        with col2:
            st.markdown("### 🚀 Quick Actions")
            if st.button("📝 Register New User", use_container_width=True, key="quick_register"):
                st.query_params["page"] = "register"
                st.rerun()
            if st.button("⏰ Mark Attendance", use_container_width=True, key="quick_attendance"):
                st.query_params["page"] = "attendance"
                st.rerun()
    
    elif "Register" in current_page:
        page_register()
    
    elif "Attendance" in current_page:
        page_attendance()
    
    elif "History" in current_page:
        page_history()
    
    elif "Manage" in current_page:
        page_manage()


if __name__ == "__main__":
    main()
