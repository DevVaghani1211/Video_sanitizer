import streamlit as st
import cv2
import tempfile
import os
import time

# --- Setup & Configuration ---
st.set_page_config(
    page_title="Video Face Sanitizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS (Dark Theme + Glassmorphism) ---
def local_css():
    st.markdown("""
    <style>
        /* Base Theme */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(-45deg, #0f172a, #1e293b, #020617);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            color: #e2e8f0;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Hide Streamlit Header & Footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        [data-testid="stStatusWidget"] {display: none;}

        /* Titles and Text */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            color: #f8fafc;
            font-weight: 700;
        }

        .main-header {
            text-align: center;
            padding-top: 1rem;
            padding-bottom: 2rem;
            animation: fadeInDown 0.8s ease-out;
        }

        .main-header h1 {
            font-size: 3.5rem;
            margin-bottom: 0px;
            background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0px 4px 10px rgba(139, 92, 246, 0.3);
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Glassmorphic Containers */
        .glass-container {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin-bottom: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-container:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        }

        /* Uploader Styling */
        [data-testid="stFileUploader"] {
            border-radius: 12px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 2px dashed rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #8b5cf6;
            background: rgba(139, 92, 246, 0.05);
        }

        /* Button Styling */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.6rem;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
            color: white;
        }

        /* Download Button Styling */
        .stDownloadButton>button {
            width: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #10b981, #059669);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.6rem;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
            transition: all 0.3s ease;
        }
        
        .stDownloadButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
            color: white;
        }

        /* Success/Info Alerts */
        [data-testid="stAlert"] {
            background-color: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            color: #e2e8f0;
            border-radius: 10px;
        }

    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Main App Logic ---

def sanitize_video(input_path, output_path, blur_intensity, scale_factor, min_neighbors):
    """Detects faces in a video and applies a blur, reporting progress."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        return False, "Error: Could not load the face detection model."

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, "Error: Could not open video file."

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'vp09') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize progress bar in the UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0

    # Ensure blur kernel is odd
    ksize = int(blur_intensity)
    if ksize % 2 == 0:
        ksize += 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=int(min_neighbors), 
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (ksize, ksize), 30)
            frame[y:y+h, x:x+w] = blurred_face

        out.write(frame)
        
        # Update progress tracking
        frame_count += 1
        if total_frames > 0:
            progress = min(int((frame_count / total_frames) * 100), 100)
            progress_bar.progress(progress / 100.0)
            status_text.markdown(f"<p style='text-align: center; color: #94a3b8;'>Processing frame {frame_count} of {total_frames}... ({progress}%)</p>", unsafe_allow_html=True)

    cap.release()
    out.release()
    
    # Cleanup progress UI
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return True, "Success"

# --- UI Layout ---

# Sidebar Settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("Fine-tune the facial recognition and blurring algorithm here:")
    
    st.markdown("---")
    blur_intensity = st.slider("Blur Intensity", min_value=15, max_value=199, value=99, step=2, help="Higher value means stronger blur on the faces.")
    
    st.markdown("---")
    scale_factor = st.slider("Detection Sensitivity (Scale)", min_value=1.01, max_value=1.5, value=1.1, step=0.01, help="Lower value detects more faces but may cause false positives.")
    min_neighbors = st.slider("Minimum Neighbors", min_value=1, max_value=10, value=5, step=1, help="Higher value means stricter detection.")
    
    st.markdown("---")
    st.markdown("### 🛡️ About")
    st.info("Privacy matters. This tool uses local AI to detect and anonymize faces in your videos completely offline. No data is sent to the cloud.")

# Main Body
st.markdown("""
<div class="main-header">
    <h1>🛡️ Auto Anonymizer</h1>
    <p style='color: #94a3b8; font-size: 1.2rem; margin-top: 10px;'>Secure, client-side video face obfuscation for ultimate privacy.</p>
</div>
""", unsafe_allow_html=True)

# Session State for tracking processing
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

st.markdown('<div class="glass-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop an MP4, MOV, or AVI file here", type=["mp4", "mov", "avi"])
st.markdown('</div>', unsafe_allow_html=True)

# Reset state if new file is uploaded
if uploaded_file is not None:
    if st.session_state.current_file != uploaded_file.name:
        st.session_state.processed_video = None
        st.session_state.current_file = uploaded_file.name

if uploaded_file is not None:
    st.markdown("### Preview & Process")
    
    # Create two columns for Before/After view
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("<p style='text-align:center; font-weight:600; color:#cbd5e1;'>Original Video</p>", unsafe_allow_html=True)
        # Using a div with simple border radius for video
        st.markdown('<div style="border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,0.1); margin-bottom:15px;">', unsafe_allow_html=True)
        st.video(uploaded_file)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action button to start
        if st.button("✨ Sanitize Video Now", use_container_width=True):
            with st.spinner("⏳ Preparing your video environment..."):
                tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile_in.write(uploaded_file.read())
                tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
                
            success, msg = sanitize_video(
                tfile_in.name, 
                tfile_out.name, 
                blur_intensity=blur_intensity,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors
            )
            
            if success:
                st.toast("Video successfully anonymized!", icon="🎉")
                st.balloons()
                # Read the output bits to session state
                with open(tfile_out.name, 'rb') as f:
                    st.session_state.processed_video = f.read()
            else:
                st.error(msg)
                
            # Cleanup temp files
            try:
                os.remove(tfile_in.name)
                os.remove(tfile_out.name)
            except OSError:
                pass

    with col2:
        st.markdown("<p style='text-align:center; font-weight:600; color:#cbd5e1;'>Anonymized Output</p>", unsafe_allow_html=True)
        
        if st.session_state.processed_video is not None:
            st.markdown('<div style="border-radius:12px; overflow:hidden; border:1px solid rgba(16,185,129,0.5); box-shadow: 0 0 15px rgba(16,185,129,0.2); margin-bottom:15px;">', unsafe_allow_html=True)
            st.video(st.session_state.processed_video)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.download_button(
                label="⬇️ Download Secure Video",
                data=st.session_state.processed_video,
                file_name="sanitized_video.webm",
                mime="video/webm",
                use_container_width=True
            )
            
            st.success("✅ Ready for download.")
            
        else:
            # Placeholder State
            st.markdown("""
            <div style='height: 100%; min-height: 250px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed rgba(255,255,255,0.1); border-radius: 12px; background: rgba(0,0,0,0.2);'>
                <p style='color: #64748b; font-size: 1.2rem; margin: 0;'>Awaiting processing...</p>
                <p style='color: #475569; font-size: 0.9rem;'>Your anonymized video will appear here.</p>
            </div>
            """, unsafe_allow_html=True)
