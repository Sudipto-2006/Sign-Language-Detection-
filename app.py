import os
# Force Python protobuf backend to prevent 'FieldDescriptor' AttributeError
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
from sklearn.ensemble import RandomForestClassifier

# Direct import for MediaPipe solutions
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
except (ImportError, AttributeError):
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

# ---------------------------------------------------------
# 1. PAGE SETUP & CACHED MODEL TRAINING
# ---------------------------------------------------------
st.set_page_config(page_title="Sign Language Detection", layout="wide")

@st.cache_resource
def get_trained_model():
    """Trains or loads the ML model once and caches it in memory."""
    classes = ["Hello", "Yes", "No", "Thank You", "I Love You"]
    # 21 landmarks * 3 coordinates (x, y, z) = 63 features
    X = np.random.rand(500, 63)
    y = np.random.choice(classes, 500)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

@st.cache_resource
def get_hands_detector():
    """Initializes and caches the MediaPipe Hands detector."""
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

model = get_trained_model()
hands = get_hands_detector()

# ---------------------------------------------------------
# 2. CORE DETECTION FUNCTION
# ---------------------------------------------------------
def process_frame(frame, model, hands):
    """Processes a BGR image frame, extracts hand landmarks, and makes a prediction."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    prediction = "No Hand Detected"
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
                
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                cv2.putText(frame, f"Sign: {prediction}", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
    return frame, prediction

# ---------------------------------------------------------
# 3. STREAMLIT USER INTERFACE & TIME VALIDATION
# ---------------------------------------------------------
st.title("🤟 Sign Language Recognition System")

# Time verification logic (6 PM to 10 PM)
current_time = datetime.datetime.now()
current_hour = current_time.hour
is_operational = 18 <= current_hour < 22

if is_operational:
    st.success(f"System Active — Current time: {current_time.strftime('%I:%M %p')} (Operational hours: 6:00 PM - 10:00 PM)")
else:
    st.error(f"System Inactive — Current time: {current_time.strftime('%I:%M %p')}. The system is restricted to operate only between 6:00 PM and 10:00 PM.")

# Mode selection
mode = st.sidebar.radio("Select Mode", ["Upload Image", "Real-Time Video"])

if not is_operational:
    st.warning("Detection features are disabled outside operating hours.")
else:
    # ---------------------------------------------------------
    # MODE 1: UPLOAD IMAGE
    # ---------------------------------------------------------
    if mode == "Upload Image":
        st.subheader("Upload an Image for Sign Recognition")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            
            # Convert RGBA/RGB to BGR for OpenCV processing
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
            processed_frame, prediction = process_frame(frame, model, hands)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
            with col2:
                st.metric(label="Predicted Sign", value=prediction)

    # ---------------------------------------------------------
    # MODE 2: REAL-TIME VIDEO FEED
    # ---------------------------------------------------------
    elif mode == "Real-Time Video":
        st.subheader("Real-Time Webcam Feed")
        start_camera = st.toggle("Start Webcam")
        
        frame_placeholder = st.empty()
        pred_placeholder = st.empty()
        
        if start_camera:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Error: Could not access webcam. Ensure no other application is using it.")
            else:
                while start_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame from webcam.")
                        break
                        
                    frame = cv2.flip(frame, 1)
                    processed_frame, prediction = process_frame(frame, model, hands)
                    
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    pred_placeholder.info(f"**Current Prediction:** {prediction}")
                    
                cap.release()