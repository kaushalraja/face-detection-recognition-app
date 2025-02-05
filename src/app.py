import streamlit as st
import sqlite3
import face_recognition
import pickle
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from PIL import Image
import onnxruntime  # Add this import

# Initialize the ArcFace model for better accuracy
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# SQLite Database to store known faces
def create_db():
    conn = sqlite3.connect('known_faces.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS known_faces (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        face_encoding BLOB
                     )''')
    conn.commit()
    conn.close()

def add_known_face(name, face_image_path):
    # Load the image and get the face encoding
    image = face_recognition.load_image_file(face_image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    encoded_face = pickle.dumps(face_encoding) # Serialize the face encoding

    conn = sqlite3.connect('known_faces.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO known_faces (name, face_encoding) VALUES (?, ?)", (name, encoded_face))
    conn.commit()
    conn.close()

def load_known_faces():
    conn = sqlite3.connect('known_faces.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM known_faces")
    known_faces = cursor.fetchall()
    
    known_face_encodings = []
    known_face_names = []
    
    for face in known_faces:
        encoding = pickle.loads(face[2]) # Deserialize the face encoding
        known_face_encodings.append(encoding)
        known_face_names.append(face[1])
    
    conn.close()
    
    return known_face_encodings, known_face_names

# Function to use ArcFace for recognition
def recognize_faces_with_arcface(image):
    img = np.array(image) # Convert PIL image to numpy array
    faces = app.get(img)
    
    for face in faces:
        name = f"Face {face.det_score:.2f}" # Using the detection score as a temporary name
        cv2.rectangle(img, (int(face.bbox[0]), int(face.bbox[1])), 
                      (int(face.bbox[2]), int(face.bbox[3])), 
                      (0, 255, 0), 2)
        cv2.putText(img, name, 
                    (int(face.bbox[0]), int(face.bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for Streamlit
    return img_rgb

# Function to recognize faces using the standard face_recognition library
def recognize_faces_with_known_faces(image, known_face_encodings, known_face_names):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB for face_recognition
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert back to RGB
    return image_rgb

# Load Haar Cascade classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces using Haar Cascade
def detect_faces_with_haar(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB
    return image_rgb

# Streamlit app interface
st.title("Face Detection and Recognition App")

# Create database (if it doesn't exist)
create_db()

# Upload a known face and add it to the database
with st.expander("Add a Known Face"):
    name = st.text_input("Enter Name of Person")
    face_image = st.file_uploader("Upload an Image of this Person", type=["jpg", "jpeg", "png"])
    if st.button("Add Face to Database", key="add_face_button"):
        if name and face_image:
            add_known_face(name, face_image)
            st.success(f"Face of {name} added to database!")
        else:
            st.error("Please provide both name and image.")

# Upload an image for face detection/recognition
uploaded_file = st.file_uploader("Upload an Image for Face Recognition", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the known faces from the database
    known_face_encodings, known_face_names = load_known_faces()

    # Open the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Option 1: Use Standard Face Recognition (for comparison)
    st.subheader("Recognizing faces using standard face_recognition library")
    face_found = False
    if len(known_face_encodings) > 0:
        try:
            face_recognition_image = recognize_faces_with_known_faces(image_bgr, known_face_encodings, known_face_names)
            st.image(face_recognition_image, caption="Face Recognition (Standard)", use_container_width=True)
            face_found = True
        except Exception as e:
            st.error(f"Standard face recognition failed: {e}")

    # Option 2: Always use ArcFace (more accurate)
    st.subheader("Recognizing faces using ArcFace (Deep Learning-based)")
    arcface_image = recognize_faces_with_arcface(image)
    st.image(arcface_image, caption="ArcFace Recognition", use_container_width=True)

    # Option 3: Use Haar Cascade for face detection
    st.subheader("Detecting faces using Haar Cascade")
    haar_image = detect_faces_with_haar(image_bgr)
    st.image(haar_image, caption="Haar Cascade Detection", use_container_width=True)

# Sidebar for webcam control
st.sidebar.title("Webcam Control")
webcam_on = st.sidebar.checkbox("Turn Webcam On")

if webcam_on:
    # Add a loop to capture frames and display them in Streamlit
    cap = cv2.VideoCapture(0)  # Open the default camera

    frame_placeholder = st.empty()  # Placeholder for video frames

    # Create the stop button outside the loop
    if st.sidebar.button("Stop Video Capture", key="stop_video_button"):
        st.session_state["stop_video"] = True
    else:
        st.session_state["stop_video"] = False

    while not st.session_state["stop_video"]:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face recognition on the frame
        known_face_encodings, known_face_names = load_known_faces()
        if len(known_face_encodings) > 0:
            try:
                frame_with_faces = recognize_faces_with_known_faces(frame, known_face_encodings, known_face_names)
            except Exception as e:
                st.error(f"Standard face recognition failed: {e}")
                frame_with_faces = frame_rgb
        else:
            frame_with_faces = frame_rgb

        # Display the frame in Streamlit
        frame_placeholder.image(frame_with_faces, channels="RGB", use_container_width=True)

        # Check if the stop button is pressed
        if st.session_state["stop_video"]:
            break

    # Release the capture
    cap.release()
else:
    st.warning("Webcam is turned off. Use the sidebar to turn it on.")