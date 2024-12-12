import streamlit as st
import cv2
import os
import sqlite3
from PIL import Image
import numpy as np
from datetime import datetime

# Database Setup
DB_FILE = "predictions.db"
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create the predictions table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT,
    prediction TEXT,
    timestamp TEXT
)
""")
conn.commit()

# Function to save predictions to the database
def save_to_database(file_name, prediction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO predictions (file_name, prediction, timestamp) VALUES (?, ?, ?)",
                   (file_name, prediction, timestamp))
    conn.commit()

# Function to query predictions from the database
def query_predictions():
    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    return rows

# Function for object detection in images
def detect_objects_in_image(image):
    """Performs object detection on the input image."""
    # Convert image to OpenCV format and grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(gray_image, (x, y), (x + w, y + h), (255, 255, 255), 2)

    prediction = f"Detected {len(faces)} face(s)."
    return gray_image, prediction

# Function for object detection in all frames of a video
def detect_objects_in_video(video_path):
    """Performs object detection on all frames of a video."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter to save the processed video
    processed_video_path = os.path.join("results", f"processed_{os.path.basename(video_path)}")
    os.makedirs("results", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        face_count += len(faces)
        out.write(gray_frame)

    cap.release()
    out.release()

    prediction = f"Processed video. Detected a total of {face_count} face(s)."
    return processed_video_path, prediction

# Function to display saved predictions
def display_predictions():
    st.header("Saved Predictions")
    results = query_predictions()
    if results:
        for row in results:
            st.write(f"ID: {row[0]}, File: {row[1]}, Prediction: {row[2]}, Timestamp: {row[3]}")
    else:
        st.write("No predictions found.")

# Main Streamlit App
def main():
    st.title("Image/Video Detection")

    # Image Upload Section
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Start Prediction (Image)"):
            processed_image, prediction = detect_objects_in_image(image)
            st.image(processed_image, caption="Processed Image", use_column_width=True, channels="GRAY")
            st.success(f"Prediction: {prediction}")

            # Save results
            file_name = uploaded_image.name
            save_to_database(file_name, prediction)
            result_path = os.path.join("results", file_name)
            os.makedirs("results", exist_ok=True)
            cv2.imwrite(result_path, processed_image)

    # Video Upload Section
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_video is not None:
        video_path = os.path.join("temp_videos", uploaded_video.name)
        os.makedirs("temp_videos", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        if st.button("Start Prediction (Video)"):
            processed_video_path, prediction = detect_objects_in_video(video_path)
            st.video(processed_video_path)
            st.success(f"Prediction: {prediction}")

            # Save results
            save_to_database(uploaded_video.name, prediction)

    # Show Saved Predictions
    if st.button("Saved Predictions"):
        display_predictions()

if __name__ == "__main__":
    main()
