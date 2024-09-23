import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load the YOLOv8n-pose model
model = YOLO('yolov8n-pose.pt')

def process_image(image):
    results = model(image)
    return results[0].plot()

def process_video(video):
    cap = cv2.VideoCapture(video)
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        output = results[0].plot()
        stframe.image(output, channels="BGR")
    cap.release()

st.title("YOLOv8n Pose Detection")

# Sidebar for input selection
input_option = st.sidebar.radio("Select input type:", ["Image", "Video", "Webcam"])

if input_option == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect Poses"):
            result_image = process_image(image)
            st.image(result_image, caption="Pose Detection Result", use_column_width=True)

elif input_option == "Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if st.button("Detect Poses"):
            process_video(tfile.name)

elif input_option == "Webcam":
    if st.button("Start Webcam"):
        process_video(1)  # 0 is the default camera index

st.sidebar.markdown("---")
st.sidebar.write("Powered by YOLOv8 and Streamlit")

