import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using a pre-trained YOLOv8 model

st.title("YOLOv8 Object Detection")

# Allow users to upload an image or video
upload_option = st.selectbox("Choose input type", ("Image", "Video", "Webcam"))

# Function to perform object detection on images
def detect_image(image):
    results = model(image)
    annotated_frame = results[0].plot()
    return annotated_frame

# Image input
if upload_option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        result_image = detect_image(image)
        st.image(result_image, caption="Detected Image", use_column_width=True)

# Video input
elif upload_option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())

        # Read video frames
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = detect_image(frame)
            stframe.image(result_frame, channels="BGR")
        cap.release()

# Webcam input
elif upload_option == "Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            result_frame = detect_image(image)
            return result_frame

    webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)

