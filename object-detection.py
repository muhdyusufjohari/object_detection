import streamlit as st
import tempfile
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import moviepy.editor as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Helper function to perform object detection on an image
def detect_image(image):
    results = model(image)
    return results[0].plot()  # Annotated image with bounding boxes

# Title of the Streamlit App
st.title("YOLOv8 Object Detection")

# Upload options: Image, Video, or Webcam
upload_option = st.selectbox("Choose Input Type", ("Image", "Video", "Webcam"))

# Image Upload
if upload_option == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        img_array = np.array(img)
        
        # Perform object detection
        result_image = detect_image(img_array)
        
        # Display the result
        st.image(result_image, caption="Detected Image", use_column_width=True)

# Video Upload using moviepy
elif upload_option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Load the video using moviepy
        video = mp.VideoFileClip(tfile.name)
        stframe = st.empty()
        
        # Loop through video frames and perform object detection
        for frame in video.iter_frames():
            result_frame = detect_image(frame)
            stframe.image(result_frame, channels="RGB")

# Webcam live detection using streamlit-webrtc
elif upload_option == "Webcam":

    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result_img = detect_image(img)
            return result_img

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
