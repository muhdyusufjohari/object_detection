import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from PIL import Image  # Instead of using cv2 for image handling

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

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
        image = Image.open(uploaded_image)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        result_image = detect_image(image)
        st.image(result_image, caption="Detected Image", use_column_width=True)

# Video input
elif upload_option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        # Use alternative video reading methods without cv2.VideoCapture
        import imageio

        cap = imageio.get_reader(tfile.name, 'ffmpeg')
        stframe = st.empty()

        for frame in cap:
            frame = np.array(frame)
            result_frame = detect_image(frame)
            stframe.image(result_frame, channels="BGR")

# Webcam input
elif upload_option == "Webcam":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")
            result_frame = detect_image(image)
            return result_frame

    webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer)
