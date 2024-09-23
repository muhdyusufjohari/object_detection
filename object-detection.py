import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

class ObjectDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_frame = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

def main():
    st.title("Object Detection App")
    
    st.write("Choose an input method:")
    input_method = st.radio("", ["Upload Image", "Upload Video", "Live Webcam"])
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
            results = model(image)
            st.image(results[0].plot(), channels="BGR")
    
    elif input_method == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            st.video(uploaded_file)
            st.write("Video object detection is not implemented in this demo.")
    
    elif input_method == "Live Webcam":
        st.write("Click 'Start' to begin webcam object detection")
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            video_processor_factory=ObjectDetector,
            async_processing=True,
        )

if __name__ == "__main__":
    main()

