# Python In-built packages
import io

import PIL
import torch
from PIL import Image

# External packages
import streamlit as st
from ultralytics import YOLO, solutions
import cv2
import threading

from db import *

# Local Modules

st.set_page_config(
    page_title="tracking",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
VIDEO_DIR = 'videos'
DEFAULT_IMAGE = "st/images/office_4.jpg"
DEFAULT_DETECT_IMAGE = 'st/images/office_4_detected.jpg'
# confidence = 0.9


# Sources
IMAGE = 'Image'
VIDEO = 'Video'
RTSP = 'RTSP'
SOURCES_LIST = [IMAGE, VIDEO, RTSP]

# Sidebar
# st.sidebar.header("Object Detection")

confidence = float(st.sidebar.slider(
    "Confidence", 25, 100, 90, step=1)) / 100

model = YOLO("runs/detect/ds7_7_10_l/weights/best.pt")


# st.sidebar.header("Image/Video Config")

source_radio = st.sidebar.radio(
    "Выберите", SOURCES_LIST)

source_img = None

# If image is selecte
if source_radio == IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            # if st.sidebar.button('Detect Objects'):
            res = model.predict(uploaded_image,
                                conf=confidence
                                )
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                     use_column_width=True)
            try:
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                # st.write(ex)
                st.write("No image is uploaded yet!")

elif source_radio == RTSP:

    cameras = get_cameras_data()
    print(cameras)





elif source_radio == VIDEO:
    uploaded_file = st.sidebar.file_uploader("Выберите видео файл", type=("mp4", "avi", "mov"))

    if uploaded_file is not None:
        g = io.BytesIO(uploaded_file.read())
        vid_location = "ultralytics.mp4"
        with open(vid_location, 'wb') as f:
            f.write(g.read())
        vid_filename = "ultralytics.mp4"

        col1, col2 = st.columns(2)

        videocapture = cv2.VideoCapture(vid_filename)
        stop_button = st.button("Stop")

        org_frame = col1.empty()
        ann_frame = col2.empty()

        while videocapture.isOpened():
            success, image = videocapture.read()
            if not success:
                break

            results = model.track(image, conf=confidence, persist=True, model=model)

            # Add annotations on frame
            annotated_frame = results[0].plot()  # Add annotations on frame

            # display frame
            frame = cv2.resize(image, (720, int(720 * (9 / 16))))
            org_frame.image(frame, channels="BGR")
            ann_frame.image(annotated_frame, channels="BGR")

            if stop_button:
                videocapture.release()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.stop()

        if torch.cuda.is_available():
            # Clear CUDA memory
            torch.cuda.empty_cache()

        # Release the capture
        videocapture.release()
        cv2.destroyAllWindows()

    else:
        st.info("Please upload video file")
else:
    st.error("Please select a valid source type!")
