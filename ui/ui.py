# UI needs to have:
###################
# a way to upload images, drag and drop or upload field or something
# dropdown field to select the OCR model
# find plates button

import streamlit as st
import cv2
import os, sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.pipeline import processImage
from src.utils.filer import make_new_session


st.title("ALPR")

upload_file = st.file_uploader("Upload image", type=['jpg', 'jpeg'])
model_choise = st.selectbox("Select OCR model", ['CNN classifier', 'other'])

if upload_file is not None:
    st.image(upload_file, caption="Uploaded image", width=400)

    if st.button("Recognize plate"):
        with st.spinner("Processing..."):
            sessionPath, sessionID = make_new_session()
            with open(f"{sessionPath}/rawinput.jpg", "wb") as rawimage:
                rawimage.write(upload_file.getbuffer())
            
            result_data, plate, confidence_values = processImage(sessionPath, sessionID)
            ## call processimage from pipeline here
            ## parameters filepath and 
            ## returns plate

            st.success(f"Detected plate: {plate}")
            st.success(f"confidence for each character:  {', '.join([str(i) for i in confidence_values])}")
        
            col1, col2, col3 = st.columns(3)
            container1 = st.container()
            container2 = st.container(horizontal=True)


            with col1:
                st.subheader("Cropped Image")
                st.image(f"{sessionPath}/cropped.jpg")
            with col2:
                st.subheader("thresholded Image")
                st.image(f"{sessionPath}/thresholded.jpg")
            with col3:
                st.subheader("Found characters")
                st.image(f"{sessionPath}/contours.jpg")
            # with col4:
            #     st.subheader("Example character")
            #     st.image(f"{sessionPath}/characters/1.jpg")

            with container1:            
                st.subheader("Individual characters")
                with container2:
                    for name in sorted(os.listdir(os.path.join(f"{sessionPath}/characters"))):
                        st.image(f"{sessionPath}/characters/{name}")