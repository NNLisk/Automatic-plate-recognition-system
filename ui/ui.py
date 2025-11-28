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
    st.image(upload_file, caption="Uploaded image", use_column_widht=True)

    if st.button("Recognize plate"):
        with st.spinner("Processing..."):
            sessionPath, sessionID = make_new_session()
            with open(f"{sessionPath}/{sessionID}/rawImage", "wb") as rawimage:
                rawimage.write(upload_file.getbuffer())
            
            ## call processimage from pipeline here
            ## parameters filepath and 
            ## returns plate

            st.success(f"Detected plate: *RESULT*")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # display resulting image
                pass
            with col2:
                pass
            with col3:
                pass
            with col4:
                pass