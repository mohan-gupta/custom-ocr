import streamlit as st

import numpy as np
from PIL import Image

import os
import base64

from ocr import perform_ocr

dir_path = os.path.dirname(__file__)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

if __name__ == '__main__':
    add_bg_from_local(os.path.join(dir_path, "../artifacts/background.jpg"))
    new_title = '<p style="font-family:sans-serif; color:black; font-size: 42px;">Receipts OCR</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a file")
    
    col1, col2 = st.columns(2)
    preds = []
    with col1:
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
    
            st.image(img)
                    
            if st.button("Predict"):
                img_pred, preds = perform_ocr(np.array(img))

                pred_image = Image.fromarray(img_pred, 'RGB')
                pred_image = pred_image.resize(img.size)
                with col2:
                    st.image(pred_image)

                for idx in range(0, len(preds)-1, 2):
                    key, val = preds[idx]
                    st.write(f"{key}: {val}")
                    with col2:
                        key, val = preds[idx+1]
                        st.write(f"{key}: {val}")
                    