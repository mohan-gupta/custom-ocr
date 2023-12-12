import io
import base64

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, UploadFile

from ocr import perform_ocr

app = FastAPI()

def encode_image(array, size):
    pil_img = Image.fromarray(array, 'RGB')
    pil_img = pil_img.resize(size)

    arr = np.array(pil_img)
    pil_img = Image.fromarray(arr, 'RGB')
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())

    img_string = img_bytes.decode()
    
    return img_string

def decode_base64(base64_str):
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
    
    np_arr = np.array(img)
    
    return np_arr

@app.get('/')
def home():
    return {'data': "Welcome to the OCR Project!!"}

@app.post("/predict")
async def predict(image: UploadFile):
    # read the image
    contents = await image.read()
    np_img = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # perform ocr on the image
    img_pred, preds = perform_ocr(img[:,:,::-1])

    #converting result to base64 string
    img_string = encode_image(img_pred, img.shape[:-1][::-1])
    
    return {"output": img_string, "preds": preds}

