import joblib

from PIL import Image

import numpy as np

import torch
import albumentations as A

from model import TextRecognizer
import config

chkpoint = torch.load(config.RECOGNIZER_PATH, map_location="cpu")
model = TextRecognizer(config.NUM_CLASSES+1)
model.to(config.DEVICE)

model.load_state_dict(chkpoint['model_state_dict'])
print("loaded the model")

idx_to_char, char_to_idx = joblib.load(config.LABEL_MAPPER)
print("loaded the encoder")

EPSILON = 'ε'

def arr_to_tnsr(img_arr):
    resize = (75, 300)
    transfroms = A.Compose([
            A.ToGray(always_apply=True),
            A.Normalize(always_apply=True)
        ])

    image = Image.fromarray(img_arr)
    image = image.resize(
                (resize[1], resize[0]), resample=Image.BILINEAR
            )
    img_arr = np.array(image)

    transformed = transfroms(image=img_arr)

    img = transformed['image']
    #channel first
    img = np.transpose(img, (2, 0, 1))

    img_tnsr = torch.tensor(img, dtype=torch.float32, device=config.DEVICE)
    
    return img_tnsr

def pred_tnsr_to_str(preds):
    global label_mapper
    pred_arr = preds.numpy()
    pred_str = ""
    for tok in pred_arr:
        if tok == 0:
            pred_str += EPSILON
        else:
            pred_str += idx_to_char[tok-1]
    
    return pred_str

def decode_pred(preds):
    res = ""
    prev = preds[0]
    for idx in range(1, len(preds)):
        curr = preds[idx]
        if prev != EPSILON and idx==1:
            res += prev
            
        if curr !=EPSILON  and prev != curr:
            res += curr
        prev = curr
    return res

def recognize_text(image: np.array):
    global model
    
    model.eval()
    
    img_tnsr = arr_to_tnsr(image)

    with torch.no_grad():
        preds = model(img_tnsr.unsqueeze(0))
        
    preds = preds.squeeze(0)
    
    probs = torch.softmax(preds, dim=-1)
    pred_labels = torch.argmax(probs, dim=-1)
    
    pred_str = pred_tnsr_to_str(pred_labels)
    decoded_pred = decode_pred(pred_str)
    
    return pred_labels, decoded_pred
    
if __name__ == "__main__":
    image_path = "../data/train/txt_images/0-13.jpg"
    image = Image.open(image_path)
    img_arr = np.array(image)
    label, string = recognize_text(img_arr)
    # print(label)
    print(string)