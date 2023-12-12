import numpy as np
import cv2

from ultralytics import YOLO

from recognizer import recognize_text
import config

detector = YOLO(config.DETECTOR_PATH)

def detect_text(img_path):    
    preds = detector(img_path)[0]
    
    boxes = preds.boxes.xyxy.cpu().numpy()
    
    return boxes

def perform_ocr(img):

    boxes = detect_text(img)
    colors = (50,40,60)
    
    img = np.ascontiguousarray(img, dtype=np.uint8)
    
    ocr_res = []    
    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = map(int, box)
        
        crpd_arr = img[ymin:ymax, xmin:xmax, :]

        _, rcg_txt = recognize_text(crpd_arr[:,:,::-1])
        
        ocr_res.append([idx, rcg_txt])
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors)
        cv2.putText(img, f"{idx}: "+rcg_txt, (xmin, ymin - 12), 0, 1e-3 * 1080, colors)
        
    return img, ocr_res

if __name__ == "__main__":
    img_path = "../data/train/0.jpeg"
    boxes = detect_text(img_path)
    res = perform_ocr(img_path, boxes)
    print(res)