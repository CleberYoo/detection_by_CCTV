import cv2
from ultralytics import YOLO
import numpy as np

import torch
print(torch.backends.mps.is_available())
import classList

class_list = classList.class_list

cap = cv2.VideoCapture(filename="dataset/잠실대교1080.mov")

model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="mps")
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(result.boxes.cls.cpu(), dtype='int')

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x2, y2), color=(0, 0, 225), thickness=5)
        cv2.putText(img=frame, text=class_list[cls], org=(x, y - 5), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(0, 0, 225), thickness=5)
    
    # print(bboxes)

    cv2.imshow(winname="Img", mat=frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()