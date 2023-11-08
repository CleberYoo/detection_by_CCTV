import matplotlib.pyplot as plt


"""MoG"""
import cv2
import tracker
import time

frame_processing_time_list_MoG = []
capture = cv2.VideoCapture(filename="./여의도사거리2.mov")
object_detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
tracker = tracker.EuclideanDistTracker()

while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False: break

    frame_Region_of_Interest = frame[000:100, 1500:1856]

    MoG_mask = object_detector.apply(image=frame_Region_of_Interest)

    object_locations = []

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= 20 ** 2:
            x, y, w, h = cv2.boundingRect(array=contour)
            object_locations.append([x, y, w, h])
            # cv2.drawContours(image=frame_Region_of_Interest, contours=contour, contourIdx=-1, color=(255, 255, 255), thickness=5)

    boxes_ids = tracker.update(objects_rect=object_locations)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(img=frame_Region_of_Interest, text=str(id), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 255, 255), thickness=3)
        cv2.rectangle(img=frame_Region_of_Interest, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=5)

    timer_end = time.time()
    frame_processing_time = round(number=timer_end - timer_start, ndigits=4)
    frame_processing_time_list_MoG.append(frame_processing_time)

    # cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()





"""YOLOv8"""
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import classList
import time
class_list = classList.class_list

print(torch.backends.mps.is_available())

frame_processing_time_list_yolo = []
capture = cv2.VideoCapture(filename="./여의도사거리2.mov")
model = YOLO(model="yolov8m.pt")

"""save the video"""
# width, height = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(capture.get(propId=cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter.fourcc(c1='F', c2='M', c3='P', c4='4')
# output = cv2.VideoWriter(filename="./YOLO.mp4", fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False: break

    frame_Region_of_Interest = frame[000:100, 1500:1856]

    results = model(source=frame_Region_of_Interest)
    result = results[0]

    bounding_boxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(result.boxes.cls.cpu(), dtype='int')

    for class_, bounding_box in zip(classes, bounding_boxes):
        x1, y1, x2, y2 = bounding_box

        cv2.rectangle(img=frame_Region_of_Interest, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=5)
        cv2.putText(img=frame_Region_of_Interest, text=str(class_), org=(x1, y1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 255, 255), thickness=2)

    # output.write(image=frame)

    timer_end = time.time()
    frame_processing_time = round(number=timer_end - timer_start, ndigits=4)
    frame_processing_time_list_yolo.append(frame_processing_time)

    # cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27: break

# output.release()
capture.release()
# cv2.destroyAllWindows()


"""plot the data"""
plt.plot(frame_processing_time_list_MoG, 'b', label="MoG")
plt.plot(frame_processing_time_list_yolo, 'r', label="YOLO")
plt.title(label="Comparison on the same condition"); plt.xlabel(xlabel="frame"); plt.ylabel(ylabel="processing time")
plt.legend()
plt.savefig("./comparison YOLO_MoG/comparson.png")
plt.show()

"""make the file"""
with open(file="./comparison YOLO_MoG/comaprison YOLO and MoG.txt", mode='w') as file:
    file.write("\t\t\tMoG\t\tYOLO\n")
    for item1, item2 in zip(frame_processing_time_list_MoG, frame_processing_time_list_yolo):
        file.write(f"\t\t\t{str(item1)}\t\t{str(item2)}\n")
    file.write("\n")
    file.write(f"maximum : \t{max(frame_processing_time_list_MoG)}\t\t{max(frame_processing_time_list_yolo)}\n")
    file.write(f"minimum : \t{min(frame_processing_time_list_MoG)}\t\t{min(frame_processing_time_list_yolo)}\n")
    file.write(f"mean : \t\t{round(number=sum(frame_processing_time_list_MoG) / len(frame_processing_time_list_MoG), ndigits=4)}\t\t{round(number=sum(frame_processing_time_list_yolo) / len(frame_processing_time_list_yolo), ndigits=4)}")