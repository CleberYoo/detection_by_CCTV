"""월드컵대교 주간 X"""

import cv2
import tracker

capture = cv2.VideoCapture(filename="./bridges/dataset/방화대교/2023-10-10 14-06-15_TUEtrim.mov")
object_detector = cv2.createBackgroundSubtractorMOG2(history=540, varThreshold=16, detectShadows=True)
tracker = tracker.EuclideanDistTracker()

while True:
    return_value, frame = capture.read()
    if return_value == False: break


    frame_Region_of_Interest = frame[450:600, 500:1500]
    frame_Region_of_Interest += 50

    MoG_mask = object_detector.apply(image=frame_Region_of_Interest)
    brightness, MoG_mask = cv2.threshold(src=MoG_mask, thresh=32, maxval=255, type=cv2.THRESH_BINARY)

    object_location = []

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= 30 ** 2:
            x, y, w, h = cv2.boundingRect(array=contour)
            object_location.append([x, y, w, h])
            # cv2.drawContours(image=frame_Region_of_Interest, contours=contour, contourIdx=-1, color=(255, 255, 255), thickness=5)

    boxes_ids = tracker.update(objects_rect=object_location)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(img=frame_Region_of_Interest, text=str(id), org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=3)
        cv2.rectangle(img=frame_Region_of_Interest, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=3)

    cv2.imshow(winname="WorldCup Bridge", mat=frame)
    # cv2.imshow(winname="mask", mat=MoG_mask)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()