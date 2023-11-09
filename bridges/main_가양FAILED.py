"""가양대교 주간 단방향 탐색 FAIL"""

import cv2
import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    


upload_url = "http://localhost:4000/api/videos/upload"

capture = cv2.VideoCapture(filename="./bridges/dataset/가양대교/2023-10-26 15-28-20oneway.mov")
object_detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=4, detectShadows=True)
tracker = EuclideanDistTracker()

while True:
    return_value, frame = capture.read()
    if return_value == False: break


    frame_Region_of_Interest = frame[800:875, 700:800]
    # frame_Region_of_Interest += 50

    MoG_mask = object_detector.apply(image=frame_Region_of_Interest)
    brightness, MoG_mask = cv2.threshold(src=MoG_mask, thresh=8, maxval=255, type=cv2.THRESH_BINARY)

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

    cv2.imshow(winname="GaYang Bridge", mat=frame)
    cv2.imshow(winname="mask", mat=MoG_mask)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()