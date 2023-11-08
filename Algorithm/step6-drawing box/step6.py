"""
make bounding box
&
counting object
history : 과거 몇 프레임을 고려하여 배경을 모델링할 것인지를 결정
varThreshold : 배경과 객체의 차이를 판단하기 위한 임계값
"""
import cv2
import tracker

capture = cv2.VideoCapture(filename="./여의도사거리2.mov")
object_detector = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)      # 300 frame : 5초
tracker = tracker.EuclideanDistTracker()

"""save the video"""
# width, height = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(capture.get(propId=cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter.fourcc(c1='F', c2='M', c3='P', c4='4')
# output = cv2.VideoWriter(filename="./step6-drawing box/result.mp4", fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

while True:

    return_value, frame = capture.read()
    if return_value == False:   break

    frame_Region_of_Interest = frame[000:100, 1500:1856]

    MoG_mask = object_detector.apply(image=frame_Region_of_Interest)
    brightness, MoG_mask = cv2.threshold(src=MoG_mask, thresh=16, maxval=255, type=cv2.THRESH_BINARY)        # thresh인자 : 어느 pixel값 아래는 무시할 것인지, maxval인자 : thresh 이상의 픽셀들은 다 maxval로 바꿈

    object_locations = []

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= (20 ** 2):
            x, y, w, h = cv2.boundingRect(array=contour)
            object_locations.append([x, y, w, h])

    boxes_ids = tracker.update(objects_rect=object_locations)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(img=frame_Region_of_Interest, text=str(id), org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=3)
        cv2.rectangle(img=frame_Region_of_Interest, pt1=(x, y), pt2=(x+w, y+h), color=(0, 255, 0), thickness=2)

    # output.write(image=frame)

    # cv2.imshow(winname='mask', mat=MoG_mask)
    # cv2.imshow(winname="Intersection", mat=frame)
    key = cv2.waitKey(delay=1)
    if key == 27:   break

# output.release()
capture.release()
cv2.destroyAllWindows()