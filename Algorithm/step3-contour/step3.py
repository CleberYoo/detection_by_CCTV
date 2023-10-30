"""
step 3 : find a contour
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from file_write import write_list

frame_processing_time_list = []

capture = cv2.VideoCapture(filename="./여의도사거리.mov")

"""save result video"""
width = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(propId=cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter.fourcc(c1='F', c2='M', c3='P', c4='4')
output = cv2.VideoWriter(filename="./step3-contour/contouring on video.mp4", fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False:   break

    MoG_mask = object_detector.apply(image=frame)

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(image=frame, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2)
    output.write(image=frame)

    timer_end = time.time()
    frame_processing_time = round(number=(timer_end - timer_start), ndigits=3)
    frame_processing_time_list.append(frame_processing_time)


    """show the video"""
    cv2.imshow(winname="Mixture of Gaussian Mask", mat=MoG_mask)
    cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27:   break

print(f"length of frame : {len(frame_processing_time_list)}")
plt.plot(frame_processing_time_list)
plt.title(label="Contouring"); plt.xlabel(xlabel="frame"); plt.ylabel(ylabel="processing time")
plt.savefig("./step3-contour/step3.png"); plt.show()

write_list(list=frame_processing_time_list, file_path='./step3-contour/processing time of contour on video.txt')

output.release()
capture.release()       # <-- 메모리 해제
cv2.destroyAllWindows()     # <-- cv2.imshow()로 생성된 창들 제거