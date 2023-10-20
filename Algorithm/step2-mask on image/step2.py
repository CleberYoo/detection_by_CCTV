"""
step 2 : apply Mixture of Gaussian object detector
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from file_write import write_list

capture = cv2.VideoCapture(filename="./여의도사거리.mov")

"""save result video"""
width = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(propId=cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter.fourcc(c1='F', c2='M',c3='P', c4='4')
output = cv2.VideoWriter(filename='./step2-mask on image/masked video.mp4', fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=False)

object_detector = cv2.createBackgroundSubtractorMOG2()

frame_processing_time_list = []

while True:
    # timer_start = time.time()
    return_value, frame = capture.read()

    if return_value == False:   break

    MoG_mask = object_detector.apply(image=frame)                   # training

    # timer_end = time.time()
    # frame_processing_time = round(number=(timer_end - timer_start), ndigits=3)
    # frame_processing_time_list.append(frame_processing_time)


    output.write(image=MoG_mask)

    """show the video"""
    # cv2.imshow(winname="Original", mat=frame)
    # cv2.imshow(winname="Mixture of Gaussian", mat=mask)

    key = cv2.waitKey(delay=1)
    if key == 27:   break       # escape button



print(f"length : {len(frame_processing_time_list)}")
# plt.plot(frame_processing_time_list)
# plt.title(label="MoG Object Detector"); plt.xlabel(xlabel="frame"); plt.ylabel(ylabel="processing time")
# plt.savefig("./step2-mask on image/step2 ver.2.png"); plt.show()

# write_list(list=frame_processing_time_list, file_path='./step2-mask on image/processing time of mask on original.txt')

output.release()
capture.release()
# cv2.destroyAllWindows()       # <--- cv2.imshow()를 사용할 때만