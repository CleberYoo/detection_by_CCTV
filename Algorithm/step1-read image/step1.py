"""
step 1 : read video(set of continous image)
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from file_write import write_list

frame_processing_time_list = []

capture = cv2.VideoCapture(filename="./여의도사거리.mov")

while True:
    timer_start = time.time()                   # timer start

    return_value, frame = capture.read()

    if return_value == False:   break
    

    timer_end = time.time()                     # timer end
    frame_proccessing_time = round(number=timer_end - timer_start, ndigits=3)                   # 소숫점 3번째 자리까지 표현
    frame_processing_time_list.append(frame_proccessing_time)

    cv2.imshow(winname="Intersection", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27:   break       # escape button



print(f"length : {len(frame_processing_time_list)}")
plt.title(label="Read Video");  plt.xlabel(xlabel="frame");     plt.ylabel(ylabel="processing time")
plt.plot(frame_processing_time_list)
plt.savefig("./step1-read image/step 1.png");   plt.show()

write_list(list=frame_processing_time_list, file_path='./step1-read image/processing time of original.txt')

capture.release()
cv2.destroyAllWindows()