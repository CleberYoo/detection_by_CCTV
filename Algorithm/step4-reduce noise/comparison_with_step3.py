import cv2
import time
import matplotlib.pyplot as plt

capture = cv2.VideoCapture(filename="./여의도사거리.mov")
list_before_reduce_noise = []
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False: break

    MoG_mask = object_detector.apply(image=frame)

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(image=frame, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2)

    timer_end = time.time()
    frame_processing_time = round(number=(timer_end - timer_start), ndigits=3)
    list_before_reduce_noise.append(frame_processing_time)

    # cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()



print(f"length : {len(list_before_reduce_noise)}")




capture = cv2.VideoCapture(filename="./여의도사거리.mov")
list_after_reduce_noise = []
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False: break

    MoG_mask = object_detector.apply(image=frame)

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= (10 ** 2):
            cv2.drawContours(image=frame, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2)

    timer_end = time.time()
    frame_processing_time = round(number=(timer_end - timer_start), ndigits=3)
    list_after_reduce_noise.append(frame_processing_time)

    # cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()

plt.plot(list_before_reduce_noise, '#0000FF', label='before reducing noise')
plt.plot(list_after_reduce_noise, '#FF0000', label='after reducing noise')
plt.title(label="comparison between before & after about noise reduction"); plt.xlabel(xlabel="frame"); plt.ylabel(ylabel="processing time")
plt.legend(); plt.savefig("./step4-reduce noise/comparison step3 & stpe4.png"); plt.show()