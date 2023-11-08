import cv2
import matplotlib.pyplot as plt
import time

before_frame_processing_time_list = []
capture = cv2.VideoCapture(filename="./여의도사거리.mov")
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False: break

    MoG_mask = object_detector.apply(image=frame)

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= (10 ** 2):       cv2.drawContours(image=frame, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2)

    timer_end = time.time()
    frame_processing_time = round(number=timer_end - timer_start, ndigits=4)
    before_frame_processing_time_list.append(frame_processing_time)

    # cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()


print("done")



after_frame_processing_time_list = []
capture = cv2.VideoCapture(filename="./여의도사거리.mov")
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False: break

    frame_Region_of_Interest = frame[100:400, 1400:1700]

    MoG_mask = object_detector.apply(image=frame_Region_of_Interest)

    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= (10 ** 2):       cv2.drawContours(image=frame_Region_of_Interest, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2)

    timer_end = time.time()
    frame_processing_time = round(number=timer_end - timer_start, ndigits=4)
    after_frame_processing_time_list.append(frame_processing_time)

    # cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27: break

capture.release()
cv2.destroyAllWindows()


print("done2")


"""make the file"""
with open(file="./step5-RoI/processing time before and after RoI.txt", mode='w') as file:
    file.write("\t\t\tbefore\t\tafter\n")
    for i in range(len(before_frame_processing_time_list)):
        file.write(f"\t\t\t{str(before_frame_processing_time_list[i])}\t\t{str(after_frame_processing_time_list[i])}\n")
    file.write("\n")
    file.write(f"maximum :\t{max(before_frame_processing_time_list)}\t\t{max(after_frame_processing_time_list)}\n")
    file.write(f"minimum :\t{min(before_frame_processing_time_list)}\t\t{min(after_frame_processing_time_list)}\n")
    file.write(f"mean :\t\t{round(number=sum(before_frame_processing_time_list) / len(before_frame_processing_time_list), ndigits=4)}\t\t{round(number=sum(after_frame_processing_time_list) / len(after_frame_processing_time_list), ndigits=4)}")


"""plot the data"""
# plt.plot(before_frame_processing_time_list, "#FF0000", label="before RoI")
# plt.plot(after_frame_processing_time_list, "#0000FF", label="after RoI")
# plt.title(label="comparison between before & after RoI"); plt.xlabel(xlabel="frame"); plt.ylabel(ylabel="processing time"); plt.legend()
# plt.savefig("./step5-RoI/comparison between before & after RoI.png")
# plt.show()