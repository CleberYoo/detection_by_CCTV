"""reduce noise"""
import cv2
import time
import matplotlib.pyplot as plt

frame_processing_time_list = []
capture = cv2.VideoCapture(filename="./여의도사거리.mov")
object_detector = cv2.createBackgroundSubtractorMOG2()

"""save result video"""
width, height = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(capture.get(propId=cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter.fourcc(c1='F', c2='M', c3='P', c4='4')
output = cv2.VideoWriter(filename='./step4-reduce noise/reduction noise for contoured video.mp4', fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)


while True:
    timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False:   break

    MoG_mask = object_detector.apply(image=frame)

    """contouring"""
    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour=contour)                                                                 #
        if area >= (10 ** 2):                                                                                   #
            cv2.drawContours(image=frame,  contours=contour,contourIdx=-1, color=(0, 255, 0), thickness=2)      #
    output.write(image=frame)

    timer_end = time.time()
    frame_processing_time = round(number=(timer_end - timer_start), ndigits=3)
    frame_processing_time_list.append(frame_processing_time)

    """show the video"""
    cv2.imshow(winname="video", mat=frame)

    key = cv2.waitKey(delay=1)
    if key == 27:   break


"""make the file"""
with open(file="./step4-reduce noise/procesing time of reducing noise & contouring on video.txt", mode='w') as file:
    for item in frame_processing_time_list:
        file.write(str(item) + "\n")
    file.write(f"maximum : {max(frame_processing_time_list)}\n")
    file.write(f"minimum : {min(frame_processing_time_list)}\n")
    file.write(f"mean : {round(number=sum(frame_processing_time_list) / len(frame_processing_time_list), ndigits=3)}")


"""plot the data"""
plt.plot(frame_processing_time_list)
plt.title(label="reduce the noise"); plt.xlabel(xlabel="frame"); plt.ylabel(ylabel="processing time")
plt.savefig("./step4-reduce noise/step4.png"); plt.show()

output.release()
capture.release()
cv2.destroyAllWindows()