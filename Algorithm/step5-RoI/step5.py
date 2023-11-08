import cv2
import time
import matplotlib.pyplot as plt

# frame_processing_time_list = []
capture = cv2.VideoCapture(filename="./여의도사거리2.mov")
object_detector = cv2.createBackgroundSubtractorMOG2()

"""save result video"""
# width, height = int(capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(capture.get(propId=cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter.fourcc(c1='F', c2='M', c3='P', c4='4')
# output = cv2.VideoWriter(filename="./step5-RoI/target to Region of Interest.mp4", fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

while True:
    # timer_start = time.time()

    return_value, frame = capture.read()
    if return_value == False: break

    frame_Region_of_Interest = frame[100:400, 1400:1700]      # 해당 지역 자르고
    # 주의!!!!!!!!!1 : 위 코드를 copy메소드로 해당 지역을 복사해서 object detection을 처리했다면 cv2.imshow() 시에 안나옴

    MoG_mask = object_detector.apply(image=frame_Region_of_Interest)    # 그 지역만 탐지하고
    
    contours, _ = cv2.findContours(image=MoG_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)      # 그 지역 contouring하고
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area >= (10 ** 2):
            cv2.drawContours(image=frame_Region_of_Interest, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2)      # contour한 부분(RoI)을 원본에 붙이고
    # output.write(image=frame_Region_of_Interest)


    # timer_end = time.time()
    # frame_processing_time = round(number=timer_end - timer_start, ndigits=4)
    # frame_processing_time_list.append(frame_processing_time)

    cv2.imshow(winname="video", mat=frame)      # 재생
    cv2.imshow(winname='Mixture of Gaussian mask', mat=frame_Region_of_Interest)

    key = cv2.waitKey(delay=1)
    if key == 27: break


"""make the file"""
# with open(file="./step5-RoI/processing time of target the Region of Interest.txt", mode='w') as file:
#     for item in frame_processing_time_list:
#         file.write(str(item) + "\n")
#     file.write(f"maximum : {max(frame_processing_time_list)}")
#     file.write(f"minimum : {min(frame_processing_time_list)}")
#     file.write(f"mean : {round(number=sum(frame_processing_time_list) / len(frame_processing_time_list), ndigits=3)}")


"""plot the data"""
# plt.plot(frame_processing_time_list)
# plt.title(label="Region of Interest"); plt.xlabel(xlabel="frame"); plt.ylabel(ylabel="proccessing time")
# plt.savefig("./step5-RoI/step5.png"); plt.show()


# output.release()
capture.release()
cv2.destroyAllWindows()