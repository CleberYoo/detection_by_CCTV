import cv2
import numpy as np

hotties = cv2.imread(filename="dataset/hottie.png")     # 이미지 읽어오기 

shape = hotties.shape
print(shape)        # (1792, 1382, 3)
print(int(shape[0] / 10), int(shape[1] / 10))       # (179, 138)


cv2.putText(img=hotties, text="ASS", org=(150, 1050), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3, color=(0, 0, 255), thickness=7)
cv2.line(img=hotties, pt1=(150, 1050), pt2=(675, 1050), color=(0, 0, 255), thickness=7)          # color : BGR
cv2.line(img=hotties, pt1=(150, 1400), pt2=(675, 1400), color=(0, 0, 255), thickness=7)
cv2.line(img=hotties, pt1=(150, 1050), pt2=(150, 1400), color=(0, 0, 255), thickness=7)
cv2.line(img=hotties, pt1=(675, 1050), pt2=(675, 1400), color=(0, 0, 255), thickness=7)

# cv2.line(img=hotties, pt1=(750, 1075), pt2=(750, 1375), color=(0, 0, 255), thickness=5)
# cv2.line(img=hotties, pt1=(1175, 1075), pt2=(1175, 1375), color=(0, 0, 255), thickness=5)
# cv2.line(img=hotties, pt1=(750, 1075), pt2=(1175, 1075), color=(0, 0, 255), thickness=5)
# cv2.line(img=hotties, pt1=(750, 1375), pt2=(1175, 1375), color=(0, 0, 255), thickness=5)
cv2.putText(img=hotties, text="ASS", org=(750,1075), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 0, 255), thickness=5)
cv2.rectangle(img=hotties, pt1=(750, 1075), pt2=(1175, 1375), color=(0, 0, 255), thickness=5)       # 위의 4 줄이 이 한 줄로 요약된다.

# cv2.circle(img=hotties, center=(500, 1300), radius=50, color=(64, 64, 255), thickness=-1)

# cv2.ellipse(img=hotties, center=(100, 100), axes=(100, 50), angle=0, startAngle=0, endAngle=360, color=(128, 128, 255), thickness=5)

# points = np.array([[100, 100],
#                    [300, 300],
#                    [200, 400],
#                    [300, 700]])
# cv2.polylines(img=hotties, pts=[points], isClosed=True, color=(0, 255, 255), thickness=5)

cv2.imwrite(filename="dataset/ass_detection.jpg", img=hotties)
cv2.imshow(winname="hottie.png", mat=hotties)       # 이미지 출력하기
cv2.waitKey(delay=0)
cv2.destroyAllWindows()