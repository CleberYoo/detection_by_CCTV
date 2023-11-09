import cv2

image = cv2.imread("dataset/hottie.png")
image_gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

cv2.imwrite(filename="dataset/hottie_gray.jpg", img=image_gray)
# -------------------- 원본 이미지와 흑백 이미지 영상 출력하기 --------------------
cv2.imshow(winname="hottie", mat=image)
cv2.imshow(winname="gray hottie", mat=image_gray)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()     # 열린 모든 OpenCV 창을 닫는 메소드