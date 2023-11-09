import cv2
import numpy as np

image_cat = cv2.imread(filename="dataset/cat.jpeg")
image_heaven = cv2.imread(filename="dataset/heaven.jpeg")
# cv2.imshow(winname='cat', mat=image_cat)
# cv2.imshow(winname='heaven', mat=image_heaven)

image_cat_gray = cv2.cvtColor(src=image_cat, code=cv2.COLOR_BGR2GRAY)
cv2.imshow(winname='cat gray', mat=image_cat_gray)

# sum = cv2.add(src1=image_cat, src2=image_heaven)
# cv2.imshow(winname='fussion', mat=sum)

# weighted_sum = cv2.addWeighted(src1=image_cat, alpha=0.5, src2=image_heaven, beta=0.5, gamma=0)
# cv2.imshow(winname='weighted sum.png', mat=weighted_sum)

return_value, threshold = cv2.threshold(src=image_cat_gray, thresh=32, maxval=128, type=cv2.THRESH_BINARY)
cv2.imshow(winname="thresholded cat grayscaled", mat=threshold)

"""maxval을 높게 잡으면 말그대로 최댓값의 한계가 커져서 아래 코드라인은 255까지 표시할 수 있게 된다"""
# return_value2, threshold2 = cv2.threshold(src=image_cat_gray, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
# cv2.imshow(winname="thresholded cat grayscaled2", mat=threshold2)

cv2.waitKey(delay=0)
cv2.destroyAllWindows()