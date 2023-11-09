import cv2
import numpy as np

ray_charles = cv2.imread(filename="dataset/ray charles.jpeg")

rows, cols, ch = ray_charles.shape



cv2.imshow(winname="ray charles", mat=ray_charles)
cv2.waitKey(0)