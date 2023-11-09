import cv2
import numpy as np
capture = cv2.VideoCapture("dataset/nba.mov")

fourcc = cv2.VideoWriter.fourcc(c1='X', c2='V', c3='I', c4='D')
output = cv2.VideoWriter(filename="dataset/flipped_nba.avi",fourcc=fourcc, fps=25, frameSize=(900, 1440))

while True:
    retutrn_value, frame = capture.read()
    
    frame2 = cv2.flip(src=frame, flipCode=1)

    # cv2.imshow(winname="original", mat=frame)
    cv2.imshow(winname="flipped", mat=frame2)

    output.write(frame2)

    key = cv2.waitKey(delay=1)
    if key == 27:
        break

output.release()
capture.release()
cv2.destroyAllWindows()











def load_camera(device=1, flip=None):
    """
    device=0 : cell phone camera
    device=1 : laptop camera
    flip=0 : vertical flip
    flip=1 : horizontal flip
    """
    
    capture = cv2.VideoCapture(device)

    while True:
        return_value, frame = capture.read()
        
        if flip == 0:
            frame = cv2.flip(src=frame, flipCode=0)
        elif flip == 1:
            frame = cv2.flip(src=frame, flipCode=1)
        else:
            pass

        cv2.imshow(winname=f"camera {device}", mat=frame)

        key = cv2.waitKey(delay=1)

        if key == 27:           # 27 means 'esc' button         followed by ascii code, 27 is escape(esc)
            break

    capture.release()   # 비디오 캡쳐 객체 해제하여 메모리 절약
    cv2.destroyAllWindows()     # 열려있는 모든 openCV 창 닫기


def load_camera_gray(device=1, flip=None):
    """
    device=0 : cell phone camera
    device=1 : laptop camera
    flip=0 : vertical flip
    flip=1 : horizontal flip
    """

    capture = cv2.VideoCapture(device)

    while True:
        return_value, frame = capture.read()
        gray_scale_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        if flip == 0:       # vertical
            gray_scale_frame = cv2.flip(src=gray_scale_frame, flipCode=0)
        elif flip == 1:     # horizontal
            gray_scale_frame = cv2.flip(src=gray_scale_frame, flipCode=1)
        else:
            pass

        cv2.imshow(winname=f"camera {device}", mat=gray_scale_frame)

        key = cv2.waitKey(delay=1)

        if key == 27:           # 27 means 'esc' button         followed by ascii code, 27 is escape(esc)
            break

    capture.release()   # 비디오 캡쳐 객체 해제하여 메모리 절약
    cv2.destroyAllWindows()     # 열려있는 모든 openCV 창 닫기