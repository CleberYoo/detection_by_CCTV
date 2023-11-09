import cv2
import mediapipe as mp
import time

maximum_time = 15

#Load Face Detector
face_detection = mp.solutions.face_detection.FaceDetection()

# activate laptop camera
capture = cv2.VideoCapture(1)

# track TIME
starting_time = time.time()

while True:
    # return_value : True/False     frame : image array
    return_value, frame = capture.read()

    height, width, channels = frame.shape

    frame = cv2.flip(src=frame, flipCode=1)
    rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

    # Draw rectangle
    cv2.rectangle(img=frame, pt1=(0, 0), pt2=(width, 70), color=(10, 10, 10), thickness=-1)

    results = face_detection.process(rgb_frame)
    # print(results.detections)

    # Is there face DETECTED?
    if results.detections:
        elapsed_time = int(time.time() - starting_time)

        if elapsed_time > maximum_time:
            cv2.rectangle(img=frame, pt1=(0, 0), pt2=(width, height), color=(0, 0, 255), thickness=10)

        # Draw elapsed time on screen
        cv2.putText(img=frame, text=f"{elapsed_time} seconds", org=(10, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 0, 255), thickness=2)

        print(f"Elapsed : {elapsed_time}")
        print("Face looking at the screen")
    
    else:
        print("No FACE")
        starting_time = time.time()

    cv2.imshow(winname="selfie", mat=frame)

    key = cv2.waitKey(delay=1)

    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()