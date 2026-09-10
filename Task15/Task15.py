import cv2
import numpy as np

cap = cv2.VideoCapture("Task15/Counter-Strike 2 2026-07-04 01-06-44.mp4")

def nothing(x):
    pass

cv2.namedWindow("Trackbar")
cv2.resizeWindow("Trackbar", 500, 500)

cv2.createTrackbar("lower - H", "Trackbar", 0, 180, nothing)
cv2.createTrackbar("lower - S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("lower - V", "Trackbar", 0, 255, nothing)

cv2.createTrackbar("upper - H", "Trackbar", 0, 180, nothing)
cv2.createTrackbar("upper - S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("upper - V", "Trackbar", 0, 255, nothing)

cv2.setTrackbarPos("upper - H", "Trackbar", 180)
cv2.setTrackbarPos("upper - S", "Trackbar", 255)
cv2.setTrackbarPos("upper - V", "Trackbar", 255)

while True: 
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_h = cv2.getTrackbarPos("lower - H", "Trackbar")
    lower_s = cv2.getTrackbarPos("lower - S", "Trackbar")
    lower_v = cv2.getTrackbarPos("lower - V", "Trackbar")

    upper_h = cv2.getTrackbarPos("upper - H", "Trackbar")
    upper_s = cv2.getTrackbarPos("upper - S", "Trackbar")
    upper_v = cv2.getTrackbarPos("upper - V", "Trackbar")

    lower_color = np.array([lower_h, lower_s, lower_v])
    upper_color = np.array([upper_h, upper_s, upper_v])

    mask = cv2.inRange(frame_hsv, lower_color, upper_color)

    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
