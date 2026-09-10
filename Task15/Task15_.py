import cv2
import numpy as np

frame = cv2.imread(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task6\Image.jpg")
frame = cv2.resize(frame, (460, 569))

if frame is None:
    print("Resim dosyası bulunamadı veya açılamadı.")
    exit()

def nothing(x):
    pass
cv2.namedWindow("Trackbar")
cv2.resizeWindow("Trackbar", 450, 450)

cv2.createTrackbar("lower - H", "Trackbar", 0, 180, nothing)
cv2.createTrackbar("lower - S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("lower - V", "Trackbar", 0, 255, nothing)

cv2.createTrackbar("upper - H", "Trackbar", 0, 180, nothing)
cv2.createTrackbar("upper - S", "Trackbar", 0, 255, nothing)
cv2.createTrackbar("upper - V", "Trackbar", 0, 255, nothing)

cv2.setTrackbarPos("upper - H", "Trackbar", 180)
cv2.setTrackbarPos("upper - S", "Trackbar", 255)
cv2.setTrackbarPos("upper - V", "Trackbar", 255)

frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

while True:

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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()