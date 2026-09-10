import cv2
import numpy as np

img = cv2.imread(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task10\IMG.png")

kernel = np.ones((5,5), np.uint8)

dilation = cv2.dilate(img, kernel, iterations = 1) #genişletme

erosion = cv2.erode(img, kernel, iterations = 1) #aşındırma

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #önce aşındırma sonra genişletme
#daha çok gürültülü resimlerdeki önce gürültüleri temizleyip sonrasında genişletmek için kullanılır


cv2.imshow("original", img)
cv2.imshow("dilation", dilation)
cv2.imshow("erosion", erosion)
cv2.imshow("opening", opening)

cv2.waitKey(0)
cv2.windowsAllDestroy()