import cv2
import numpy as np


img = cv2.imread(r'C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task8\chess.jpg')
img = cv2.resize(img, (750,750))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Deneme', img)
cv2.waitKey(0)
cv2.destroyAllWindows()