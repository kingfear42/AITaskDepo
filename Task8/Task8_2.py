import cv2
import numpy as np

img = cv2.imread("Task8/tiger.jpg", 0)
img_blur = cv2.GaussianBlur(img, (3,3), sigmaX = 0, sigmaY= 0)

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5) 
sobelxy = cv2.magnitude(sobelx, sobely)

cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
 
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
 
cv2.imshow('Sobel gradient magnitude', cv2.normalize(sobelxy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U))
cv2.waitKey(0)
