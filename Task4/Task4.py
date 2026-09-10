import cv2

img = cv2.imread(r'C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task4\IMG2.png', 1)
designed_img = img.copy()
designed_img2 = img.copy()
cv2.rectangle(designed_img, (390,30),(490,190),(0,255,255),2)
cv2.rectangle(designed_img2, (390,30),(490,190),(0,255,255),-1)
cv2.imshow("Masuaku daha kucuk", designed_img)
cv2.imshow("Masuaku daha kucuk ve sansurlu", designed_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('Task4/IMG2.png', resized_img)