import cv2

img = cv2.imread(r'C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task2\IMG1.png', 0)
cv2.imshow("Masuaku", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Task2/IMG2.png', img)