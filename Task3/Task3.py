import cv2

img = cv2.imread(r'C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task3\IMG1.png', 1)
boyut = (768, 512)
resized_img = cv2.resize(img, boyut)
cv2.imshow("Masuaku daha kucuk", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Task3/IMG2.png', resized_img)