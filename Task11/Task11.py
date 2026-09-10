import cv2

img = cv2.imread(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task11\IMG1.png")
img = cv2.resize(img, (768, 512))

x_ekseni_flip = cv2.flip(img, 0)

y_ekseni_flip = cv2.flip(img, 1)

orjin_flip = cv2.flip(img, -1)

cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.imshow("X eksenine göre döndürülmüş", x_ekseni_flip)
cv2.waitKey(0)
cv2.imshow("Y eksenine göre döndürülmüş", y_ekseni_flip)
cv2.waitKey(0)
cv2.imshow("Orjine göre döndürülmüş", orjin_flip)
cv2.waitKey(0)

cv2.destroyAllWindows()
