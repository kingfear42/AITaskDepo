import cv2

img = cv2.imread(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task6\Image.jpg", 1)
img = cv2.resize(img, (460, 569))
_, thresh_simple = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Deneme", thresh_simple)
print("Org Resim: ", img.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()