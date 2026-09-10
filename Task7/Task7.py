import cv2
import numpy

img = cv2.imread(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task7\birtl.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
sure_fg = numpy.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers_ws = cv2.watershed(img, markers)

img[markers_ws == -1] = [0, 0, 255]

label_ids = numpy.unique(markers_ws)
label_ids = label_ids[label_ids > 1]
coin_count = len(label_ids[label_ids > 1])

cv2.imshow("Deneme", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
