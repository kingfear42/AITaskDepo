import cv2

img = cv2.imread(r'C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task5\IMG2.png', 1)

output_img = img.copy()
x1, y1 = 388, 30
x2, y2 = 492, 190

area = img[y1:y2, x1:x2] 
#amaç sansürlemek istediğimiz kısmı bölmek rahat bir şekilde istediğimizi yapalım

area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY) #gri yaptım

area_gray_blurred = cv2.GaussianBlur(area_gray, (25,25), 0)
#parantez içindeki sayı ne kadar büyürse o kadar bulanık olur

area_bgr = cv2.cvtColor(area_gray_blurred, cv2.COLOR_GRAY2BGR)
#burada pixel formatını değiştirdiğimiz fotomuzun pixel formatını normale dönüştürüyoruzki tekrar eski fotoyla birleşsin

output_img[y1:y2, x1:x2] = area_bgr
#birleştirdim

cv2.imshow("Masu ama daha cok sansur", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("Task5/IMG3.png", output_img)
#cv2.imwrite('Task4/IMG2.png', resized_img)