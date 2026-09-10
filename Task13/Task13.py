import cv2

kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = kamera.read()
    if not ret:
        print("Kamera görüntüsü yok")
        break
    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
