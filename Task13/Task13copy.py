import cv2
import numpy as np

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  if not ret:
    print('Kameradan görüntü alınamadı!')
    break

  # Görüntüyü HSV formatına çevir
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # SIYAH RENK ARALIĞI (Işığın en az olduğu, en koyu tonlar)
  # H (0-180), S (0-255), V (0-50 -> Sadece karanlık/parlaklığı az yerler)
  lower_black = np.array([0, 0, 0])
  upper_black = np.array([180, 255, 50])

  # Siyah bölgeleri maskele (Siyah yerler beyaz, diğer yerler siyah olur)
  mask = cv2.inRange(hsv, lower_black, upper_black)

  # Gürültü temizleme (Gölgeleri ve küçük siyah noktaları elemek için önemli)
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # Beyaz alanların (siyah cismin) konturlarını bul
  contours, _ = cv2.findContours(
      mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  if len(contours) > 0:
    # Ekrandaki en büyük siyah alanı bul
    c = max(contours, key=cv2.contourArea)

    # Gölgeleri tamamen elemek için alanı biraz daha büyük tutuyoruz (Örn: 1500 piksel)
    if cv2.contourArea(c) > 1300:
      # Nesnenin koordinatlarını al
      x, y, w, h = cv2.boundingRect(c)

      # Canlı görüntü üzerine MAVİ bir dikdörtgen çiz (Siyah nesne üzerinde belli olsun diye)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 130, 255), 2)
      """
      # Merkez noktası çiz
      center_x = int(x + w / 2)
      center_y = int(y + h / 2)
      cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
      """

      cv2.putText(
          frame,
          'Siyah Nesne',
          (x, y - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (100, 130, 140),
          2,
      )

  # Görüntüleri ekranda göster
  cv2.imshow('Siyah Nesne Takibi', frame)
  cv2.imshow('Siyah Maskesi', mask)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
