import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task17\yolov3-face.cfg",
                                 r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task17\yolov3-wider_16000.weights")
output_layer_names = net.getUnconnectedOutLayersNames()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Görüntünün yükseklik (h) ve genişlik (w) değerlerini alıyoruz
    h, w = frame.shape[:2]

    # 3. Kareyi Modele Uygun Formata Getirme (Blob Oluşturma)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Modele görüntüyü verip tahmin çıktılarını (outs) alıyoruz
    outs = net.forward(output_layer_names)

    boxes = []
    confidences = []
    conf_threshold = 0.5  # %50 güven eşiği

    # 4. Çıktıları İşleme ve Filtreleme
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Güven skoru belirlediğimiz eşikten yüksekse koordinatları hesapla
            if confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)

                # Sol üst köşe koordinatlarını bul
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))

    # 5. Üst Üste Binen Kutuları Temizleme (Non-Maxima Suppression)
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # 6. "Yüz Var / Yok" Kontrolü ve Ekrana Çizdirme
    if len(indices) > 0:
        # Tespit edilen her yüz için yeşil kutu çizdirme
        for i in indices.flatten():
            x, y, box_w, box_h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (130, 120, 110), 2)
    # Sonucu ekranda göster
    cv2.imshow("Yüz Tespiti", frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
