import cv2
import numpy as np
import time

net = cv2.dnn.readNetFromDarknet(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task18\yolov4.cfg",
                                r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task18\yolov4.weights")
output_layer_names = net.getUnconnectedOutLayersNames()

prev_time = 0

def nothing(x):
    pass

cv2.namedWindow("İnsan Tespiti")
cv2.createTrackbar("Confidence Treshold (%)", "İnsan Tespiti", 50, 100, nothing)

cap = cv2.VideoCapture(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task18\Germany - Argentina World Cup 2014 final _ Highlights _ 4K UHD 60 fps - YouTube - Opera 2026-07-03 01-43-00.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Videoya ulaşılamamıştır")
        break
    h, w, _ = frame.shape

    conf_percent = cv2.getTrackbarPos("Confidence Treshold (%)", "İnsan Tespiti")
    conf_treshold = conf_percent / 100.0

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    boxes, confidences = [], []

    for out in outs:
        for detection in out:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if class_id == 0 and confidence > conf_treshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_treshold, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            # Değişken atamaları düzeltildi
            x, y, w_box, h_box = boxes[i]
            conf = confidences[i]
            
            # Kutuyu çiz ve üzerine etiketi ekle
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, f"Person %{int(conf * 100)}", (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f"Anlik FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    

    cv2.imshow("YOLOv4 İnsan Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()