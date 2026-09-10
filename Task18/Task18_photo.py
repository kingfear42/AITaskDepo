import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet(r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task18\yolov4.cfg",
                                 r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task18\yolov4.weights")
output_layer_names = net.getUnconnectedOutLayersNames()

image_path = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task18\Modric.jpg"
original_img = cv2.imread(image_path)

if original_img is None:
    print("Görsel okunamadı, dosya yolunu kontrol edin!")
    exit()

def update_detection(val):
    
    frame = original_img.copy()
    h, w, _ = frame.shape

    conf_percent = cv2.getTrackbarPos("Confidence Threshold (%)", "İnsan Tespiti")
    conf_threshold = conf_percent / 100.0

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer_names)

    boxes, confidences = [], []

    for out in outs:
        for detection in out:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if class_id == 0 and confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w_box, h_box = boxes[i]
            conf = confidences[i]
            
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            cv2.putText(frame, f"Person %{int(conf * 100)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Aktif Conf Esigi: %{conf_percent}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("İnsan Tespiti", frame)

cv2.namedWindow("İnsan Tespiti")
cv2.createTrackbar("Confidence Threshold (%)", "İnsan Tespiti", 50, 100, update_detection)

update_detection(50)
cv2.waitKey(0)
cv2.destroyAllWindows()