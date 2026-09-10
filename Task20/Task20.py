import subprocess
import os

# 1. Klasör ve dosya yolları
base_dir = r"C:/Users/krkmz/Desktop/Aware Robotics Opencv/Task20"
weights_path = os.path.join(base_dir, r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task20\yolov5-master\runs\train-seg\exp9\weights\best.pt")
source_path = os.path.join(base_dir, r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task20\deneme_resimleri\forexample6.png")
yolov5_dir = os.path.join(base_dir, "yolov5-master")

# 2. YOLOv5 Predict komutu
command = [
    "python", "segment/predict.py",
    "--weights", weights_path,
    "--source", source_path,
    "--img", "640",
    "--conf-thres", "0.05"
]

print("Inference başlatılıyor...")

# 3. Komutu yolov5-master klasöründe çalıştır
result = subprocess.run(command, cwd=yolov5_dir, capture_output=True, text=True)

# 4. Çıktıları ekrana yazdır
print(result.stdout)
if result.stderr:
    print("Hata/Uyarı Çıktısı:\n", result.stderr)