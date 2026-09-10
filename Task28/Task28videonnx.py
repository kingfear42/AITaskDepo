import os
import subprocess

# 1. Klasör ve Dosya Yolları
base_dir = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task20"

# NOT: Task28 klasöründeki ONNX model yolu düzeltildi
weights_path = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task28\best.onnx"
source_path = os.path.join(base_dir, "Roadcam.mp4")
yolov5_dir = os.path.join(base_dir, "yolov5-master")

# 2. YOLOv5 Predict Komutu
command = [
    "python",
    "segment/predict.py",
    "--weights",
    weights_path,
    "--source",
    source_path,
    "--img",
    "640",
    "--conf-thres",
    "0.07",
    "--view-img",
]

print("Video işleniyor... Terminalden FPS ve kare sürelerini takip edebilirsiniz.")

# 3. Komutu yolov5-master Klasöründe Çalıştır
subprocess.run(command, cwd=yolov5_dir)