import os
import subprocess

# 1. Klasör ve Dosya Yolları
base_dir = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task20"
weights_path = os.path.join(
    base_dir, r"yolov5-master\runs\train-seg\exp9\weights\best.pt"
)
source_path = os.path.join(base_dir, "Roadcam.mp4")
yolov5_dir = os.path.join(base_dir, "yolov5-master")

# 2. YOLOv5 Predict Komutu (--view-img ekranda açar)
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

print("Video işleniyor ve FPS ile ekranda gösteriliyor...")

# 3. Komutu yolov5-master Klasöründe Çalıştır
subprocess.run(command, cwd=yolov5_dir)
