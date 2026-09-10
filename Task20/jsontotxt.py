import json
import cv2
import os

# 1. JSON dosyasının ve resimlerin bulunduğu klasör
json_file_path = 'Task20/label.json'  # MakeSense'den indirdiğiniz json adı
images_dir = '.'                # Resimlerin bulunduğu klasör (aynı yerdeyse '.')

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for key, value in data.items():
    img_name = value['filename']
    img_path = os.path.join(images_dir, img_name)
    
    # Resim boyutlarını al (Normalize etmek için şart)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Uyarı: {img_name} bulunamadı, atlanıyor...")
        continue
    
    h, w, _ = img.shape
    txt_name = os.path.splitext(img_name)[0] + ".txt"

    with open(txt_name, "w", encoding='utf-8') as txt_file:
        regions = value.get('regions', [])
        
        # 'regions' liste veya dict olabilir, ikisini de destekle:
        region_list = regions.values() if isinstance(regions, dict) else regions

        for region in region_list:
            shape_attr = region.get('shape_attributes', {})
            
            if shape_attr.get('name') == 'polygon':
                px = shape_attr.get('all_points_x', [])
                py = shape_attr.get('all_points_y', [])

                class_id = 0  # Sınıf ID'niz (asfalt_yol için 0)
                
                # Koordinatları 0-1 arasında normalize et
                normalized_coords = []
                for x, y in zip(px, py):
                    norm_x = x / w
                    norm_y = y / h
                    normalized_coords.append(f"{norm_x:.6f} {norm_y:.6f}")

                # YOLO Segmentation formatında yaz: <class_id> <x1> <y1> <x2> <y2> ...
                line = f"{class_id} " + " ".join(normalized_coords) + "\n"
                txt_file.write(line)

print("Dönüştürme tamamlandı! .txt dosyaları oluşturuldu.")