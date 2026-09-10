import json
import os
import glob
from PIL import Image

label_json = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task20\label.json"
img_dir = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task20\dataset\images"
lbl_dir = r"C:\Users\krkmz\Desktop\Aware Robotics Opencv\Task20\dataset\labels"

# Cache dosyalarını sil
for cache in glob.glob(os.path.join(lbl_dir, "*/*.cache*")):
    try: os.remove(cache)
    except: pass

with open(label_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

for key, value in data.items():
    img_name = value['filename']
    base_name = os.path.splitext(img_name)[0]
    txt_name = f"{base_name}.txt"
    
    sub_folder = "train"
    img_path = os.path.join(img_dir, "train", img_name)
    if not os.path.exists(img_path):
        img_path = os.path.join(img_dir, "val", img_name)
        sub_folder = "val"
        
    if not os.path.exists(img_path):
        continue
        
    with Image.open(img_path) as img:
        w, h = img.size

    target_txt_path = os.path.join(lbl_dir, sub_folder, txt_name)
    
    regions = value.get('regions', [])
    region_list = regions.values() if isinstance(regions, dict) else regions

    lines = []
    for region in region_list:
        shape_attr = region.get('shape_attributes', {})
        if shape_attr.get('name') == 'polygon':
            px = shape_attr.get('all_points_x', [])
            py = shape_attr.get('all_points_y', [])

            class_id = 0  # asfalt_yol
            
            coords = []
            for x, y in zip(px, py):
                # Değerleri 0.001 - 0.999 arasında sınırla ki IndexError vermesin
                norm_x = max(0.001, min(0.999, float(x) / w))
                norm_y = max(0.001, min(0.999, float(y) / h))
                coords.append(f"{norm_x:.6f} {norm_y:.6f}")

            if len(coords) >= 3:
                lines.append(f"{class_id} " + " ".join(coords) + "\n")

    with open(target_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.writelines(lines)

print("Etiketler güvenli sınırlar (0.001 - 0.999) ile tekrar üretildi!")