# build_feature_db.py

import os
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np

from model2 import ImageFeatureExtractor

# ⚙️ 配置
NDJSON_DIR = "ndjson"               # 存放 .ndjson 文件的目录
OUT_PATH = "feature/features.pt"  # 输出特征库路径
DEVICE = 'cpu'
IMG_SIZE = 64                     # 模型输入尺寸
MAX_SAMPLES_PER_CLASS = 2000      # 每类抽取图像数

def render_strokes(drawing, size=IMG_SIZE, stroke_width=3):
    img = Image.new("L", (256, 256), color=255)
    draw = ImageDraw.Draw(img)
    for stroke in drawing:
        x, y = stroke
        points = list(zip(x, y))
        if len(points) > 1:
            draw.line(points, fill=0, width=stroke_width)
    img = img.resize((size, size)).convert("L")
    return img

def load_drawings_from_ndjson(file_path, max_items=200):
    drawings = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break
            item = json.loads(line)
            drawings.append(item['drawing'])
    return drawings

def main():
    print("🚀 加载模型中...")
    model = ImageFeatureExtractor().to(DEVICE)
    model.eval()

    all_features = []
    all_labels = []
    all_drawings = []

    files = [f for f in os.listdir(NDJSON_DIR) if f.endswith(".ndjson")]
    print(f"📂 找到 {len(files)} 个类：{files}")

    for fname in tqdm(files, desc="提取特征"):
        label = os.path.splitext(fname)[0]
        path = os.path.join(NDJSON_DIR, fname)
        drawings = load_drawings_from_ndjson(path, MAX_SAMPLES_PER_CLASS)

        for d in drawings:
            img = render_strokes(d)
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]

            with torch.no_grad():
                feat = model(tensor).squeeze(0)  # [128]

            all_features.append(feat.cpu())
            all_labels.append(label)
            all_drawings.append(d)

    # 构建并保存数据库
    db = {
        "features": F.normalize(torch.stack(all_features), dim=1),  # [N,128]
        "labels": all_labels,
        "drawings": all_drawings
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(db, OUT_PATH)
    print(f"✅ 特征库已保存至: {OUT_PATH}")

if __name__ == "__main__":
    main()
