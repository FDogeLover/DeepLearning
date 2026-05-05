import os
import ndjson
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from model3 import ImageFeatureExtractor

# 设置参数
DATA_DIR = "ndjson"  # 存放 .ndjson 文件的文件夹
MAX_PER_CLASS = 5000  # 每类最多抽取多少图像
MODEL_PATH = "model/model.pth"
OUTPUT_PATH = "feature/feature_db.pt"
IMG_SIZE = 64

device = torch.device("cpu")

# 加载模型
model = ImageFeatureExtractor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def draw_strokes_to_image(drawing, size=IMG_SIZE, stroke_width=3):
    img = Image.new("L", (256, 256), color=255)
    draw = ImageDraw.Draw(img)
    for stroke in drawing:
        x, y = stroke
        points = list(zip(x, y))
        if len(points) > 1:
            draw.line(points, fill=0, width=stroke_width)
    img = img.resize((size, size)).convert("L")
    return img

def preprocess(img):
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    return tensor

features = []
labels = []
drawings = []

# 获取所有 .ndjson 文件
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".ndjson")]

print(f"📁 共找到 {len(files)} 个类的 .ndjson 文件")

# 处理每个类别
for filename in tqdm(files, desc="提取所有特征"):
    class_name = filename.replace(".ndjson", "")
    path = os.path.join(DATA_DIR, filename)

    # 读取当前类的 .ndjson 文件
    with open(path, "r") as f:
        data = ndjson.load(f)  # 使用 ndjson.load() 代替 ndjson.reader()

    count = 0
    # 遍历当前类中的每个样本
    for item in data:
        drawing = item["drawing"]
        img = draw_strokes_to_image(drawing)
        tensor = preprocess(img)

        with torch.no_grad():
            # 提取特征并进行归一化
            feature = model(tensor).squeeze(0)
            feature = F.normalize(feature.unsqueeze(0), dim=1).squeeze(0)

        features.append(feature.cpu())
        labels.append(class_name)
        drawings.append(drawing)

        count += 1
        if count >= MAX_PER_CLASS:
            break

# 汇总保存
features_tensor = torch.stack(features)
torch.save({
    "features": features_tensor,         # [N, 128]
    "labels": labels,                    # [N]
    "drawings": drawings                 # 原始 strokes，用于可视化
}, OUTPUT_PATH)

print(f"\n✅ 特征提取完毕，共 {len(labels)} 张图像，已保存为 {OUTPUT_PATH}")
