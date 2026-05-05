import os
import torch
import numpy as np
from model import ImageFeatureExtractor
from PIL import Image, ImageDraw

def draw_strokes_to_image(drawing, size=64, stroke_width=3):
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
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor  # [1, 1, 64, 64]

def main():
    device = 'cpu'
    model = ImageFeatureExtractor().to(device)
    model.eval()

    feature_dir = "feature_by_class"
    os.makedirs(feature_dir, exist_ok=True)

    # 加载 QuickDraw 数据（预处理好的）
    data = torch.load("feature/features.pt")
    labels = data['labels']
    drawings = data['drawings']

    # 按类别分组
    label_to_features = {}
    for label, drawing in zip(labels, drawings):
        img = draw_strokes_to_image(drawing)
        x = preprocess(img).to(device)
        with torch.no_grad():
            feature = model(x).squeeze(0)
            feature = feature / feature.norm()
        label_to_features.setdefault(label, []).append(feature)

    # 每类计算平均特征
    class_names = []
    class_features = []
    for label, features in label_to_features.items():
        center = torch.stack(features).mean(dim=0)
        center = center / center.norm()
        class_names.append(label)
        class_features.append(center)

    # 保存中心库
    torch.save({
        'class_names': class_names,
        'class_features': torch.stack(class_features)
    }, os.path.join(feature_dir, "class_centers.pt"))
    print(f"✅ 已保存 {len(class_names)} 类的中心特征")

if __name__ == "__main__":
    main()
