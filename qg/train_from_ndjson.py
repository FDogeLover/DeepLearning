import os
import json
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model3 import ImageFeatureExtractor  # 你的模型文件

# 超参数
IMG_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "ndjson"
MAX_SAMPLES_PER_CLASS = 5000
MODEL_SAVE_PATH = "model/model.pth"

# 将 strokes 转为图像
def draw_strokes_to_image(drawing, size=64, stroke_width=3):
    img = Image.new("L", (256, 256), color=255)
    draw = ImageDraw.Draw(img)
    for stroke in drawing:
        points = list(zip(stroke[0], stroke[1]))
        if len(points) > 1:
            draw.line(points, fill=0, width=stroke_width)
    img = img.resize((size, size)).convert("L")
    return np.array(img).astype(np.float32) / 255.0

# 自定义数据集
class QuickDrawNDJSONDataset(Dataset):
    def __init__(self, ndjson_dir, max_per_class=2000):
        self.samples = []
        self.labels = []
        self.class_names = []

        for idx, filename in enumerate(os.listdir(ndjson_dir)):
            if not filename.endswith(".ndjson"):
                continue
            label = filename[:-7]
            self.class_names.append(label)
            print(f"📂 读取类：{label}")
            path = os.path.join(ndjson_dir, filename)

            with open(path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_per_class:
                        break
                    data = json.loads(line)
                    drawing = data["drawing"]
                    img = draw_strokes_to_image(drawing)
                    self.samples.append(img)
                    self.labels.append(idx)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        x = torch.tensor(x).unsqueeze(0)  # [1, H, W]
        y = self.labels[idx]
        return x, y

# 训练过程
def train():
    dataset = QuickDrawNDJSONDataset(DATA_DIR, MAX_SAMPLES_PER_CLASS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ 数据量：{len(dataset)} 张图，共 {len(dataset.class_names)} 类")

    model = ImageFeatureExtractor().to(DEVICE)
    classifier = nn.Linear(128, len(dataset.class_names)).to(DEVICE)

    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        classifier.train()
        total_loss = 0

        for imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            features = model(imgs)
            logits = classifier(features)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📉 Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    # 保存提取器权重
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ 已保存特征提取器模型为 {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
