import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from QuickDrwa_CNN2 import QuickDrawCNN2  # 确保模型定义在这个模块中
from sklearn.model_selection import train_test_split

from pytorch.simliarity_test import label_map, category_names

# 设置参数
BATCH_SIZE = 64
EPOCHS = 25
NUM_CLASSES = 2  # 根据你的类别数量调整
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
#category_names = ['bed', 'apple', 'banana', 'airplane', 'book', 'bread']
# category_paths = [f"similarity_row/full_numpy_bitmap_{name}.ndjson" for name in category_names]
# label_map = {name: i for i, name in enumerate(category_names)}
category_names=["apple","banana"]
category_paths = [f"converted_npy/full_resized_{name}.npy" for name in category_names]
label_map = {"apple":0,"banana": 1}

# 自定义数据集类
class QuickDrawDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.astype(np.uint8)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(64, 64).astype(np.uint8)
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# 加载所有数据
def load_data():
    all_data = []
    all_labels = []
    for path, name in zip(category_paths, category_names):
        print(f"Loading: {path}")
        data = np.load(path)
        data = data[:3000]  # 每类取部分数据
        labels = np.full(len(data), label_map[name])
        all_data.append(data)
        all_labels.append(labels)
    return np.vstack(all_data), np.hstack(all_labels)


# 数据加载与划分
raw_data, raw_labels = load_data()
train_data, val_data, train_labels, val_labels = train_test_split(
    raw_data, raw_labels, test_size=0.2, random_state=42, stratify=raw_labels
)

# 数据变换
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 数据集与加载器
train_dataset = QuickDrawDataset(train_data, train_labels, transform=transform)
val_dataset = QuickDrawDataset(val_data, val_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型定义
model = QuickDrawCNN2(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train():
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

        evaluate()

    # 保存模型
    torch.save(model.state_dict(), "model/similarity_model_weights2.pth")
    torch.save(model, "model/similarity_model2.pkl")


# 验证函数
def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
    acc = correct / len(val_loader.dataset)
    print(f"Validation Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train()
