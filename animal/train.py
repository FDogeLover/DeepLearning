import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 参数设置
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "dataset_augmented"
MODEL_PATH = "animal_model.pth"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 均值
                         [0.229, 0.224, 0.225])  # ImageNet 标准差
])

# 加载数据集
train_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 加载预训练的 MobileNetV2 模型
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    epoch_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    avg_loss = epoch_loss / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

# 保存模型
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ 模型已保存到 {MODEL_PATH}")
