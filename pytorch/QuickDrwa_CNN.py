import torch.nn as nn

class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.feature_layer = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv_layers(x)
        x = self.flatten(x)
        features = self.feature_layer(x)
        if return_features:
            return features  # 返回用于相似度计算的特征
        out = self.classifier(features)
        return out  # 返回分类结果
