import torch.nn as nn

class QuickDrawCNN2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 64×64
            nn.ReLU(),
            nn.MaxPool2d(2),                # 32×32

            nn.Conv2d(32, 64, 3, padding=1), # 32×32
            nn.ReLU(),
            nn.MaxPool2d(2),                # 16×16

            nn.Conv2d(64, 128, 3, padding=1), # 16×16
            nn.ReLU(),
            nn.MaxPool2d(2),                # 8×8
        )

        self.flatten = nn.Flatten()
        self.feature_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv_layers(x)
        x = self.flatten(x)
        features = self.feature_layer(x)
        if return_features:
            return features
        return self.classifier(features)
