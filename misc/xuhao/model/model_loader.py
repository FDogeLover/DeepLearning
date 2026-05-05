import torch
import torchvision.models as models
import torchvision.transforms as transforms

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-1])  # 去掉最后的分类层
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # 添加 batch 维度
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().cpu()
