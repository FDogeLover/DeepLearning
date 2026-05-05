import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn.functional as F
import numpy as np
from model3 import ImageFeatureExtractor

CANVAS_SIZE = 256
IMG_SIZE = 64
STROKE_WIDTH = 8

class DrawingApp:
    def __init__(self, root, feature_center_path="feature_by_class/class_centers.pt", model_path="model/model.pth"):
        self.root = root
        self.root.title("按类别检索最相似图像")

        # 创建画布
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10, rowspan=2)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)

        # 控制按钮
        tk.Button(root, text="清空", command=self.clear).grid(row=2, column=0)
        tk.Button(root, text="检索类别", command=self.search).grid(row=2, column=1)

        # 结果文本框
        self.result_text = tk.Text(root, height=10, width=40)
        self.result_text.grid(row=0, column=1)

        # 模型和特征库加载
        self.device = 'cpu'
        self.model = ImageFeatureExtractor().to(self.device)

        # ✅ 加载训练好的模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print("✅ 模型权重加载成功")

        # 加载每类的特征中心
        data = torch.load(feature_center_path)
        self.class_names = data['class_names']
        self.class_features = data['class_features']  # shape: [N_class, 128]
        self.class_features = F.normalize(self.class_features, dim=1)  # 归一化

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_text.delete(1.0, tk.END)

    def paint(self, event):
        x, y = event.x, event.y
        r = STROKE_WIDTH // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def preprocess(self):
        img = self.image.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(self.device)  # shape: [1,1,64,64]
        return tensor

    def search(self):
        self.result_text.delete(1.0, tk.END)
        with torch.no_grad():
            img_tensor = self.preprocess()
            feature = self.model(img_tensor).squeeze(0)  # shape: [128]
            feature = F.normalize(feature.unsqueeze(0), dim=1)  # shape: [1,128]

            sims = torch.mm(feature, self.class_features.T).squeeze(0)  # shape: [N_class]
            topk_scores, topk_indices = torch.topk(sims, k=5)

        self.result_text.insert(tk.END, "🔍 最相似的类别 Top 5：\n\n")
        for i in range(5):
            name = self.class_names[topk_indices[i]]
            score = topk_scores[i].item()
            self.result_text.insert(tk.END, f"{i+1}. {name:<15} 相似度: {score:.4f}\n")

# 启动主界面
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root, feature_center_path="feature_by_class/class_centers.pt", model_path="model/model.pth")
    root.mainloop()
