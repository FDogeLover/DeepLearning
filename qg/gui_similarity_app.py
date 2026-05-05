import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn.functional as F
import numpy as np
from model3 import ImageFeatureExtractor

CANVAS_SIZE = 256
IMG_SIZE = 64
STROKE_WIDTH = 8
TOPK = 5  # 显示前五相似图

# class DrawingApp:
#     def __init__(self, root, model_path="model.pth", feature_db_path="feature_db.pt"):
#         self.root = root
#         self.root.title("手绘图相似度识别")
#
#         # UI 布局
#         self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
#         self.canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
#         self.canvas.bind("<B1-Motion>", self.paint)
#
#         self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
#         self.draw = ImageDraw.Draw(self.image)
#
#         tk.Button(root, text="清空", command=self.clear).grid(row=2, column=0)
#         tk.Button(root, text="开始检索", command=self.search).grid(row=2, column=1)
#
#         self.result_text = tk.Text(root, height=10, width=40)
#         self.result_text.grid(row=0, column=1, sticky="nw")
#
#         self.sim_imgs = [tk.Label(root) for _ in range(TOPK)]
#         for i, label in enumerate(self.sim_imgs):
#             label.grid(row=1, column=1 + i)
#
#         # 加载模型和特征库
#         self.device = torch.device('cpu')
#         self.model = ImageFeatureExtractor().to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#
#         db = torch.load(feature_db_path)
#         self.db_features = F.normalize(db['features'], dim=1)  # [N, 128]
#         self.db_labels = db['labels']  # [N]
#         self.db_drawings = db['drawings']  # 原始ndjson格式 stroke 列表
#
#         print(f"✅ 特征库加载完成，图像数: {len(self.db_labels)}")
#
#     def clear(self):
#         self.canvas.delete("all")
#         self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
#         self.draw = ImageDraw.Draw(self.image)
#         self.result_text.delete(1.0, tk.END)
#         for label in self.sim_imgs:
#             label.config(image='')
#             label.image = None
#
#     def paint(self, event):
#         x, y = event.x, event.y
#         r = STROKE_WIDTH // 2
#         self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
#         self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")
#
#     def preprocess(self):
#         img = self.image.resize((IMG_SIZE, IMG_SIZE))
#         arr = np.array(img).astype(np.float32) / 255.0
# #        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,64,64]
# #        return tensor
#         # 背景抑制：将背景（白色）权重降低
#         # 方法：把像素值乘以 (1 - arr)，白色越多，值越趋近于0
#         weighted_arr = arr * (1.0 - arr)
#         tensor = torch.tensor(weighted_arr).unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,H,W]
#         return tensor
#
#     def stroke_to_image(self, drawing, size=64):
#         img = Image.new("L", (256, 256), color=255)
#         draw = ImageDraw.Draw(img)
#         for stroke in drawing:
#             x, y = stroke
#             points = list(zip(x, y))
#             if len(points) > 1:
#                 draw.line(points, fill=0, width=3)
#         img = img.resize((size, size), Image.NEAREST)
#         return img
#
#     def search(self):
#         self.result_text.delete(1.0, tk.END)
#
#         with torch.no_grad():
#             img_tensor = self.preprocess()
#             feature = self.model(img_tensor).squeeze(0)  # [128]
#             feature = F.normalize(feature.unsqueeze(0), dim=1)  # [1,128]
#
#             sims = torch.mm(feature, self.db_features.T).squeeze(0)  # [N]
#             topk_scores, topk_indices = torch.topk(sims, TOPK)
#
#         self.result_text.insert(tk.END, "🔍 Top 5 相似图：\n\n")
#         for i in range(TOPK):
#             idx = topk_indices[i].item()
#             label = self.db_labels[idx]
#             score = topk_scores[i].item()
#             self.result_text.insert(tk.END, f"{i+1}. 类别: {label:<15} 相似度: {score:.4f}\n")
#
#             drawing = self.db_drawings[idx]
#             img = self.stroke_to_image(drawing, size=64)
#             tk_img = ImageTk.PhotoImage(img)
#             self.sim_imgs[i].config(image=tk_img)
#             self.sim_imgs[i].image = tk_img  # 防止图像被回收

# 运行 GUI 应用

class DrawingApp:
    def __init__(self, root, model_path="model.pth", feature_db_path="feature_db.pt"):
        self.root = root
        self.root.title("手绘图相似度识别")

        # UI 布局
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)

        tk.Button(root, text="清空", command=self.clear).grid(row=2, column=0)
        tk.Button(root, text="开始检索", command=self.search).grid(row=2, column=1)

        self.result_text = tk.Text(root, height=10, width=40)
        self.result_text.grid(row=0, column=1, sticky="nw")

        self.sim_imgs = [tk.Label(root) for _ in range(TOPK)]
        for i, label in enumerate(self.sim_imgs):
            label.grid(row=1, column=1 + i)

        # 加载模型和特征库
        self.device = torch.device('cpu')
        self.model = ImageFeatureExtractor().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        db = torch.load(feature_db_path)
        self.db_features = F.normalize(db['features'], dim=1)  # [N, 128]
        self.db_labels = db['labels']  # [N]
        self.db_drawings = db['drawings']  # 原始ndjson格式 stroke 列表

        print(f"✅ 特征库加载完成，图像数: {len(self.db_labels)}")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_text.delete(1.0, tk.END)
        for label in self.sim_imgs:
            label.config(image='')
            label.image = None

    def paint(self, event):
        x, y = event.x, event.y
        r = STROKE_WIDTH // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def is_blank_image(self, img, threshold=250):
        arr = np.array(img)
        return np.mean(arr) > threshold

    # def preprocess(self):
    #     img = self.image.resize((IMG_SIZE, IMG_SIZE))
    #     arr = np.array(img).astype(np.float32) / 255.0
    #     weighted_arr = arr * (1.0 - arr)
    #     tensor = torch.tensor(weighted_arr).unsqueeze(0).unsqueeze(0).to(self.device)
    #     return tensor

    def preprocess(self, method='auto_weighted_sim', alpha=2.0):
        """
        method: 预处理方法，可选：
            - 'raw'：原始图像
            - 'weight'：背景加权抑制，arr * (1 - arr)^α
            - 'mask'：白色背景屏蔽
            - 'auto_weighted_sim'：返回图像强度用于后续相似度调整
        alpha: 背景抑制指数（适用于 'weight' 模式）
        """
        img = self.image.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0  # [0,1]

        if method == 'raw':
            processed = arr

        elif method == 'weight':
            processed = arr * ((1.0 - arr) ** alpha)

        elif method == 'mask':
            mask = (arr < 0.95).astype(np.float32)
            processed = arr * mask

        elif method == 'auto_weighted_sim':
            self.img_strength = arr.mean()
            processed = arr

        else:
            raise ValueError(f"未知预处理方法: {method}")

        tensor = torch.tensor(processed).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    def stroke_to_image(self, drawing, size=64):
        img = Image.new("L", (256, 256), color=255)
        draw = ImageDraw.Draw(img)
        for stroke in drawing:
            x, y = stroke
            points = list(zip(x, y))
            if len(points) > 1:
                draw.line(points, fill=0, width=3)
        img = img.resize((size, size), Image.NEAREST)
        return img

    def search(self):
        self.result_text.delete(1.0, tk.END)

        # 检查是否空白
        if self.is_blank_image(self.image):
            self.result_text.insert(tk.END, "⚠️ 当前绘图为空白，请画一些内容再检索。\n")
            return

        with torch.no_grad():
            img_tensor = self.preprocess()
            feature = self.model(img_tensor).squeeze(0)  # [128]

            if feature.norm().item() < 1e-3:
                self.result_text.insert(tk.END, "⚠️ 绘图内容过少，相似度无参考意义。\n")
                return

            feature = F.normalize(feature.unsqueeze(0), dim=1)  # [1,128]
            sims = torch.mm(feature, self.db_features.T).squeeze(0)  # [N]
            sims *= self.img_strength  # 强度越小，图像越“空”，降低影响
            topk_scores, topk_indices = torch.topk(sims, TOPK)

        self.result_text.insert(tk.END, "🔍 Top 5 相似图：\n\n")
        for i in range(TOPK):
            idx = topk_indices[i].item()
            label = self.db_labels[idx]
            score = topk_scores[i].item()
            self.result_text.insert(tk.END, f"{i+1}. 类别: {label:<15} 相似度: {score:.4f}\n")

            drawing = self.db_drawings[idx]
            img = self.stroke_to_image(drawing, size=64)
            tk_img = ImageTk.PhotoImage(img)
            self.sim_imgs[i].config(image=tk_img)
            self.sim_imgs[i].image = tk_img

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root, model_path="model/model.pth", feature_db_path="feature/feature_db.pt")
    root.mainloop()
