import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from QuickDrwa_CNN import QuickDrawCNN  # 自定义模型结构

# 类别配置
categories = ['apple', 'airplane', 'bed', 'bread', 'banana', 'book']
label_to_file = {
    'apple': 'similarity_row/full_numpy_bitmap_apple.npy',
    'airplane': 'similarity_row/full_numpy_bitmap_airplane.npy',
    'bed': 'similarity_row/full_numpy_bitmap_bed.npy',
    'bread': 'similarity_row/full_numpy_bitmap_bread.npy',
    'banana': 'similarity_row/full_numpy_bitmap_banana.npy',
    'book': 'similarity_row/full_numpy_bitmap_book.npy'
}

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuickDrawCNN(num_classes=len(categories)).to(device)
model = torch.load("model/similarity_model.pkl", map_location=device,weights_only=False)
model.eval()

# 画布参数
WIDTH, HEIGHT = 560, 560
black = (0, 0, 0)
default_pen_size = 16

# 初始化窗口
root = tk.Tk()
root.title("🎨 QuickDraw 相似度比较系统")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='black')
canvas.pack()

# 图像和画笔控制
image = Image.new("RGB", (WIDTH, HEIGHT), black)
draw = ImageDraw.Draw(image)
pen_size = tk.IntVar(value=default_pen_size)
draw_history = []

# 当前状态
selected_category = None
reference_image = None
last_x, last_y = None, None

# 绘图函数
def draw_callback(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x is not None and last_y is not None:
        canvas.create_line(last_x, last_y, x, y, width=pen_size.get(), fill='white', capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill='white', width=pen_size.get())
        draw_history.append(((last_x, last_y), (x, y)))  # 存储绘图历史
    last_x, last_y = x, y

def reset_coords(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", draw_callback)
canvas.bind("<ButtonRelease-1>", reset_coords)

# 工具函数
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=black)
    draw_history.clear()

def undo_last():
    if draw_history:
        clear_canvas()
        for (x0, y0), (x1, y1) in draw_history[:-1]:
            canvas.create_line(x0, y0, x1, y1, width=pen_size.get(), fill='white', capstyle=tk.ROUND, smooth=True)
            draw.line([x0, y0, x1, y1], fill='white', width=pen_size.get())
        draw_history.pop()

def save_image():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 文件", "*.png")])
    if file_path:
        image.save(file_path)

def preprocess(img):
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(img.convert("L"))
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor).unsqueeze(0)
    return tensor.to(device)

def compute_similarity(vec1, vec2):
    vec1 = vec1 / vec1.norm(dim=1, keepdim=True)
    vec2 = vec2 / vec2.norm(dim=1, keepdim=True)
    return torch.sum(vec1 * vec2).item()

def on_category_selected(category):
    global selected_category, reference_image
    selected_category = category
    data = np.load(label_to_file[category])
    idx = random.randint(0, len(data) - 1)
    reference_image = Image.fromarray(data[idx].reshape(28, 28)).convert("L")
    result_label.config(text=f"🎯 已选类别: {category}，请在画布中绘图")

def predict_similarity():
    if not selected_category:
        result_label.config(text="⚠️ 请先选择一个类别")
        return
    drawn_tensor = preprocess(image)
    ref_tensor = preprocess(reference_image)

    with torch.no_grad():
        feat1 = model(drawn_tensor, return_features=True)
        feat2 = model(ref_tensor, return_features=True)

    similarity = compute_similarity(feat1, feat2)
    result_label.config(text=f"✅ 类别: {selected_category} | 相似度: {similarity:.4f}")
    visualize_comparison(reference_image, image, similarity)

def visualize_comparison(ref_img, drawn_img, similarity):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(ref_img, cmap='gray')
    axs[0].set_title("参考图")
    axs[1].imshow(drawn_img.resize((28, 28)).convert("L"), cmap='gray')
    axs[1].set_title("你的图")
    for ax in axs:
        ax.axis("off")
    plt.suptitle(f"相似度: {similarity:.4f}")
    plt.tight_layout()
    plt.show()

# 类别选择区域
category_frame = tk.Frame(root)
category_frame.pack(pady=5)
tk.Label(category_frame, text="选择类别:").pack(side=tk.LEFT)
for cat in categories:
    tk.Button(category_frame, text=cat, command=lambda c=cat: on_category_selected(c)).pack(side=tk.LEFT, padx=2)

# 按钮区域
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="清空画布", command=clear_canvas).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="撤销", command=undo_last).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="保存图像", command=save_image).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="比较相似度", command=predict_similarity).pack(side=tk.LEFT, padx=5)

# 画笔大小调节
pen_frame = tk.Frame(root)
pen_frame.pack()
tk.Label(pen_frame, text="笔粗:").pack(side=tk.LEFT)
tk.Scale(pen_frame, from_=4, to=32, orient=tk.HORIZONTAL, variable=pen_size).pack(side=tk.LEFT)

# 显示标签
result_label = tk.Label(root, text="请选择类别并绘图")
result_label.pack(pady=5)

root.mainloop()
