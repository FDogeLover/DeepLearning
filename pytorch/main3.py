import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from QuickDrwa_CNN import QuickDrawCNN  # 模型定义

# 设置类别
# categories = ['apple', 'airplane','bed','bread','banana','book']
categories = ['apple','banana']
label_to_file = {
    'apple': 'converted_npy/full_resized_apple.npy',
    # 'airplane': 'similarity_row/full_numpy_bitmap_airplane.npy',
    # 'bed': 'similarity_row/full_numpy_bitmap_bed.npy',
    # 'bread': 'similarity_row/full_numpy_bitmap_bread.npy',
    'banana': 'converted_npy/full_resized_banana.npy',
    # 'book': 'similarity_row/full_numpy_bitmap_book.npy'
}

# 设备与模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model/similarity_model2.pkl", map_location=device, weights_only=False)
model = model.to(device)
model.eval()

# GUI设置
WIDTH, HEIGHT = 640, 640
black = (0, 0, 0)  # 黑色背景
root = tk.Tk()
root.title("QuickDraw - 相似度比较")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='black')  # 画布背景设置为黑色
canvas.pack()
image = Image.new("RGB", (WIDTH, HEIGHT), black)  # 使用黑色背景创建图像
draw = ImageDraw.Draw(image)

# 随机选图
selected_category = random.choice(categories)
dataset = np.load(label_to_file[selected_category])
idx = random.randint(0, len(dataset) - 1)
reference_image_np = dataset[idx].reshape(64, 64)
reference_image = Image.fromarray(reference_image_np).convert("L")

def draw_callback(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='white', outline='white')  # 使用白色笔绘制
    draw.ellipse([x - r, y - r, x + r, y + r], fill='white')  # 使用白色笔绘制

canvas.bind("<B1-Motion>", draw_callback)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=black)  # 清空画布，恢复黑色背景

def preprocess(img):
    img = img.convert("L").resize((64, 64), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor).unsqueeze(0)
    return tensor.to(device)


def compute_similarity(vec1, vec2):
    vec1 = vec1 / vec1.norm(dim=1, keepdim=True)
    vec2 = vec2 / vec2.norm(dim=1, keepdim=True)
    return torch.sum(vec1 * vec2).item()

def predict_similarity():
    drawn_tensor = preprocess(image)
    ref_tensor = preprocess(reference_image)

    # 获取中间层特征向量
    with torch.no_grad():
        feat1 = model(drawn_tensor, return_features=True)
        feat2 = model(ref_tensor, return_features=True)

    similarity = compute_similarity(feat1, feat2)

    result_label.config(text=f"类别: {selected_category} | 相似度: {similarity:.4f}")
    visualize_comparison(reference_image, image, similarity)

def visualize_comparison(ref_img, drawn_img, similarity):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(ref_img, cmap='gray')
    axs[0].set_title("original picture")
    axs[1].imshow(drawn_img.resize((64, 64)).convert("L"), cmap='gray')
    axs[1].set_title("your picture")
    for ax in axs:
        ax.axis("off")
    plt.suptitle(f"similarity: {similarity:.4f}")
    plt.tight_layout()
    plt.show()

# 按钮
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="清空", command=clear_canvas).pack(side=tk.LEFT)
tk.Button(btn_frame, text="比较相似度", command=predict_similarity).pack(side=tk.LEFT)

result_label = tk.Label(root, text="请选择类别并绘制图像")
result_label.pack()

root.mainloop()
