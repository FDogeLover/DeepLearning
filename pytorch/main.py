import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
from torchvision import transforms
from QuickDrwa_CNN import QuickDrawCNN  # 导入模型类
import numpy as np  # 导入 numpy
import matplotlib.pyplot as plt
import random

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("model/similarity_model.pkl", map_location=device, weights_only=False)
model = model.to(device)
model.eval()

# 类别标签
categories = ['bed', 'apple', 'banana', 'airplane', 'book', 'bread']
label_map_inv = {i: name for i, name in enumerate(categories)}

# 图像库中的样本数据（假设这些是你的图库中的图片）
# 用于提供参考图片
# 假设图库样本是预处理好的28x28大小的图像
image_samples = {i: Image.open(f"reference_images/{category}.png").convert("L").resize((28, 28)) for i, category in
                 enumerate(categories)}

# Tkinter画布设置
WIDTH, HEIGHT = 560, 560  # 提高画布尺寸，使用更大的画布
white = (255, 255, 255)

root = tk.Tk()
root.title("QuickDraw - 手绘识别")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
canvas.pack()

image = Image.new("RGB", (WIDTH, HEIGHT), white)
draw = ImageDraw.Draw(image)


def draw_callback(event):
    x, y = event.x, event.y
    r = 5  # 增加绘制的圆点大小
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
    draw.ellipse([x - r, y - r, x + r, y + r], fill='black')


canvas.bind("<B1-Motion>", draw_callback)


def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill='white')


def predict_drawing():
    # 缩小图像为28x28，进行灰度处理
    img = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    img = ImageOps.invert(img)
    img_tensor = transforms.ToTensor()(img)
    img_tensor = transforms.Normalize((0.5,), (0.5,))(img_tensor).unsqueeze(0).to(device)

    # 模型推理
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
        pred_prob = torch.nn.functional.softmax(output, dim=1).max().item()  # 获取置信度
        result_label.config(text=f"识别结果: {label_map_inv[pred]} (置信度: {pred_prob:.2f})")

        # 数据可视化（绘图 + 结果）
        visualize_result(img, pred, pred_prob)


def visualize_result(img, pred, pred_prob):
    # 将图像转换为 NumPy 数组并显示
    img = img.convert("RGB")
    img_arr = np.array(img)  # 确保 NumPy 被导入
    plt.figure(figsize=(6, 6))
    plt.imshow(img_arr, cmap="gray")
    plt.title(f"预测: {label_map_inv[pred]}\n置信度: {pred_prob:.2f}")
    plt.axis("off")

    # 从图库中随机选择一张对照图像
    sample_img = image_samples[pred]
    sample_img_arr = np.array(sample_img)

    # 显示参考图片
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_img_arr, cmap="gray")
    plt.title(f"参考图片: {label_map_inv[pred]}")
    plt.axis("off")

    # 显示两个图像（用户绘制的与参考图库图像）
    plt.show()


# 按钮区域
btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="清空", command=clear_canvas).pack(side=tk.LEFT)
tk.Button(btn_frame, text="识别", command=predict_drawing).pack(side=tk.LEFT)

result_label = tk.Label(root, text="识别结果: ")
result_label.pack()

root.mainloop()
