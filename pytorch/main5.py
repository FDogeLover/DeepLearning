import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import time

from QuickDrwa_CNN import QuickDrawCNN  # 自定义模型结构

# 类别设置与文件路径
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

# 图像参数
WIDTH, HEIGHT = 560, 560
black = (0, 0, 0)
r = 16

# 初始化窗口
root = tk.Tk()
root.title("QuickDraw 游戏")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='black')
canvas.pack()
image = Image.new("RGB", (WIDTH, HEIGHT), black)
draw = ImageDraw.Draw(image)

# 变量初始化
current_round = 0
target_category = ""
score = 0
timer_id = None
time_left = 20
game_running = False
rounds_total = 6

# 绘图处理
last_x, last_y = None, None
def draw_callback(event):
    global last_x, last_y
    if not game_running: return
    x, y = event.x, event.y
    if last_x is not None and last_y is not None:
        canvas.create_line(last_x, last_y, x, y, width=r, fill='white', capstyle=tk.ROUND, smooth=True)
        draw.line([last_x, last_y, x, y], fill='white', width=r)
    last_x, last_y = x, y

def reset_coords(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", draw_callback)
canvas.bind("<ButtonRelease-1>", reset_coords)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=black)

# 图像预处理
def preprocess(img):
    img = img.resize((28, 28), Image.Resampling.LANCZOS).convert("L")
    tensor = transforms.ToTensor()(img)
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor).unsqueeze(0)
    return tensor.to(device)

# 模型预测类别
def predict_category(img):
    tensor = preprocess(img)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return categories[pred]

# 倒计时
def update_timer():
    global time_left, timer_id, game_running
    if time_left <= 0:
        check_prediction()
        return
    time_left -= 1
    timer_label.config(text=f"剩余时间: {time_left}s")
    timer_id = root.after(1000, update_timer)

# 检查预测是否正确
def check_prediction():
    global current_round, score, game_running
    drawn_pred = predict_category(image)
    if drawn_pred == target_category:
        result_label.config(text=f"✅ 恭喜你画出了 {target_category}!")
        score += 1
    else:
        result_label.config(text=f"❌ 你画的是 {drawn_pred}，目标是 {target_category}")
    current_round += 1
    game_running = False

    if current_round < rounds_total:
        root.after(2000, start_round)
    else:
        result_label.config(text=f"🎉 游戏结束！你共答对 {score}/{rounds_total} 次。")

# 开始一轮
def start_round():
    global target_category, time_left, game_running
    clear_canvas()
    time_left = 20
    target_category = random.choice(categories)
    timer_label.config(text=f"剩余时间: {time_left}s")
    result_label.config(text=f"第 {current_round+1} 轮：请画一个 “{target_category}”")
    game_running = True
    update_timer()

# 启动游戏
def start_game():
    global current_round, score
    current_round = 0
    score = 0
    start_round()

# UI组件
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
tk.Button(btn_frame, text="开始游戏", command=start_game).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="清空画布", command=clear_canvas).pack(side=tk.LEFT, padx=10)

timer_label = tk.Label(root, text="剩余时间: 20s", font=("Arial", 14))
timer_label.pack(pady=5)

result_label = tk.Label(root, text="点击开始游戏", font=("Arial", 12))
result_label.pack(pady=5)

root.mainloop()
