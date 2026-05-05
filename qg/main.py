import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk
import torch
import torch.nn.functional as F
import numpy as np
from model3 import ImageFeatureExtractor
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTextEdit, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

CANVAS_SIZE = 256
IMG_SIZE = 64
STROKE_WIDTH = 8
TOPK = 5  # 显示前五相似图

class DrawingApp:
    def __init__(self, root, model_path="model.pth", feature_db_path="feature_db.pt"):
        self.root = root
        self.root.title("手绘图相似度识别")
        self.root.configure(bg="#f0f2f5")
        
        # 设置窗口最小尺寸
        self.root.minsize(1000, 700)
        
        # 创建主框架
        main_frame = tk.Frame(root, bg="#f0f2f5")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)
        
        # 创建标题
        title_label = tk.Label(
            main_frame,
            text="手绘图相似度识别",
            font=("Microsoft YaHei UI", 24, "bold"),
            bg="#f0f2f5",
            fg="#1a1a1a"
        )
        title_label.pack(pady=(0, 20))
        
        # 创建题目选择区域
        question_frame = tk.Frame(main_frame, bg="#ffffff", bd=1, relief="solid")
        question_frame.pack(fill="x", pady=(0, 20))
        
        # 题目选择区域内部布局
        question_inner = tk.Frame(question_frame, bg="#ffffff", padx=15, pady=10)
        question_inner.pack(fill="x")
        
        tk.Label(
            question_inner,
            text="选择题目：",
            font=("Microsoft YaHei UI", 12),
            bg="#ffffff",
            fg="#333333"
        ).pack(side="left")
        
        # 创建题目下拉框
        self.question_var = tk.StringVar()
        self.question_combo = ttk.Combobox(
            question_inner,
            textvariable=self.question_var,
            width=30,
            state="readonly",
            font=("Microsoft YaHei UI", 10)
        )
        self.question_combo.pack(side="left", padx=10)
        
        # 创建题目预览区域
        self.preview_label = tk.Label(
            question_inner,
            text="",
            font=("Microsoft YaHei UI", 10),
            bg="#ffffff",
            fg="#666666"
        )
        self.preview_label.pack(side="left", padx=10)
        
        # 创建内容区域
        content_frame = tk.Frame(main_frame, bg="#f0f2f5")
        content_frame.pack(expand=True, fill="both")
        
        # 左侧绘图区域
        left_frame = tk.Frame(content_frame, bg="#ffffff", bd=1, relief="solid")
        left_frame.pack(side="left", padx=(0, 20), fill="both", expand=True)
        
        # 绘图区域标题
        canvas_title = tk.Label(
            left_frame,
            text="绘图区域",
            font=("Microsoft YaHei UI", 12),
            bg="#ffffff",
            fg="#333333"
        )
        canvas_title.pack(pady=(15, 10))
        
        # 创建画布容器
        canvas_container = tk.Frame(
            left_frame,
            bg="#e8f0fe",  # 浅蓝色背景
            bd=2,
            relief="solid"
        )
        canvas_container.pack(padx=20, pady=(0, 20))
        
        # 绘图画布
        self.canvas = tk.Canvas(
            canvas_container,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="white",
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack(padx=2, pady=2)  # 添加内边距，使画布与边框有一定间距
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # 添加提示文本
        hint_label = tk.Label(
            canvas_container,
            text="在此区域绘制图形",
            font=("Microsoft YaHei UI", 9),
            bg="#e8f0fe",
            fg="#666666"
        )
        hint_label.pack(pady=(0, 5))
        
        # 按钮区域
        button_frame = tk.Frame(left_frame, bg="#ffffff")
        button_frame.pack(pady=(0, 20))
        
        # 自定义按钮样式
        button_style = {
            "font": ("Microsoft YaHei UI", 10),
            "bg": "#4a90e2",
            "fg": "white",
            "padx": 20,
            "pady": 8,
            "borderwidth": 0,
            "cursor": "hand2"
        }
        
        self.clear_btn = tk.Button(
            button_frame,
            text="清空画布",
            command=self.clear,
            **button_style
        )
        self.clear_btn.pack(side="left", padx=5)
        
        self.search_btn = tk.Button(
            button_frame,
            text="开始检索",
            command=self.search,
            **button_style
        )
        self.search_btn.pack(side="left", padx=5)
        
        # 右侧结果区域
        right_frame = tk.Frame(content_frame, bg="#ffffff", bd=1, relief="solid")
        right_frame.pack(side="right", fill="both", expand=True)
        
        # 结果区域标题
        result_title = tk.Label(
            right_frame,
            text="检索结果",
            font=("Microsoft YaHei UI", 12),
            bg="#ffffff",
            fg="#333333"
        )
        result_title.pack(pady=(15, 10))
        
        # 结果文本区域
        self.result_text = tk.Text(
            right_frame,
            height=10,
            width=40,
            font=("Microsoft YaHei UI", 10),
            bg="#ffffff",
            fg="#333333",
            relief="solid",
            borderwidth=1
        )
        self.result_text.pack(padx=20, pady=(0, 20))
        
        # 相似图像展示区域
        sim_frame = tk.Frame(right_frame, bg="#ffffff")
        sim_frame.pack(pady=(0, 20))
        
        self.sim_imgs = []
        for i in range(TOPK):
            frame = tk.Frame(sim_frame, bg="#ffffff")
            frame.pack(side="left", padx=5)
            
            label = tk.Label(frame, bg="#ffffff")
            label.pack()
            
            sim_label = tk.Label(
                frame,
                text=f"相似图 {i+1}",
                font=("Microsoft YaHei UI", 9),
                bg="#ffffff",
                fg="#666666"
            )
            sim_label.pack()
            
            self.sim_imgs.append(label)
        
        # 初始化图像和绘图对象
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        # 加载模型和特征库
        self.device = torch.device('cpu')
        self.model = ImageFeatureExtractor().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        db = torch.load(feature_db_path)
        self.db_features = F.normalize(db['features'], dim=1)  # [N, 128]
        self.db_labels = db['labels']  # [N]
        self.db_drawings = db['drawings']  # 原始ndjson格式 stroke 列表
        
        # 初始化题目列表
        self.questions = self._get_unique_questions()
        self.question_combo['values'] = self.questions
        if self.questions:
            self.question_combo.set(self.questions[0])
            self.question_combo.bind('<<ComboboxSelected>>', self._on_question_selected)
        
        # 绑定按钮悬停效果
        for btn in [self.clear_btn, self.search_btn]:
            btn.bind("<Enter>", lambda e, b=btn: b.configure(bg="#357abd"))
            btn.bind("<Leave>", lambda e, b=btn: b.configure(bg="#4a90e2"))
        
        print(f"✅ 特征库加载完成，图像数: {len(self.db_labels)}")
        print(f"✅ 题目数量: {len(self.questions)}")

    def _get_unique_questions(self):
        """获取所有唯一的题目"""
        return sorted(list(set(self.db_labels)))
    
    def _on_question_selected(self, event=None):
        """当选择新题目时更新预览"""
        selected = self.question_var.get()
        self.preview_label.config(text=f"当前题目: {selected}")
        self.clear()  # 清空画布
    
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
        
        # 检查是否选择了题目
        if not self.question_var.get():
            self.result_text.insert(tk.END, "⚠️ 请先选择一道题目。\n")
            return
        
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
            
            # 只考虑当前题目的相似度
            current_question = self.question_var.get()
            question_mask = torch.tensor([label == current_question for label in self.db_labels])
            question_sims = sims[question_mask]
            question_indices = torch.nonzero(question_mask).squeeze(1)
            
            topk_scores, topk_local_indices = torch.topk(question_sims, min(TOPK, len(question_sims)))
            topk_indices = question_indices[topk_local_indices]
        
        self.result_text.insert(tk.END, f"🔍 当前题目 '{current_question}' 的相似度结果：\n\n")
        for i in range(len(topk_scores)):
            idx = topk_indices[i].item()
            score = topk_scores[i].item()
            self.result_text.insert(tk.END, f"{i+1}. 相似度: {score:.4f}\n")
            
            drawing = self.db_drawings[idx]
            img = self.stroke_to_image(drawing, size=64)
            tk_img = ImageTk.PhotoImage(img)
            self.sim_imgs[i].config(image=tk_img)
            self.sim_imgs[i].image = tk_img


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("文本处理工具")
        self.setMinimumSize(800, 600)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #4a90e2;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2d6da3;
            }
            QTextEdit {
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                font-size: 14px;
            }
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建标题
        title_label = QLabel("文本处理工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        """)
        main_layout.addWidget(title_label)
        
        # 创建输入区域框架
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                padding: 15px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setSpacing(15)
        
        # 创建输入区域
        input_label = QLabel("输入文本:")
        input_label.setStyleSheet("font-weight: bold;")
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("在此输入要处理的文本...")
        self.input_text.setMinimumHeight(150)
        
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_text)
        
        # 创建按钮区域
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.uppercase_btn = QPushButton("转大写")
        self.lowercase_btn = QPushButton("转小写")
        self.reverse_btn = QPushButton("反转文本")
        self.count_btn = QPushButton("统计字数")
        self.clear_btn = QPushButton("清空")
        
        for btn in [self.uppercase_btn, self.lowercase_btn, self.reverse_btn, 
                   self.count_btn, self.clear_btn]:
            button_layout.addWidget(btn)
        
        input_layout.addLayout(button_layout)
        main_layout.addWidget(input_frame)
        
        # 创建输出区域框架
        output_frame = QFrame()
        output_frame.setStyleSheet("""
            QFrame {
                padding: 15px;
            }
        """)
        output_layout = QVBoxLayout(output_frame)
        output_layout.setSpacing(15)
        
        # 创建输出区域
        output_label = QLabel("处理结果:")
        output_label.setStyleSheet("font-weight: bold;")
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("处理结果将显示在这里...")
        self.output_text.setMinimumHeight(150)
        
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_text)
        
        # 创建文件操作按钮区域
        file_button_layout = QHBoxLayout()
        file_button_layout.setSpacing(10)
        
        self.save_btn = QPushButton("保存结果")
        self.load_btn = QPushButton("加载文件")
        
        for btn in [self.save_btn, self.load_btn]:
            file_button_layout.addWidget(btn)
        
        output_layout.addLayout(file_button_layout)
        main_layout.addWidget(output_frame)
        
        # 连接信号和槽
        self.uppercase_btn.clicked.connect(self.to_uppercase)
        self.lowercase_btn.clicked.connect(self.to_lowercase)
        self.reverse_btn.clicked.connect(self.reverse_text)
        self.count_btn.clicked.connect(self.count_chars)
        self.clear_btn.clicked.connect(self.clear_text)
        self.save_btn.clicked.connect(self.save_result)
        self.load_btn.clicked.connect(self.load_file)
        
        # 设置窗口图标
        self.setWindowIcon(QIcon("icon.png"))  # 如果有图标文件的话

# 运行 GUI 应用
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root, model_path="model/model.pth", feature_db_path="feature/feature_db.pt")
    root.mainloop()
