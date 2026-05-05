import sys
import os
import random
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPainter, QPen, QPixmap, QFont
from PyQt5.QtCore import Qt, QPoint
from PIL import Image

from model.similarity import CLIPSimilarity


class DrawingBoard(QWidget):
    def __init__(self, prompt_file=None):
        super().__init__()
        self.setWindowTitle("🎨 AI绘图练习板")
        self.setFixedSize(1000, 700)

        self.last_point = QPoint()
        self.drawing = False
        self.canvas = QPixmap(800, 600)
        self.canvas.fill(Qt.white)

        # 加载题目
        self.prompts = []
        if prompt_file and os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        self.prompt_text = self.get_random_prompt()
        self.reference_image_path = ""

        self.similarity_checker = CLIPSimilarity()

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # ===== 顶部提示 =====
        self.label = QLabel(f"📝 请绘制：{self.prompt_text}")
        self.label.setFont(QFont("微软雅黑", 16))
        self.label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.label)

        # ===== 中间区域 =====
        center_layout = QHBoxLayout()

        drawing_area = QWidget()
        drawing_area.setFixedSize(800, 600)
        center_layout.addWidget(drawing_area)

        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(20)

        btn_clear = QPushButton("🧹 清空画布")
        btn_clear.clicked.connect(self.clear_canvas)

        btn_next = QPushButton("➡️ 下一题")
        btn_next.clicked.connect(self.next_prompt)

        btn_save = QPushButton("💾 保存图像")
        btn_save.clicked.connect(self.save_canvas)

        btn_check = QPushButton("🔍 相似度检测")
        btn_check.clicked.connect(self.check_similarity)

        for btn in [btn_clear, btn_next, btn_save, btn_check]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 15px;
                    padding: 6px 12px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            btn_layout.addWidget(btn)

        btn_layout.addStretch()
        center_layout.addLayout(btn_layout)

        main_layout.addLayout(center_layout)
        self.setLayout(main_layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(50, 90, self.canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_in_canvas(event.pos()):
            self.last_point = event.pos() - QPoint(50, 90)
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing and self.is_in_canvas(event.pos()):
            painter = QPainter(self.canvas)
            pen = QPen(Qt.black, 3, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos() - QPoint(50, 90))
            self.last_point = event.pos() - QPoint(50, 90)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def is_in_canvas(self, pos):
        return 50 <= pos.x() <= 850 and 90 <= pos.y() <= 690

    def clear_canvas(self):
        self.canvas.fill(Qt.white)
        self.update()

    def next_prompt(self):
        self.prompt_text = self.get_random_prompt()
        self.label.setText(f"📝 请绘制：{self.prompt_text}")
        self.clear_canvas()

    def save_canvas(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "PNG Files (*.png)")
        if save_path:
            self.canvas.save(save_path)

    def get_random_prompt(self):
        if self.prompts:
            prompt = random.choice(self.prompts)
            self.set_reference_image(prompt)
            return prompt
        self.set_reference_image("自由绘画")
        return "自由绘画"

    def set_reference_image(self, prompt_text):
        ref_path = os.path.join("assets", "参考图", f"{prompt_text}.png")
        if os.path.exists(ref_path):
            self.reference_image_path = ref_path
        else:
            self.reference_image_path = ""

    def is_blank_image(self, img_path, threshold=10):
        image = Image.open(img_path).convert("L")
        arr = np.array(image)
        std = arr.std()
        return std < threshold

    def check_similarity(self):
        os.makedirs("data", exist_ok=True)
        temp_path = "data/temp.png"
        self.canvas.save(temp_path)

        # 检测是否为空白图像
        if self.is_blank_image(temp_path):
            QMessageBox.warning(self, "⚠️ 提示", "检测到画布几乎为空，请绘制后再进行检测。")
            return

        if self.reference_image_path:
            score = self.similarity_checker.image_to_image_similarity(temp_path, self.reference_image_path)
            result = f"与参考图像相似度：{score:.4f}"
        else:
            score = self.similarity_checker.image_to_prompt_similarity(temp_path, self.prompt_text)
            result = f"与题目“{self.prompt_text}”匹配度：{score:.4f}"

        QMessageBox.information(self, "🎯 检测结果", result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrawingBoard(prompt_file="assets/prompts.txt")
    window.show()
    sys.exit(app.exec_())
