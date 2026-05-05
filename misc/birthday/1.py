import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
import numpy as np

TEXT = "生日快乐"
FONT_PATH = "simhei.ttf"  # 黑体或任意支持中文的字体
FONT_SIZE = 32
CELL_SIZE = 10
THRESHOLD = 128
DELAY_BETWEEN_CHARS = 400  # 毫秒

class HighResLED:
    def __init__(self, master, text):
        self.master = master
        self.text = text
        self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        self.char_matrices = []
        self.char_dots = []
        self.total_width = 0
        self.rows = 0
        self.prepare_all_characters()

        self.canvas = tk.Canvas(master, bg="black", width=self.total_width * CELL_SIZE, height=self.rows * CELL_SIZE)
        self.canvas.pack()
        self.current_char_index = 0
        self.animate_next_character()

    def render_char_to_matrix(self, char):
        img = Image.new("L", (FONT_SIZE, FONT_SIZE), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, fill=255, font=self.font)
        matrix = np.array(img)
        matrix = (matrix > THRESHOLD).astype(int)
        # 去掉多余空行和空列
        rows = np.any(matrix, axis=1)
        cols = np.any(matrix, axis=0)
        matrix = matrix[rows][:, cols]
        return matrix

    def prepare_all_characters(self):
        for char in self.text:
            matrix = self.render_char_to_matrix(char)
            self.char_matrices.append(matrix)
            self.rows = max(self.rows, matrix.shape[0])
            self.total_width += matrix.shape[1] + 2  # 加间距

    def draw_character(self, matrix, offset_x):
        char_dots = []
        for y in range(matrix.shape[0]):
            row = []
            for x in range(matrix.shape[1]):
                on = matrix[y, x]
                dot = self.canvas.create_oval(
                    (offset_x + x) * CELL_SIZE,
                    y * CELL_SIZE,
                    (offset_x + x) * CELL_SIZE + CELL_SIZE - 2,
                    y * CELL_SIZE + CELL_SIZE - 2,
                    fill="gray10", outline=""
                )
                row.append((dot, on))
            char_dots.append(row)
        self.char_dots.append(char_dots)

    def animate_next_character(self):
        if self.current_char_index >= len(self.char_matrices):
            self.show_final_message()
            return

        # 画当前字符的 LED 点
        matrix = self.char_matrices[self.current_char_index]
        offset_x = sum(self.char_matrices[i].shape[1] + 2 for i in range(self.current_char_index))
        self.draw_character(matrix, offset_x)

        # 动画点亮当前字符
        def light_up(i):
            if i >= len(self.char_dots[self.current_char_index]):
                self.current_char_index += 1
                self.master.after(DELAY_BETWEEN_CHARS, self.animate_next_character)
                return
            row = self.char_dots[self.current_char_index][i]
            for dot, on in row:
                if on:
                    self.canvas.itemconfig(dot, fill="lime")
            self.master.after(30, lambda: light_up(i + 1))

        light_up(0)

    def show_final_message(self):
        self.canvas.create_text(
            self.total_width * CELL_SIZE // 2,
            self.rows * CELL_SIZE + 20,
            text="🎉 祝你生日快乐！",
            fill="white",
            font=("Arial", 16, "bold")
        )

if __name__ == "__main__":
    root = tk.Tk()
    root.title("🎂 自适应LED祝福动画")
    app = HighResLED(root, TEXT)
    root.mainloop()
