from rembg import remove
from PIL import Image
import os
import random

# 输入图像路径（动物图片）
animal_dir = "animals/"
# 背景图片路径
background_dir = "backgrounds/"
# 输出增强图像路径
output_dir = "augmented/"

os.makedirs(output_dir, exist_ok=True)

# 遍历所有动物图
for filename in os.listdir(animal_dir):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    animal_path = os.path.join(animal_dir, filename)
    animal_img = Image.open(animal_path).convert("RGBA")

    # 使用 rembg 抠图（去除背景）
    fg = remove(animal_img)

    # 随机选择一个背景图
    bg_filename = random.choice(os.listdir(background_dir))
    bg_path = os.path.join(background_dir, bg_filename)
    bg = Image.open(bg_path).convert("RGBA")

    # 调整背景尺寸与前景一致（或你想要的尺寸）
    bg = bg.resize((224, 224))
    fg = fg.resize((224, 224))

    # 合并前景与背景
    composite = Image.alpha_composite(bg, fg)

    # 保存结果
    save_name = f"{os.path.splitext(filename)[0]}_aug.jpg"
    composite.convert("RGB").save(os.path.join(output_dir, save_name))

print("✅ 处理完成，增强图像保存在：", output_dir)
