from PIL import Image, ImageEnhance
import random
import os
import warnings

# 忽略像素炸弹警告
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# 参数配置
map_path = 'map.jpg'                  # 地图图片
animal_folder = 'animals_pic/'        # 存放动物 PNG 的文件夹
output_dir = 'outputs/'               # 输出目录
generate_num = 100                     # 要生成几张随机图

animal_size = (1000, 1000)            # 动物统一缩放尺寸
os.makedirs(output_dir, exist_ok=True)

# 加载地图
map_base = Image.open(map_path).convert("RGBA")
map_w, map_h = map_base.size

# 获取所有动物图像文件
animal_imgs = [
    Image.open(os.path.join(animal_folder, f)).convert("RGBA")
    for f in os.listdir(animal_folder) if f.lower().endswith('.png')
]

print(f"加载动物数量：{len(animal_imgs)}")

# 主循环：生成多张图片
for idx in range(generate_num):
    # 复制地图图像
    map_img = map_base.copy()

    # === 对地图亮度进行随机调整 ===
    map_brightness_factor = random.uniform(0.5, 1.5)  # 可调范围
    map_img = ImageEnhance.Brightness(map_img).enhance(map_brightness_factor)

    # 处理每个动物图像（读取路径名以判断是否是象）
    for filename in os.listdir(animal_folder):
        if not filename.lower().endswith('.png'):
            continue

        animal_path = os.path.join(animal_folder, filename)
        animal = Image.open(animal_path).convert("RGBA")

        # 判断是否是“象”，决定缩放尺寸
        if 'elephant' in filename.lower() or '象' in filename:
            a = animal.resize((animal_size[0] * 2, animal_size[1] * 2))  # 大象放大2倍
        else:
            a = animal.resize(animal_size)  # 其他动物正常大小

        # 随机亮度
        brightness_factor = random.uniform(0.5, 1.5)
        a = ImageEnhance.Brightness(a).enhance(brightness_factor)

        # 随机旋转
        angle = random.uniform(0, 360)
        a = a.rotate(angle, expand=True)

        # 随机位置
        a_w, a_h = a.size
        x = random.randint(0, max(0, map_w - a_w))
        y = random.randint(0, max(0, map_h - a_h))

        # 合成
        map_img.paste(a, (x, y), a)

    # 保存结果
    output_path = os.path.join(output_dir, f"output_{idx+1:03d}.jpg")
    map_img.convert("RGB").save(output_path)
    print(f"[{idx+1}/{generate_num}] 已保存：{output_path}")
