import os

# 设置目标图片目录
folder_path = r'D:\PycharmProject\picturetest\yolov5_animal_dataset\captured_center_integral'
folder_name = os.path.basename(folder_path.rstrip("/\\"))

# 支持的图片扩展名
image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

# 获取所有图片文件
images = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_exts]
images.sort()

for idx, filename in enumerate(images):
    old_path = os.path.join(folder_path, filename)
    ext = os.path.splitext(filename)[1].lower()
    new_name = f"{folder_name}_{idx:03d}{ext}"
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    print(f"✅ {filename} → {new_name}")
