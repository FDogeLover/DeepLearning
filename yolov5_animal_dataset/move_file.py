import os
import shutil

folderA = 'images/train'
folderB = 'labels/train'
target_folderA = 'images/val'
target_folderB = 'labels/val'

# 创建目标文件夹（如果不存在）
os.makedirs(target_folderA, exist_ok=True)
os.makedirs(target_folderB, exist_ok=True)

# 获取两个文件夹中所有文件名
filesA = set(os.listdir(folderA))
filesB = set(os.listdir(folderB))

# 找出两个文件夹中共有的文件（同名）
common_files = filesA & filesB

print(f"共有 {len(common_files)} 个同名文件，将被移动：")
for filename in common_files:
    print(f" - {filename}")

    # 移动文件
    shutil.move(os.path.join(folderA, filename), os.path.join(target_folderA, filename))
    shutil.move(os.path.join(folderB, filename), os.path.join(target_folderB, filename))

print("文件移动完成。")
