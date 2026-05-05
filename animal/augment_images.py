import os
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
# 每类目标图片数量
TARGET_COUNT = 1000

# 数据目录结构：dataset_raw/elephant/, dataset_raw/monkey/, ...
INPUT_DIR = "dataset_raw"
OUTPUT_DIR = "dataset_augmented"

# 定义数据增强组合
augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
    transforms.ToTensor(),  # ！！！必须先转为 Tensor 后面才支持 shape
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])


# PIL图像保存格式
def save_augmented_image(image_tensor, save_path, idx):
    filename = f"aug_{idx}.jpg"
    image_pil = to_pil_image(image_tensor)  # ✅ 转为 PIL 图像
    image_pil.save(os.path.join(save_path, filename))

# 对每个类别进行增强
for class_name in os.listdir(INPUT_DIR):
    input_class_dir = os.path.join(INPUT_DIR, class_name)
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    images = [f for f in os.listdir(input_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    original_count = len(images)
    print(f"🔍 类别 {class_name} 原始数量: {original_count}")

    # 复制原始图像
    for i, img_name in enumerate(images):
        img = Image.open(os.path.join(input_class_dir, img_name)).convert("RGB")
        img.save(os.path.join(output_class_dir, f"orig_{i:04d}.jpg"))

    # 增强图像直到达到目标数量
    idx = original_count
    with tqdm(total=TARGET_COUNT - original_count, desc=f"增强 {class_name}") as pbar:
        while idx < TARGET_COUNT:
            img_path = os.path.join(input_class_dir, random.choice(images))
            image = Image.open(img_path).convert("RGB")
            augmented = augment(image)
            save_augmented_image(augmented, output_class_dir, idx)
            idx += 1
            pbar.update(1)

print("✅ 所有类别增强完成，已保存至:", OUTPUT_DIR)
