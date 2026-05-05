import os
import cv2
import albumentations as A
from glob import glob
import random

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.3),
    A.ToGray(p=0.3),  # 增加灰度转换，设置概率为30%
    A.Resize(640, 640),
    A.RandomCrop(height=480, width=480, p=0.5),
    A.HorizontalFlip(p=0.5),
],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)


def augment_image(image_path, label_path, save_img_dir, save_label_dir, prefix='aug_', augment_index=0):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read image {image_path}")
        return False

    class_labels = []
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, x_c, y_c, w, h = line.strip().split()
            class_labels.append(int(cls))
            bboxes.append([float(x_c), float(y_c), float(w), float(h)])

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_classes = augmented['class_labels']

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_label_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    img_save_path = os.path.join(save_img_dir, f"{prefix}{base_name}_{augment_index}.jpg")
    cv2.imwrite(img_save_path, aug_image)

    label_save_path = os.path.join(save_label_dir, f"{prefix}{base_name}_{augment_index}.txt")
    with open(label_save_path, 'w') as f:
        for cls, bbox in zip(aug_classes, aug_bboxes):
            x_c, y_c, w, h = bbox
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    print(f"Saved augmented image to {img_save_path} and label to {label_save_path}")
    return True

if __name__ == '__main__':
    image_dir = 'images/train'
    label_dir = 'labels/train'

    train_img_dir = 'augmented/train/images'
    train_label_dir = 'augmented/train/labels'
    val_img_dir = 'augmented/val/images'
    val_label_dir = 'augmented/val/labels'

    num_augments = 5  # 每张图生成多少张增强图

    image_files = glob(os.path.join(image_dir, '*.jpg')) + glob(os.path.join(image_dir, '*.png'))
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base_name + '.txt')
        if os.path.exists(label_path):
            val_index = random.randint(0, num_augments - 1)  # 随机选一个放验证集
            for i in range(num_augments):
                if i == val_index:
                    # 保存到验证集目录
                    augment_image(img_path, label_path, val_img_dir, val_label_dir, augment_index=i)
                else:
                    # 保存到训练集目录
                    augment_image(img_path, label_path, train_img_dir, train_label_dir, augment_index=i)
        else:
            print(f"Label file not found for image {img_path}")
