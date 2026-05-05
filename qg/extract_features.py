# 导入必要的库
import json
import torch
from model import ImageFeatureExtractor  # 引入自定义的CNN特征提取模型
from PIL import Image, ImageDraw         # 用于绘图，将笔画转换成图像
import numpy as np
from tqdm import tqdm                    # 显示处理进度条

# 将一条笔画数据（drawing）转换为 64x64 的灰度图像
def draw_strokes(drawing, size=64, stroke_width=3):
    img = Image.new("L", (256, 256), color=255)  # 创建白底图像（L为灰度模式）
    draw = ImageDraw.Draw(img)  # 获取画笔对象

    # 遍历每一笔 stroke（x[], y[]）
    for stroke in drawing:
        x, y = stroke
        points = list(zip(x, y))
        if len(points) > 1:
            draw.line(points, fill=0, width=stroke_width)  # 黑色线条画笔

    # 缩放到目标尺寸并归一化
    img = img.resize((size, size)).convert("L")
    img_array = np.array(img, dtype=np.float32) / 255.0  # 归一化为 0~1
    return torch.tensor(img_array).unsqueeze(0)  # 返回张量形状为 [1, 64, 64]

# 主函数：批量处理 .ndjson 文件并提取所有图像特征
def process_ndjson(path, model, device='cpu', max_samples=None):
    features = []  # 存储所有特征向量
    labels = []    # 存储所有对应标签

    model.to(device)      # 将模型加载到设备（CPU/GPU）
    model.eval()          # 设置为推理模式

    # 打开 .ndjson 文件，逐行读取每一条图像记录
    with open(path, 'r') as f:
        for idx, line in enumerate(tqdm(f, desc="Processing")):
            if max_samples and idx >= max_samples:
                break  # 如果设置了最大处理数量，则提前终止

            obj = json.loads(line)          # 解析JSON字符串为字典对象
            image = draw_strokes(obj['drawing'])   # 转为图像张量 [1, 64, 64]
            image = image.unsqueeze(0).to(device)  # 加 batch 维度变成 [1, 1, 64, 64]

            with torch.no_grad():           # 不需要梯度，节省内存
                feature = model(image)      # 前向传播得到特征向量 [1, 128]

            features.append(feature.squeeze(0).cpu())  # 移除 batch 维度并转为CPU
            labels.append(obj['word'])                 # 提取对应标签

    # 返回特征张量列表和标签列表
    return torch.stack(features), labels

# 脚本主入口
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # 添加命令行参数：输入路径、输出路径、最多处理数量
    parser.add_argument('--ndjson', type=str, default='ndjson/banana.ndjson')
    parser.add_argument('--output', type=str, default='feature/features.pt')
    parser.add_argument('--max', type=int, default=None, help="Max samples to process")
    args = parser.parse_args()

    # 初始化模型
    model = ImageFeatureExtractor(output_dim=128)

    # 执行处理过程
    features, labels = process_ndjson(args.ndjson, model, max_samples=args.max)

    # 保存到 PyTorch 文件
    torch.save({'features': features, 'labels': labels}, args.output)
    print(f"Saved {features.shape[0]} features to {args.output}")
