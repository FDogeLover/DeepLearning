import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def draw_image(drawing, image_size=64, line_width=5):
    img = Image.new("L", (256, 256), color=255)  # 白底
    draw = ImageDraw.Draw(img)

    for stroke in drawing:
        for i in range(len(stroke[0]) - 1):
            x1, y1 = stroke[0][i], stroke[1][i]
            x2, y2 = stroke[0][i + 1], stroke[1][i + 1]
            draw.line([x1, y1, x2, y2], fill=0, width=line_width)

    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return np.array(img)

def convert_ndjson_to_npy(ndjson_path, output_path, max_samples=3000):
    drawings = []
    with open(ndjson_path, 'r') as f:
        for line in tqdm(f, desc=f"Processing {ndjson_path}"):
            if len(drawings) >= max_samples:
                break
            data = json.loads(line)
            image = draw_image(data['drawing'], image_size=64)
            drawings.append(image)

    drawings = np.array(drawings, dtype=np.uint8)
    np.save(output_path, drawings)
    print(f"Saved {len(drawings)} images to {output_path}")

if __name__ == "__main__":
    # 示例：转换 apple 类别
    os.makedirs("converted_npy", exist_ok=True)

    category = ["apple","banana"]
    for i in range(2):
        input_ndjson = f"similarity_row/full_raw_{category[i]}.ndjson"  # 你的原始数据路径
        output_npy = f"converted_npy/full_resized_{category[i]}.npy"   # 输出保存路径

    convert_ndjson_to_npy(input_ndjson, output_npy, max_samples=3000)
