import numpy as np
from PIL import Image

def preprocess(canvas_image):
    img = canvas_image.resize((28, 28)).convert('L')  # 转为灰度
    img = np.array(img)
    img = 255 - img  # 反色处理（黑底白笔）
    img = img / 255.0
    return img.reshape(1, 28, 28, 1)
