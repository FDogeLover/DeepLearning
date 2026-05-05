import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 数据（例如飞机图像）
data = np.load("converted_npy/full_resized_apple.npy")

# 随机选择前5张图像进行可视化
plt.figure(figsize=(10, 2))
for i in range(5):
    img = data[i].reshape(64, 64)  # 将784维向量恢复为28x28图像
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")

plt.suptitle("Dataset Sample: ")
plt.show()
