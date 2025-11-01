# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')  # 使用 TkAgg 后端来显示图像窗口

import cv2
import matplotlib.pyplot as plt

# 读取图片（请放一张图片在同目录下，比如 test.jpg）
image = cv2.imread('test.jpg')

# 检查是否成功读取
if image is None:
    print("❌ 图片未找到，请确认文件名或路径是否正确！")
    exit()

# OpenCV 默认读取是 BGR，我们先转为 RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化（阈值为128）
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

# 显示结果
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gray")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Binary")
plt.imshow(binary, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
