# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt

# 读取图片
image = cv2.imread('test.jpg')
if image is None:
    print("❌ 未找到图像，请检查路径。")
    exit()

# 转换为 RGB，方便 Matplotlib 显示
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取原始尺寸
h, w = image_rgb.shape[:2]
print(f"原图尺寸: {w} x {h}")

# 缩小为原来的 0.5 倍
small = cv2.resize(image_rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
h_small, w_small = small.shape[:2]
print(f"缩小后尺寸: {w_small} x {h_small}")

# 放大为原来的 2 倍
large = cv2.resize(image_rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
h_large, w_large = large.shape[:2]
print(f"放大后尺寸: {w_large} x {h_large}")

# 显示结果（保持比例）
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title(f"Original\n{w}x{h}")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Smaller (0.5x)\n{w_small}x{h_small}")
plt.imshow(small)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"Larger (2x)\n{w_large}x{h_large}")
plt.imshow(large)
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存结果
cv2.imwrite('output_small.jpg', cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
cv2.imwrite('output_large.jpg', cv2.cvtColor(large, cv2.COLOR_RGB2BGR))
print("✅ 缩放结果已保存为 output_small.jpg 与 output_large.jpg")
