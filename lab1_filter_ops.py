# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('test.jpg')
if image is None:
    print("⚠️ 未找到 test.jpg，请检查文件路径！")
    exit()

# 转为灰度图（滤波和边缘检测常在灰度图上进行）
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1️⃣ 均值滤波（去除细小噪声）
mean_blur = cv2.blur(gray, (5, 5))

# 2️⃣ 高斯滤波（更柔和的模糊效果）
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

# 3️⃣ 中值滤波（对椒盐噪声非常有效）
median_blur = cv2.medianBlur(gray, 5)

# 4️⃣ Sobel 边缘检测（提取水平与垂直边）
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# 5️⃣ Canny 边缘检测（常用的边缘检测算法）
canny = cv2.Canny(gray, 100, 200)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1); plt.title("Original Gray"); plt.imshow(gray, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 2); plt.title("Mean Blur"); plt.imshow(mean_blur, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 3); plt.title("Gaussian Blur"); plt.imshow(gaussian_blur, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 4); plt.title("Median Blur"); plt.imshow(median_blur, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 5); plt.title("Sobel Edge"); plt.imshow(sobel, cmap='gray'); plt.axis('off')
plt.subplot(2, 3, 6); plt.title("Canny Edge"); plt.imshow(canny, cmap='gray'); plt.axis('off')
plt.tight_layout()
plt.show()

# 保存结果
cv2.imwrite('blur_mean.jpg', mean_blur)
cv2.imwrite('blur_gaussian.jpg', gaussian_blur)
cv2.imwrite('blur_median.jpg', median_blur)
cv2.imwrite('edge_sobel.jpg', sobel)
cv2.imwrite('edge_canny.jpg', canny)
print("✅ 滤波与边缘检测结果已保存：blur_*.jpg, edge_*.jpg")
