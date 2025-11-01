# -*- coding: utf-8 -*-
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

# 读取图像
img_path = 'test.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"❌ Image not found: {img_path}")
    exit()

# 转为 RGB 以便 matplotlib 显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1) 固定角度旋转（90, 180, 270）使用 cv2.rotate
rot90 = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
rot180 = cv2.rotate(img_rgb, cv2.ROTATE_180)
rot270 = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)

# 2) 任意角度旋转（保持中心、不裁剪可用边界扩展，下面示例为 30 度）
def rotate_arbitrary(image, angle_deg):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # 计算旋转后图像边界大小以避免裁剪
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    # 调整平移以使图像居中
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

rot30 = rotate_arbitrary(img_rgb, 30)

# 3) 翻转（水平、垂直、同时）
flip_h = cv2.flip(img_rgb, 1)   # 水平翻转
flip_v = cv2.flip(img_rgb, 0)   # 垂直翻转
flip_both = cv2.flip(img_rgb, -1)  # 水平+垂直

# 4) 显示：把一些结果放在同一窗口展示
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Rot 90")
plt.imshow(rot90)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Rot 180")
plt.imshow(rot180)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Rot 270")
plt.imshow(rot270)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Rot 30° (arbitrary)")
plt.imshow(rot30)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Flip H / V / Both (samples below)")
# 为了不占太多空间，只显示水平翻转（可在保存中查看其他两张）
plt.imshow(flip_h)
plt.axis('off')

plt.tight_layout()
plt.show()

# 5) 保存结果到文件
cv2.imwrite('rot90.jpg', cv2.cvtColor(rot90, cv2.COLOR_RGB2BGR))
cv2.imwrite('rot180.jpg', cv2.cvtColor(rot180, cv2.COLOR_RGB2BGR))
cv2.imwrite('rot270.jpg', cv2.cvtColor(rot270, cv2.COLOR_RGB2BGR))
cv2.imwrite('rot30.jpg', cv2.cvtColor(rot30, cv2.COLOR_RGB2BGR))
cv2.imwrite('flip_h.jpg', cv2.cvtColor(flip_h, cv2.COLOR_RGB2BGR))
cv2.imwrite('flip_v.jpg', cv2.cvtColor(flip_v, cv2.COLOR_RGB2BGR))
cv2.imwrite('flip_both.jpg', cv2.cvtColor(flip_both, cv2.COLOR_RGB2BGR))

print("✅ Rotation & flip outputs saved: rot90.jpg, rot180.jpg, rot270.jpg, rot30.jpg, flip_h.jpg, flip_v.jpg, flip_both.jpg")
