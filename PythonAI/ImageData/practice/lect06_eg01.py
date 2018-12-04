"""
    practice1
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 图像数据基本操作

# 随机生成500x500的多维数组
random_image = np.random.random([20, 20])
# print(random_image)
plt.imshow(random_image,cmap='gray')
plt.colorbar()
# plt.show()

# 加载灰度图像数据
img_gray_data = cv2.imread('./images/messi.jpg', cv2.IMREAD_GRAYSCALE)
# print(img_gray_data)
# print('数据类型{}'.format(type(img_gray_data)))
# print('数据信息{}'.format(img_gray_data.info))
# print('数组类型{}'.format(img_gray_data.dtype))
# print('数组形状{}'.format(img_gray_data.shape))
# print('数组最大值{}，最小值{}'.format(img_gray_data.max(),img_gray_data.min()))

# 加载RGB图像数据
img_rgb_data = cv2.imread('./images/messi.jpg', cv2.IMREAD_ANYCOLOR)
# print(img_rgb_data)
# print('数据类型{}'.format(type(img_rgb_data)))
# print('数组类型{}'.format(img_rgb_data.dtype))
# print('数组形状{}'.format(img_rgb_data.shape))
# print('数组最大值{}，最小值{}'.format(img_rgb_data.max(), img_rgb_data.min()))

# 2. 显示图像
cv2.imshow('gray', img_gray_data)
cv2.imshow('rgb', img_rgb_data)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. 像素值及访问操作
