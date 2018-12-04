"""
    practice1
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 图像数据基本操作

# 随机生成500x500的多维数组
# random_image = np.random.random([20, 20])
# print(random_image)
# plt.imshow(random_image,cmap='gray')
# plt.colorbar()
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
# cv2.imshow('gray', img_gray_data)
# cv2.imshow('rgb', img_rgb_data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 3. 像素值及访问操作
# 通过切片操作只显示某个通道上的图像数据
# OpenCV读取的图像数据通道排序为 BGR
# cv2.imshow('B',img_rgb_data[:, :, 0])
# cv2.imshow('G', img_rgb_data[:, :, 1])
# cv2.imshow('R', img_rgb_data[:, :, 2])
# cv2.imshow('BGR', img_rgb_data[:, :, :])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 使用split进行通道分割
b_data, g_data, r_data = cv2.split(img_rgb_data)
# cv2.imshow('B',b_data)
# cv2.imshow('G',g_data)
# cv2.imshow('R', r_data)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 合并通道
# bgr_data = cv2.merge((b_data, g_data, r_data))
# cv2.imshow('BGR',bgr_data)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 在图片上叠加一个蓝色方块
# img_rgb_data[10:110, 10:110,:]=[255, 255, 0]
# cv2.imshow('color with blue box', img_rgb_data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ROI
# ball = img_rgb_data[280:340, 330:390, :] # 截取图中某些部分
# cv2.imshow('ball', ball)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 4. 图像融合

# numpy和opencv中加法的区别
# x = np.uint8([250])
# y = np.uint8([10])
#
# print(cv2.add(x, y))  # 250 + 10 = 260 => 255
# print(x + y)          # 250 + 10 = 260 % 256 = 4

# img1 = cv2.imread('./images/python_ai.png')
# img2 = cv2.imread('./images/python_logo.png')
# img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]))
# dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
# cv2.imshow('dst',dst)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imwrite('./output/python_ai_logo.jpg',dst)

# 5. 色彩空间
# img_gray_data2 = cv2.imread('./images/messi.jpg', cv2.IMREAD_GRAYSCALE)
# img_bgr_data2 = cv2.cvtColor(img_gray_data, cv2.COLOR_GRAY2BGR)
#
# cv2.imshow('gray', img_gray_data2)
# cv2.imshow('bgr', img_bgr_data2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img_bgr_data3 = cv2.imread('./images/messi.jpg')
# img_hsv_data = cv2.cvtColor(img_bgr_data3, cv2.COLOR_BGR2HSV)
# cv2.imshow('bgr', img_bgr_data3)
# cv2.imshow('hsv', img_hsv_data)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 6. 颜色直方图
# hist, bins = np.histogram(img_gray_data.ravel(), bins=10)
# print(hist)
# print(bins)
#
# plt.hist(img_gray_data.ravel(), bins=50)
# plt.show()

# 彩色图像直方图
# img_bgr_data = cv2.imread('./images/messi.jpg')
#
# plt.figure(figsize=(15, 5))

# B通道 直方图
# ax1 = plt.subplot(131)
# ax1.hist(img_bgr_data[:, :, 0].ravel(),bins=50,color='b')
# # G通道 直方图
# ax2 = plt.subplot(132)
# ax2.hist(img_bgr_data[:, :, 1].ravel(), bins=50, color='g')
#
# # # R通道 直方图
# ax3 = plt.subplot(133)
#
# ax3.hist(img_bgr_data[:, :, 2].ravel(), bins=50, color='r')

# plt.show()

# 直方图均衡化
# wiki_img = cv2.imread('./images/wiki.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('wiki_img', wiki_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# plt.hist(wiki_img.ravel(), bins=256, range=[0, 255])
# plt.show()

# equ_wiki_img = cv2.equalizeHist(wiki_img)
# cv2.imshow('equ_wiki_img', equ_wiki_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# plt.hist(equ_wiki_img.ravel(), bins=256, range=[0,255])
# plt.show()

# 7. 图像滤波
# 7.1 中值滤波
# img = cv2.imread("./images/nosiy.jpg",cv2.IMREAD_GRAYSCALE)
# med1 = cv2.medianBlur(img, 3)
# med2 = cv2.medianBlur(img, 5)

# 图像对比
# cv2.imshow('img', img)
# cv2.imshow('med1', med1)
# cv2.imshow('med2', med2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 7.2 高斯滤波
# img = cv2.imread('./images/camera.jpg', cv2.IMREAD_GRAYSCALE)
# gas1 = cv2.GaussianBlur(img, (3, 3), 0) # 3 * 3
# gas2 = cv2.GaussianBlur(img, (5, 5), 0) # 5 * 5
# 图像对比
# cv2.imshow('img', img)
# cv2.imshow('gas1', gas1)
# cv2.imshow('gas2', gas2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 7.3 均值滤波
# img = cv2.imread('./images/camera.jpg', cv2.IMREAD_GRAYSCALE)
# mean1 = cv2.blur(img, (3, 3), 0) # 3x3
# mean2 = cv2.blur(img, (5, 5), 0) # 5x5

# 图像对比
# cv2.imshow('img', img)
# cv2.imshow('mean1', mean1)
# cv2.imshow('mean2', mean2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 边缘检测
img = cv2.imread('./images/messi.jpg', cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

# 图像对比
plt.figure(figsize=(10, 5))
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')
plt.xticks([])
plt.yticks([])

plt.show()