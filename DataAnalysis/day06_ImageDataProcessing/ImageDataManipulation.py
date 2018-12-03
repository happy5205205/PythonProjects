'''
    时间:2018年4月18日
    内容：图像数据的操作
    作者：张鹏
'''

import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from skimage import data
# 1 、skimage的图像数据

# 随机生成500*500的多维数组
# random_image = np.random.random([500,500])
# print(random_image)
# plt.imshow(random_image, cmap='gray')
# plt.colorbar()
# plt.show()


# 加载skimage中的coin数据
# coins = data.coins()
# print(type(coins), coins.dtype, coins.shape)
# plt.imshow(coins, cmap='gray')
# plt.colorbar()
# plt.show()

# cat = data.chelsea()
# print('图片的形状：', cat.shape)
# print('最大值{}，最小值{}'.format(cat.max(),cat.min()))
# plt.imshow(cat)
# plt.colorbar()
# plt.show()

# 在图形上方增加一个红色方块
# cat[10:110, 10:110, :] = [225, 0, 0]# [red green blue]
# plt.imshow(cat, cmap='gray')
# plt.imshow(cat)
# plt.colorbar()
# plt.show()


# 2 、数据类型和像素值
# 生成0-1间的2500个数据
# linear0 = np.linspace(0, 1, 2500).reshape((50, 50)) # 0-1之间50*50的数组
# 生成0-255间的2500个数据
# linear1 = np.linspace(0, 255, 2500).reshape((50, 50))
# print('linear0', linear0.dtype, linear0.min(), linear0.max())
# print('linear1', linear1.dtype, linear1.min(), linear1.max())
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(linear0, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(linear1, cmap='gray')
# plt.show()

from skimage import img_as_float, img_as_ubyte
# image =data.chelsea()
# image_float = img_as_float(image) # 像素范围0-1
# image_ubyte = img_as_ubyte(image) # 像素范围0-255
# print('type max min', image_float.dtype, image_float.max(), image_float.min())
# print('type max min', image_ubyte.dtype, image_ubyte.max(), image_ubyte.min())
# print('231/255=',231/255 ) # 验证0-255转换大到0-1

# 3 、显示图像

# image = data.camera()
# plt.figure(figsize=(10, 5))
# 使用不同的colormap
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='jet')

# plt.subplot(1, 2, 2)
# plt.imshow(image, cmap='gray')
# plt.show()

# 通过数组切片操作获得人脸区域
# face = image[80:160, 200:280]
# plt.figure(figsize=(10, 5))
# # 使用不同的color map
# plt.subplot(1, 2, 1)
# plt.imshow(face, cmap='jet')

# plt.subplot(1, 2, 2)
# plt.imshow(face, cmap='gray')
# plt.show()

# 4 图像I/O
from skimage import io
# image = io.imread('./images/balloon.jpg')
# print(type(image))
# plt.imshow(image)
# plt.show()
# 同时加载多张图片
# ic = io.imread_collection('./images/*.png')
# print(type(ic),'\n\n', ic)
# f ,axes = plt.subplots(nrows=1, ncols=len(ic), figsize=(20, 15))
# for i, image in enumerate(ic):
#     axes[i].imshow(image)
#     axes[i].axis('off')
# # 保存图像
# saving_img = ic[0]
# io.imsave('./images/zhangpeng.jpg', saving_img)
# plt.show()

# 5、 分割和索引
# color_image =data.chelsea()
# print(color_image.shape)
# plt.imshow(color_image)
#
# red_channel = color_image[:, :, 0] # 红色通道
# plt.imshow(red_channel)
# print(red_channel.shape)
# plt.show()

# 6 、色彩空间
import skimage
# RGB ---Gray
# gray_img = skimage.color.rgb2gray(color_image)
# plt.imshow(gray_img, cmap='gray')
# print(gray_img.shape)
# plt.show()

# 7 颜色直方图
from skimage import data
from skimage import exposure
#灰度图胭脂色直方图
# image = data.camera()
# print(image.shape)
# hist, bin_centers = exposure.histogram(image)
# plt.figure()
# plt.fill_between(bin_centers, hist)
# plt.ylim(0)
# plt.show()
# plt.imshow(image, cmap='gray')
# plt.show()

# 彩色图像直方图
# cat = data.chelsea()
# # R通道
# hist_r, bin_center_r= exposure.histogram(cat[:,:,0])
# # G通道
# hist_g, bin_center_g= exposure.histogram(cat[:,0,:])
# # B通道
# hist_b, bin_center_b= exposure.histogram(cat[0,:,:])
#
# plt.figure(figsize=(10, 5))

# R通道直方图
# ax =plt.subplot(1, 3, 1)
# plt.fill_between(hist_r,bin_center_r,facecolor='r')
# plt.ylim(0)

# G通道直方图
# ax1 =plt.subplot(1, 3, 2,sharey=ax)
# plt.fill_between(hist_g,bin_center_g,facecolor='g')
# plt.ylim(0)

# B通道直方图
# ax2 =plt.subplot(1, 3, 3,sharey=ax)
# plt.fill_between(hist_b,bin_center_b,facecolor='b')
# plt.ylim(0)
# plt.show()

# 8、 对比度
# 原图像
# image = data.camera()
# hist, bin_center =exposure.histogram(image)

# 改变对比度
# image中小于10的像素设置为0， 大于180的像素设置为255
# high_constrast =exposure.rescale_intensity(image, in_range=(10, 180))
# hist2, bin_center2 =exposure.histogram(high_constrast)
# 图像对比
# fig, (ax_1, ax_2)=plt.subplots(ncols=2, figsize=(10, 5))
# ax_1.imshow(image, cmap='gray')
# ax_2.imshow(high_constrast,cmap='gray')
#
# fig, (ax_hist1, ax_hist2)=plt.subplots(ncols=2, figsize=(10, 5), sharey=True)
# ax_hist1.fill_between(bin_center,hist)
# ax_hist2.fill_between(bin_center2,hist2)
# plt.ylim(0)
# # plt.show()`
#
# # 直方图均衡化
# equalied = exposure.equalize_hist(image)
# hist3, bin_center3 = exposure.histogram(equalied)

# 图像对比
# fig, (ax_3, ax_4)=plt.subplots(ncols=2, figsize=(10, 5))
# ax_3.imshow(image, cmap='gray')
# ax_4.imshow(equalied,cmap='gray')
#
# fig, (ax_hist3, ax_hist4)=plt.subplots(ncols=2, figsize=(10, 5), sharey=True)
# ax_hist3.fill_between(bin_center,hist)
# ax_hist4.fill_between(bin_center3,hist3)
# plt.ylim(0)
# plt.show()


# 9、图像滤波

# 中值滤波
# from skimage import data
# from skimage.morphology import disk
# from skimage.filters.rank import median
# img =data.camera()
# med1 = median(img, disk(3))
# med2 = median(img, disk(5))
# # 图像对比
# fig, (ax_1, ax_2, ax_3) = plt.subplots(ncols=3,figsize= (15, 10))
# ax_1.imshow(img, cmap='gray')
# ax_2.imshow(med1, cmap='gray')
# ax_3.imshow(med2, cmap='gray')
# plt.show()
#
# # 高斯滤波
# from skimage import data
# from skimage.filters import gaussian
#
# img1 = data.camera()
# gas1 = gaussian(img1, sigma=3)
# gas2 = gaussian(img1, sigma=5)
# # 图像对比
# fig, (ax_4, ax_5,ax_6) = plt.subplots(ncols=3, figsize=(15, 10))
# ax_4.imshow(img1, cmap='gray')
# ax_5.imshow(gas1, cmap='gray')
# ax_6.imshow(gas2, cmap='gray')
# plt.show()
#
# # 均值滤波
# from skimage import data
# from skimage.filters.rank import mean
# img2 = data.camera()
# mean1 = mean(img2, disk(3))
# mean2 = mean(img2, disk(5))
#
# # 图像对比
# fig, (ax_7, ax_8, ax_9,) = plt.subplots(ncols=3, figsize = (15, 10))
# ax_7.imshow(img2, cmap='gray')
# ax_8.imshow(mean1, cmap='gray')
# ax_9.imshow(mean2, cmap='gray')
# plt.show()

# 10 边缘检测
from skimage.filters import prewitt, sobel
image = data.camera()
edge_prewitt = prewitt(image)
edge_sobel = sobel(image)

fig, (ax_1, ax_2) = plt.subplots(ncols=2, figsize=(15, 10))
ax_1.imshow(edge_prewitt, cmap='gray')
ax_2.imshow(edge_sobel, cmap='gray')
plt.show()


