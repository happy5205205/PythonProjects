'''
    时间：2018年4月19日
    内容：常用的图像特征
    作者：张鹏
'''

# 1 颜色特征
from skimage import data, img_as_float, exposure
# 如果需要使用参数nbins, 需要将图像数据从[0, 255]转换成[0, 1]
camera = img_as_float(data.camera())

# 颜色直方图
hist, bin_center =exposure.histogram(camera, nbins=10)
print(hist)
print(bin_center)

# 2 SIFT 特征 (DAISY特征)
from skimage.feature import daisy
import matplotlib.pyplot as plt
daisy_feat, daisy_img = daisy(camera, step=180, radius=58, rings=2, histograms=6, visualize=True)
print(daisy_feat.shape)
plt.imshow(daisy_img)
plt.show()

# 3 skimage -- HOG 特征
from skimage.feature import hog
hot_feat, hot_img = hog(camera, visualise=True)
# print(hot_feat.shape)
plt.imshow(hot_img)
plt.show()