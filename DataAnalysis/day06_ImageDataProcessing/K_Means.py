'''
    时间：2018年4月19日
    内容：K-Means聚类及图像压缩
    作者：张鹏
'''
import numpy as np
from skimage import img_as_ubyte
from skimage import io
import matplotlib.pyplot as plt

original_img =io.imread('./images/ColorfulBird.jpg')
original_img = img_as_ubyte(original_img)
# print('1111',original_img)
# print('图像维度：', original_img.shape)
plt.imshow(original_img)
# plt.show()

height, width, depth = original_img.shape
print(height, width, depth)
# 将图像数据点平铺
# 每个数据点为一个三维的样本
pixel_sample = np.reshape(original_img, (height * width, depth))
# print(pixel_sample[:10,:])
# print('pixel_sample:',pixel_sample.shape)

from sklearn.cluster import KMeans

# 压缩后图片包含的颜色个数，即为聚类的个数
k = 5
kmeans = KMeans(n_clusters=k, random_state=0)
# 训练模型
kmeans.fit(pixel_sample)
# 找到每个3维像素点对应的聚类中心
cluster_assignments = kmeans.predict(pixel_sample)

# 每个样本聚类的结果是对应的聚类中心的索引
print(set(cluster_assignments))

# 聚类中心值
cluster_centers = kmeans.cluster_centers_
print(cluster_centers)
print(cluster_centers.shape)

# 压缩图片
compressed_img = np.zeros((height, width, depth), dtype=np.uint)

# 遍历每个像素点，找到聚类中兴对应的像素值
pixel_count = 0
for i in range(height):
    for j in range(width):
        # 获取像素点的聚类中心索引
        cluster_idx = cluster_assignments[pixel_count]
        # 获取聚类中心点上面的像素值
        cluster_value = cluster_centers[cluster_idx]
        # 赋值
        compressed_img[i][j] = cluster_value
        pixel_count += 1
io.imsave('./images/compressed.jpg', compressed_img)

# 对比压缩前后图像
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Compressed Image')
plt.imshow(compressed_img, cmap='gray')

# plt.show()