# _*_ coding:utf-8 _*_
"""
    常用的图像特征
"""

import cv2

import numpy as np

# 1. 颜色特征
img_gray_data = cv2.imread('./images/messi.jpg', cv2.IMREAD_GRAYSCALE)
hist, bins = np.histogram(img_gray_data.ravel(), bins=50)
print(hist,end='\n')
print(bins)

# 2. SIFT 特征
# img = cv2.imread('./images/messi.jpg', cv2.IMREAD_GRAYSCALE)
# sift = cv2.xfeatures2d.SIFT_create()
# kp, desc = sift.detectAndCompute(img, None)
# img_w_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('img_w_kp',img_w_kp)
# cv2.waitKey()
# cv2.destroyAllWindows()

img = cv2.imread('./images/messi.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
kp, desc = sift.detectAndCompute(img, None)
img_w_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('img_w_kp', img_w_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()