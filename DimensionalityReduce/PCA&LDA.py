# -*- coding: utf-8 -*-
"""
    created on: 2018-9-27
    @author: zhangpeng
    introduce：PCA和LDA降维
"""

# 生成三类三维特征的数据
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_classification

X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3,
                           n_informative=2, n_clusters_per_class=1, class_sep=0.5, random_state=10)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)
plt.show()


# 使用PCA降维到二维的情况，注意PCA无法使用类别信息来降维
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()

# 使用LDA降维到二维的情况

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X_lda_new = lda.transform(X)
plt.scatter(X_lda_new[:, 0], X_lda_new[:, 1], marker='o', c=y)
plt.show()