'''
    时间：2018/03/06
    作者：张鹏
    任务：机器学习基础
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

# 1、 加载数据
data_path = './data'
data_file = os.path.join(data_path, 'fruit_data_with_colors.txt')
fruits_df = pd.read_table(data_file)
# print(fruits_df.head()) # 预览前五行数据
# print('样本个数：{}'.format(len(fruits_df)))

# 创建目标标签和名称的字典
fruits_name_dict = dict(zip(fruits_df['fruit_label'], fruits_df['fruit_name']))
# print(fruits_name_dict)

# 划分数据集
X = fruits_df[['mass', 'width', 'height', 'color_score']]
y = fruits_df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)
# print(y_train)
print('样本总数：{}，训练样本数：{}，测试样本数：{}'.format(len(fruits_df), len(y_train), len(y_test)))

# 2、可视化查看特征变量
# sns.pairplot(data= fruits_df, hue='fruit_name', vars=['mass', 'width', 'height', 'color_score'])
# plt.tight_layout()
# plt.show()

# 3、建立模型
from sklearn.neighbors import KNeighborsClassifier as kn
knn =kn(n_neighbors=5)

# 4、训练模型
knn.fit(X_train, y_train)

# 5、测试模型
y_pred = knn.predict(X_test) #c测试集中预测y的值

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred) # 真实值与预测值对比后，得到准确率
print('准确率：{}'.format(acc))

# 6、查看k值对结果的影响
k_range = range(1, 20)
acc_score = []
for k in k_range:
    knn = kn(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc_score.append(accuracy_score(y_test, knn.predict(X_test)))
print(acc_score)
print(len(acc_score))

plt.figure(figsize=(12, 6))
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.scatter(k_range, acc_score)
# plt.xticks([0, 5, 10, 15,20])
plt.show()
