'''
    时间：2018年4月16日
    学习内容：特征工程
'''
# 特征工程
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

# 数据加载
data_path = './data'
data_file = os.path.join(data_path, 'fruit_data_with_colors.txt')
fruit_df = pd.read_table(data_file)
# print(fruit_df.head())

# 样本个数
print('样本个数{}'.format(len(fruit_df)))

# 创建目标标签和名称的字典
fruit_name_dict = dict(zip(fruit_df['fruit_label'], fruit_df['fruit_name']))
print(fruit_name_dict)

# 划分数据集
X = fruit_df[['mass', 'width', 'height', 'color_score']]
y = fruit_df['fruit_label']

X_train, X_test, y_train , y_test = train_test_split(X, y , test_size=1/4, random_state=0)
print(X_train)
# print(y)

print('数据样本个数{}, 训练样本个数{}, 测试样本个数{}'.format(len(X), len(X_train), len(X_test)))

# 特征归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('X_train_scaled',X_train_scaled)

for i in range(4):
    print('归一化前，训练数据第{}维特征最大值：{:.3f}, 最小值：{:.3f}'
          .format(i+1, X_train.iloc[:, i].max(), X_train.iloc[:, i].min()))
    print('归一化后，训练数据第{}维特征最大值:{:.3f}, 最小值：{:.3f}'
          .format(i+1, X_train_scaled[:, i].max(), X_train_scaled[:, i].min()))
    print()

# 比较归一化前后对KNN 模型的影响
from sklearn.neighbors import KNeighborsClassifier
knn =KNeighborsClassifier(n_neighbors=5)

# 在未归一化的数据进行训练并测试
knn.fit(X_train, y_train)
# print('归一化前特征的测试，准确率：{:.3f}'.format(knn.score(X_test, y_test)))

# 在归一化的数据进行训练并测试
knn.fit(X_train_scaled, y_train)
# print('归一化后特征的测试，准确率：{:.3f}'.format(knn.score(X_test_scaled, y_test)))

# 标签编码和独热编码
# 随机生成有序型特征和类别特征作为例子
X_train_1 = np.array([['male', 'low'],
                  ['female', 'low'],
                  ['female', 'middle'],
                  ['male', 'low'],
                  ['female', 'high'],
                  ['male', 'low'],
                  ['female', 'low'],
                  ['female', 'high'],
                  ['male', 'low'],
                  ['male', 'high']])

X_test_1 = np.array([['male', 'low'],
                  ['male', 'low'],
                  ['female', 'middle'],
                  ['female', 'low'],
                  ['female', 'high']])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# 在训练集上面进行编码操作
label_enc1 = LabelEncoder()# 将male和female首先进行数字编码
one_hot_enc = OneHotEncoder()# 将数字编码装换成独热编码

label_enc2 = LabelEncoder()# 将high、middle和low进行数字编码

tr_feat1_tmp = label_enc1.fit_transform(X_train_1[:, 0]).reshape(-1, 1)# reshape(-1, 1)保证为一维向量
tr_feat1 = one_hot_enc.fit_transform(tr_feat1_tmp)
tr_feat1 = tr_feat1.todense()

tr_feat2_tmp = label_enc2.fit_transform(X_train_1[:, 1]).reshape(-1, 1)
tr_feat2 = one_hot_enc.fit_transform(tr_feat2_tmp)
tr_feat2 = tr_feat2.todense()

X_train_enc = np.hstack((tr_feat1, tr_feat2))
print(X_train_enc)

# 在测试集上面进行编码操作
te_feat1_tmp = label_enc1.fit_transform(X_test_1[:, 0]).reshape(-1, 1) # reshape(-1, 1)保证为一维向量
te_feat1 = one_hot_enc.fit_transform(te_feat1_tmp)
te_feat1 = te_feat1.todense()

te_feat2 = label_enc2.fit_transform(X_test_1[:, 1]).reshape(-1, 1)
X_test_enc = np.hstack((te_feat1, te_feat2))
print(X_test_enc)

