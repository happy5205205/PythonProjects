'''
    时间：2018/03/06
    作者：张鹏
    作业：几种算法的模型KNN,线性回归，逻辑回归，SVM，决策树
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载数据集
data_path = './data'
data_file = os.path.join(data_path, 'fruit_data_with_colors.txt')
fruits_df = pd.read_table(data_file)

# 1、KNN 邻近算法
print('*******************************KNN*******************************')
X = fruits_df[['mass', 'width', 'height', 'color_score']]
y = fruits_df['fruit_label']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# 建立模型
knn = KNeighborsClassifier(5)

# 训练模型
knn.fit(X_train, y_train)

# 验证模型
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('KNN模型精准率：{}'.format(acc))

print('*******************************线性回归*******************************')
# 2、线性回归
# 人工生成用于测试回归的数据集
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

plt.figure()
plt.title('Sample regression problem with one input variable')

# 每个样本只有一个变量
X_R1, y_R1 = make_regression(n_samples=100, n_features=1,
                             n_informative=1, bias=150,
                             noise=30, random_state=0)
plt.scatter(X_R1, y_R1, marker='o', s=50)
plt.show()

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state=0)

# 调用线性回归模型
linreg = LinearRegression()

# 训练模型
linreg.fit(X_train, y_train)

# 输出结果
print('线性模型的系数（w）:{}'.format(linreg.coef_))
print('线性模型的常数项（b)：{}'.format(linreg.intercept_))
print('训练集中R-square得分：{:.3f}'.format(linreg.score(X_train, y_train)))
print('测试集中R-square得分：{:.3f}'.format(linreg.score(X_test, y_test)))

# 逻辑回归
from sklearn.linear_model import LogisticRegression

X = fruits_df[['width', 'height']]
y = fruits_df['fruit_label'].copy()

# 将不是apple的标签设置为0
y[y != 1 ] = 0

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

# 不同的C值
c_value = [0.1, 1, 100]

for c_value in c_value:
    # 建立模型
    lr_model = LogisticRegression(C = c_value)
    #训练模型
    lr_model.fit(X_train, y_train)
    # 验证模型
    y_pred = lr_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('C={}, 模型精准率为：{}'.format(c_value, acc))

print('*******************************SVM*******************************')
# SVM
from sklearn.svm import SVC

X = fruits_df[['width', 'height']]
y = fruits_df['fruit_label'].copy()

# 将不是apple的标签设置为0
y[y != 1] = 0

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=0)

c_value = [0.001, 1, 100]
for c_value in c_value:
    # 建立模型
    svm_model = SVC(C=c_value)

    # 训练模型
    svm_model.fit(X_train, y_train)

    # 验证模型
    y_pred = svm_model.predict(X_test)
    acc= accuracy_score(y_test, y_pred)
    print('C={},模型精准率为：{}'.format(c_value, acc))


print('*******************************决策树*******************************')
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

max_depth_values = [2, 3, 4]
for max_depth_values in max_depth_values:
    # 建立模型
    dt_model = DecisionTreeClassifier(max_depth=max_depth_values)

    # 训练模型
    dt_model.fit(X_train, y_train)

    print('max_depth=', max_depth_values)
    print('训练集上的准确率为：{:.3f}'.format(dt_model.score(X_train, y_train)))
    print('测试集的准确率为: {:.3f}'.format(dt_model.score(X_test, y_test)))
    print()