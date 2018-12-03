'''
    时间：2018年4月16日
    任务：交叉验证
    作者：张鹏
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 数据读取
data_path = './data'
data_file = os.path.join(data_path, 'fruit_data_with_colors.txt')
fruit_df =pd.read_table(data_file)
print('样本个数{}'.format(len(fruit_df)))

# 创建目标标签和名称的字典
fruit_name_dict = dict(zip(fruit_df['fruit_label'], fruit_df['fruit_name']))
print(fruit_name_dict)

# 划分数据集
X = fruit_df[['mass', 'width', 'height', 'color_score']]
y = fruit_df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
# print(X.head())
# print(y.head())
# print('1111',X_train)
# print('2222',X_test)
# print('3333',y_train)
# print('4444',y_test)
print('样本总数:{}条,其中训练样本数{}条，测试样本数:{}'.format(len(fruit_df), len(X_train), len(X_test)))

# 特征归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled= scaler.transform(X_test)

for i in range(4):
    print('归一化前的最大值{:.3f},最小值{:.3f},'
          .format(X_train.iloc[:,i].max(), X_train.iloc[:,i].min()))
    print('归一化后的最大值{:.3f},最小值{:.3f},'
          .format(X_train_scaled[:,i].max(), X_train_scaled[:,i].min()))
    print()

# 交叉验证

# 单一超参数
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
k_range = [5, 10, 15, 20]
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=3)
    cv_score = np.mean(scores)
    print('k={}时，验证集上面的准确率={:.3f}'.format(k, cv_score))
    cv_scores.append(cv_score)

# 得到最好的模型进行测试集
best_k = k_range[np.argmax(cv_scores)]
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)
print('测试精准率为：{}'.format(best_knn.score(X_test_scaled,y_test)))

#  调用  validation_curve 绘制超参数对训练集和验证集的影响
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
c_range = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
train_scores, test_scores =validation_curve(SVC(), X_train_scaled, y_train,param_name='C',
                                            param_range=c_range,cv=3,scoring='accuracy')
print(train_scores)
print(test_scores)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 8))
plt.title('Validation Curve with SVM')
plt.xlabel('C')
plt.ylabel('Score')
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(c_range, train_scores_mean, label='Training score', color="darkorange", lw=lw)
plt.fill_between(c_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(c_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(c_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
# plt.show()

# 由图可知对SVM，C = 10 或者100 时模型最优
svm_model = SVC(C=10)
svm_model.fit(X_train_scaled, y_train)
print('SVM模型准确率:', svm_model.score(X_test_scaled, y_test))

# 多个超参数

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
paramets = {'max_depth':[3, 5, 7, 9], 'min_samples_leaf':[1, 2, 3, 4]}
clf =  GridSearchCV(DecisionTreeClassifier(), paramets, cv=3, scoring='accuracy')
clf.fit(X_train_scaled, y_train)

print('最优参数',clf.best_params_)
print('验证集最高得分：',clf.best_score_)

# 获得最优模型
best_model = clf.best_estimator_
print('测试集的准确率为：',best_model.score(X_test_scaled, y_test))

