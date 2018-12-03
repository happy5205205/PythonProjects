'''
    时间：2018年4月17日
    作者：张鹏
    内容：朴素贝叶斯、随机森林、GBDT
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# 文件读取
data_path = './data'
data_file = os.path.join(data_path, 'fruit_data_with_colors.txt')
fruit_df = pd.read_table(data_file)
print(fruit_df.head())
print('样本个数：{}'.format(len(fruit_df)))

# 创建目标标签和名称的字典
fruit_name_dict = dict(zip(fruit_df['fruit_label'], fruit_df['fruit_name']))
print(fruit_name_dict)

# 划分数据
X = fruit_df[['mass', 'width', 'height', 'color_score']]
# print('FFFFFFFFFFFFFF',X)
y = fruit_df['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

print('样本个数：{}条,其中训练样本个数：{}条，测试样本个数：{}条。'.format(len(fruit_df),
                                                 len(X_train), len(X_test)))
# 特征最大值最小值归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_sca = scaler.fit_transform(X_train)
X_test_sca = scaler.transform(X_test)

for i in range(4):
    print('归一化前，第{}维度的最大值为{:.3f}，最小值为{:.3f}'
          .format(i+1, X_train.iloc[i].max(), X_train.iloc[:, i].min()))
    print('归一化后，第{}维度的最大值为{:.3f}，最小值为{:.3f}'
          .format(i+1, X_train_sca[i].max(), X_train_sca[:, i].min()))
    print()

# 数据建模

# 1 朴素贝叶斯
print('************************朴素贝叶斯*******************************')
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_sca, y_train)
print('准确率：{}'.format(gnb.score(X_test_sca, y_test)))


# 2 随机森林
print('************************随机森林*******************************')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[10, 50, 100, 150, 200]}
clf = GridSearchCV(RandomForestClassifier(random_state=0),
                   parameters, cv=3, scoring='accuracy')
clf.fit(X_train_sca, y_train)
print('最优参数：{}'.format(clf.best_params_))
print('验证集最高得分：{}'.format(clf.best_score_))
print('测试集准确率：{}'.format(clf.score(X_test_sca, y_test)))

# 3 GBDT
print('*************************GBDT******************************')
from sklearn.ensemble import GradientBoostingClassifier
parameters = {'learning_rate': [0.01, 0.1, 1, 10, 100]}
clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=3 , scoring='accuracy')
clf.fit(X_train_sca, y_train)
print('最优参数:',clf.best_params_)
print('验证集最高得分：',clf.best_score_)
print('测试集准确率：',clf.score(X_test_sca, y_test))