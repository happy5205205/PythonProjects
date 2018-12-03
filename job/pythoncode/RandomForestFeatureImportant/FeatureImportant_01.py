# _*_ coding:utf_8 _*_
"""
    随机森林做特征选择
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


import pandas as pd
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header = None)
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
              'Alcalinity of ash', 'Magnesium', 'Total phenols',
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
# print(df.head())

# 查看数据有几类class label
print(np.unique(df['Class label']))
# 查看数据信息
# print(df.info())
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
# print(x[0:2,:])
# print(y[0])
x_train, y_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
feat_label = df.columns[1:]
print(feat_label)
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(x_train,y_train)

# 随机森林就训练好后，其中已经把特征的重要性评估也做好了，可以拿出来看下
importance = forest.feature_importances_
indices = np.argsort(importance)[::1]
for f in range(x_train.shape[1]):
      print("%2d) %-*s %f" % (f + 1, 30, feat_label[indices[f]], importance[indices[f]]))



