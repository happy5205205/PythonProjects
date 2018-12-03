'''
    时间：2018/03/27
    功能：预测吸毒人员犯罪
    作者：张鹏
    版本：V1.0
'''


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
def model_train():
    pass

def main():
    #文件读取

    # 指定文件路径
    data_path = './data'
    data_file = os.path.join(data_path, 'kuangbiao_1210.csv')
    poison_data = pd.read_csv(data_file)
    print(len(poison_data))

    # 特征处理
    print(len(poison_data.columns[2:].values))

    #标签处理
    # label_enc = LabelEncoder()
    # X_train = label_enc.fit(poison_data['JA5_NL'])
    # print(label_enc.classes_)

    # 划分数据
    X = poison_data[poison_data.columns[2:].values]
    print(X.head())
    y= poison_data['FZBQ']
    print(y.head())
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

if __name__ == '__main__':
    main()
