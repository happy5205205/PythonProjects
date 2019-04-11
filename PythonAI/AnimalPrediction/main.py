# _*_ coding: utf-8 _*_
"""
    时间：2018年11月28日
    作者:张鹏
    文件命：main.py
    功能：主程序
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PythonAI.AnimalPrediction import config
from PythonAI.AnimalPrediction import utils
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import  SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from mlxtend.classifier import StackingClassifier

def main():
    """
        主函数
    """
    # 数据加载
    raw_data = pd.read_csv(os.path.join(config.data_path, 'zoo.csv'),
                           usecols=config.all_cols)
    # 数据分割
    train_data, test_data = train_test_split(raw_data, test_size=1/4, random_state=10)

    # 查看数据
    utils.inspect_dataset(train_data, test_data)
    # utils.inspect_dataset(trian_data=train_data, test_data=test_data)

    print('\n===================== 特征工程 =====================')
    X_train, X_test = utils.transform_data(train_data, test_data)
    # print(X_train)
    # print('=====')
    # print(X_test)

    # 标签
    y_train = train_data[config.label_col[0]].values
    # print(y_train)
    y_test = test_data[config.label_col[0]].values
    # print('=====')
    # print(y_test)

    # 数据建模及验证

    sclf = StackingClassifier(classifiers=[KNeighborsClassifier(), DecisionTreeClassifier(), SVC()],
                              meta_classifier=LogisticRegression())

    print('\n===================== 数据建模及验证 =====================')
    model_name_param_dict = {'KNN':(KNeighborsClassifier(), {'n_neighbors':[5, 25, 55]}),
                             'LR':(LogisticRegression(), {'C':[0.01, 0.1, 1, 10]}),
                             'SVM':(SVC(), {'C': [0.01, 1, 100]}),
                             'DT':(DecisionTreeClassifier(criterion='entropy'), {'max_depth':[50, 100, 150]}),
                             'Stacking':(sclf, {'kneighborsclassifier__n_neighbors':[5, 25, 55],
                                                'svc__C':[0.01, 1, 100],
                                                'decisiontreeclassifier__max_depth': [50, 100, 150],
                                                'meta-logisticregression__C': [0.01, 1, 100]
                                                }),
                             'Adaboost':(AdaBoostClassifier(), {'n_estimators':[50, 100, 150, 200]}),
                             'GBDT': (GradientBoostingClassifier(),
                                      {'learning_rate': [0.01, 0.1, 1, 10, 100]}),
                             'RF': (RandomForestClassifier(),
                                    {'n_estimators': [100, 150, 200, 250]})
                             }
    result_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                             index=list(model_name_param_dict.keys()))
    for model_name, (model, param_range) in model_name_param_dict.items():
        _, best_acc, mean_duration = utils.train_test_model(X_train, y_train, X_test, y_test,
                                                            model_name, model, param_range)

        result_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
        result_df.loc[model_name, 'Time (s)'] = mean_duration
    result_df.to_csv(os.path.join(config.output_path,'result.csv'))

    # 模型及结果比较可视化
    print('\n===================== 模型及结果比较 =====================')
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    result_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[60, 100], ax=ax1,
                   title='Accuracy (%)', legend=False)
    ax2 = plt.subplot(1, 2, 2)
    result_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='Time (s)', legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path,'result.jpg'))
if __name__ == '__main__':
    main()
