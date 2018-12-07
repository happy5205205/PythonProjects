# _*_ coding: utf-8 _*_
"""
    时间：2018年12月6日
    作者：张鹏
    文件命：main.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from PythonAI.ImageData.FashionMnist import config, utils

IS_SAMPLE_EXP = True


def main():
    """
        主程序
    """
    # 加载数据
    print('加载训练数据。。。。')
    X_train, y_trian = utils.load_fashion_mnist_data(config.train_data_file)
    X_test, y_test = utils.load_fashion_mnist_data(config.test_data_file)

    # print(type(X_test))

    # 随机查看9张图片
    utils.polt_random_samples(X_train)

    # 训练数据特征提取
    print('训练数据特征提取。。。。')
    feats_train = utils.extract_feats(X_train)
    # 测试数据特征提取
    print('测试数据特征提取。。。。')
    feats_test = utils.extract_feats(X_test)

    # 特征归一化处理
    proc_feats_train, pro_feats_test = utils.do_feature_engineering(feats_train, feats_test)

    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')
    if IS_SAMPLE_EXP:
        # 耗时比较短，简单测试
        print('简单的Logistic Regression分类：')
        lf = LogisticRegression()
        lf.fit(proc_feats_train, y_trian)
        print('测试准确率{}'.format(lf.score(pro_feats_test, y_test)))
    else:
        # 耗时比较长
        print('多个模型交叉验证分类比较：')
        model_name_param_dict = {'KNN' : (KNeighborsClassifier(),
                                        {'n_neighbors' : [5, 15, 20]}),
                                 'LR' : (LogisticRegression(),
                                         {'C' : [0.01, 1, 10]}),
                                 'SVM' : (SVC(),
                                          {'C' : [0.01, 0.1, 1, 10]}),
                                 'DT' : (DecisionTreeClassifier(),
                                         {'max_depth' : [25, 55, 95]}),
                                 'Adaboost' : (AdaBoostClassifier(),
                                               {'n_estimators' : [100, 150, 200]}),
                                 'GBDT' : (GradientBoostingClassifier(),
                                           {'learning_rate' : [0.01, 1, 10]}),
                                 'RF' : (RandomForestClassifier(),
                                         {'n_estimators' : [100, 200, 250]})
                                 }
        # 比较结果的DataFrame
        result_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                                 index=list(model_name_param_dict.keys()))
        result_df.index.name = 'model'

        for model_name, (model, param_range) in model_name_param_dict.items():
            best_clf, best_acc, mean_duration = utils.train_and_test_model(proc_feats_train, y_trian,
                                                                           pro_feats_test, y_test,
                                                                           model_name, model, param_range)
            result_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
            result_df.loc[model_name, 'Time (s)'] = mean_duration
        result_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))

        # 模型及结果比较
        #         print('\n===================== 模型及结果比较 =====================')
        plt.figure(figsize=(10, 5))
        ax1= plt.subplot(1, 2, 1)
        result_df.plot(y=['Accuracy (%)'], kind = 'bar', ylim=[50, 100], ax= ax1, title='Accuracy (%)', legend=False)

        ax2 = plt.subplot(1, 2, 2)
        result_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='Time (s)', legend=False)
        plt.tight_layout()
        plt.savefig(os.path.join(config.output_path, 'pred_results.png'))
        plt.show()


if __name__ == '__main__':
    main()