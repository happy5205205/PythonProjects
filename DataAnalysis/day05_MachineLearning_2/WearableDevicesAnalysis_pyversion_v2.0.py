'''
    时间：2018-4-17

    项目名称：实战案例4-2：根据可穿戴设备识别用户行为
    版本：python版本
    项目描述：
            用户行为识别数据集是通过采集30天用户的行为创建的。数据是由绑定在用户腰间的智能手机记录的，该智能手机内嵌有传感器。
        创建该数据集的目的是用于识别/分类6组不同的用户行为。
            数据集中的用户是由19-48岁间的30个志愿者组成的。戴有智能手机（Samsung Galaxy S II）的每个志愿者会做出6个行
        为（WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING）。
        通过智能手机的加速计和陀螺仪能够以50Hz的频率采集3个方向的加速度和3个方向的角速度。采集后的数据集随机分为两部分，
        70%用于模型的训练，30%用于模型的验证。
            传感器信号已经预处理去除了噪声，并且在固定时间窗口（2.56s）内进行采样，
        每两个窗口间有50%的重叠部分（每个窗口有128个数据）。每个时间窗口同时提供时间和频率上的统计数据作为特征。
    数据集描述:
        Kaggle提供的数据集。数据集包含训练集（train.csv）和测试集（test.csv），形式均为CSV文件。
        每条记录提供有以下数据
        3个方向的加速度，估计的身体加速度，3个方向的角速度。最终是561维的向量。
        对应的标签
        志愿者编号
    任务：1 数据查看
         2 数据建模及验证
         3 模型及结果比较
    版本：python版本
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 防止中文乱码
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False

# 交叉验证固定几折
cv_val = 3

def model_train(X_train, y_train, X_test, y_test, model_config):
    model = model_config[0]
    paramaters = model_config[1]
    if paramaters is not None:
        clf = GridSearchCV(model, paramaters, cv=cv_val, scoring='accuracy')
        clf.fit(X_train, y_train)
        print('最优参数为：{}\n验证集准确率最高为：{}'.format(clf.best_params_, clf.best_score_))

    else:
        # clf = GaussianNB()

        model.fit(X_train, y_train)
        clf =model
        print('最优参数为：{}\n 验证机准确率最高为：{}'.format(clf.best_params_, clf.best_score_))

    test_acc= clf.score(X_test, y_test)
    print('测试集的准确率为:',test_acc)

    return test_acc

def main():

    # 数据读取
    data_path = './data'
    train_datafile = os.path.join(data_path, 'train.csv')
    test_datafile = os.path.join(data_path, 'test.csv')
    train_data = pd.read_csv(train_datafile)
    test_data = pd.read_csv(test_datafile)
    print('训练数据有{}条，测试数据有{}条'.format(len(train_data), len(test_data)))

    # 特征处理
    feature_name = train_data.columns[:-2].tolist()
    # print(feature_name)
    X_train = train_data[feature_name].values
    X_test = test_data[feature_name].values

    # 标签处理
    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(train_data['Activity'].values)
    y_test = label_enc.fit_transform(test_data['Activity'].values)
    # print('aaa',label_enc.classes_)
    for i in range(len(label_enc.classes_)):
        print('{}对应的标签为{}'.format(label_enc.classes_[i], i))
    # print(y_test)
    # print(y_train)

    # 归一化数据两种方法MaxMin和standard scaler
    # 1 MaxMin
    max_min = MinMaxScaler()
    X_train_max_min_scaler = max_min.fit_transform(X_train)
    X_test_max_min_scaler = max_min.transform(X_test)

    # 2 standard scaler
    std_scaler = StandardScaler()
    X_train_std_scaler = std_scaler.fit_transform(X_train)
    X_test_std_scaler = std_scaler.transform(X_test)

    model_dict = {'KNN':    (KNeighborsClassifier(),            {'n_neighbors':[10, 50 ,100]}),
                 'LR':      (LogisticRegression(),              {'C': [0.1, 1, 10]}),
                 'SVM':     (SVC(),                             {'C': [100, 1000, 10000]}),
                 'DT':      (DecisionTreeClassifier(),          {'max_depth': [50,100,150]}),
                 'GBN':     (GaussianNB(),None),
                 'RF':      (RandomForestClassifier(),          {'n_estimators': [10, 20, 30]}),
                 'GDBT':    (GradientBoostingClassifier(),      {'learning_rate': [0.1, 1, 0.5]})
                  }
    results_df = pd.DataFrame(columns=['Not Scaled (%)', 'Min Max Scaled (%)', 'Std Scaled (%)'],
                              index=list(model_dict.keys()))
    results_df.index.name = 'mode'

    for model_name, mode_config in model_dict.items():
        # print('训练模型：', model_name)
        print('************************{}模型*******************************'.format(model_name))
        print('没有归一化')
        acc1 = model_train(X_train, y_train, X_test, y_test, mode_config)
        print('MaxMin归一化')
        acc2 = model_train(X_train_max_min_scaler, y_train, X_test_max_min_scaler, y_test, mode_config)
        print('standard scaler归一化')
        acc3 = model_train(X_train_std_scaler, y_train, X_test_std_scaler, y_test, mode_config)
        results_df.loc[model_name] = [acc1 * 100, acc2 * 100, acc3 * 100]
        # results_df.loc[model_name] = [acc2 * 100, 100, 100]
        print()
    # 结果写入文件保存
    results_df.to_csv('./pred_results.csv')

    # 将结果绘制成图
    plt.figure(figsize = (10, 5))
    results_df.plot(kind = 'bar')
    plt.ylim([75, 105])
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('./pred_results.jpg')
    plt.show()


if __name__ == '__main__':
    main()