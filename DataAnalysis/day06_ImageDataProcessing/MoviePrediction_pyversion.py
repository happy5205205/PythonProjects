# -*- coding: utf-8 -*-

"""
    时间：2018年4月20日
    作者：张鹏
    版本：python版本
    名称：根据海报预测电影分类
    项目描述：
        电影海报是获取电影内容和类型的途径之一。用户可以通过海报的一些信息（如：颜色，演员的表情等）
        推测出其所属的类型（恐怖片，喜剧，动画等）。研究表明图像的颜色是影响人类感觉（心情）的因素之一，
        在本次项目中，我们会通过海报的颜色信息构建模型，并对其所属的电影类型进行推测。

    项目任务：
        1、数据查看几及处理
        2、特征表示
        3、数据建模及验证
        4、数据预测
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import exposure, io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# 解决matplotlib中乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 几折交叉验证
cv_var =3
# 指定数据路径
dataset_path = './data'
csv_fielpath = os.path.join(dataset_path, 'MovieGenre.csv')
poster_path = os.path.join(dataset_path, 'SampleMoviePosters')

def train_model(X_train, y_train, X_test, y_test, config):
    model_name = config[0]
    paramaters = config[1]
    if paramaters is not None:
        clf = GridSearchCV(model_name, paramaters, cv=cv_var, scoring='accuracy')
        clf.fit(X_train, y_train)
        print('最优参数:{},\n测试集准确为:{}'
              .format(clf.best_params_, clf.best_score_))
    else:

        model_name.fit(X_train, y_train)
        clf =model_name
        # print('可知{}模型中，最优参数为：{},测试集准确率最高为：{}'
        #       .format(mode_name, clf.best_params_, clf.best_score_))

    test_acc = clf.score(X_test, y_test)
    print('测试机的准确率为：{}\n'.format(test_acc))
    return test_acc, clf

def extract_hist_feat(img_path, nbins=50, as_grey=True):
    image_data = io.imread(img_path, as_grey=as_grey)
    #直方图均衡化
    eq_image_data = exposure.equalize_hist(image_data)
    if as_grey:
        hist_feat, _ = exposure.histogram(eq_image_data, nbins=nbins)
    else:
        pass
    norm_hist_feat =hist_feat / sum(hist_feat)
    return norm_hist_feat

def main():
    # 数据查看及处理
    print('\n===================== 任务1. 数据查看及处理 =====================')
    movie_df = pd.read_csv(csv_fielpath, encoding = 'ISO-8859-1',
                           usecols =['imdbId', 'Title', 'IMDB Score', 'Genre'])
    print(movie_df.head(2))
    movie_df['Single_Genre'] = movie_df['Genre'].str.split('|', expand= True)[0]
    print('movie_df\n', movie_df.head(2))
    # 创建一个DataFrame处理图像数据
    poster_df = pd.DataFrame(columns=['imdbId', 'img_path'])
    # print(poster_df)
    poster_df['img_path'] = os.listdir(poster_path)
    #这样写也可以，但是在后面合并是一个是字符串类型一个是数字类型，所以应转换成数字类型
    # poster_df['imdbId'] = poster_df['img_path'][0:-4]
    poster_df['imdbId'] =poster_df['img_path'].str[:-4].astype('int')
    print('poster_df\n', poster_df.head(2))

    # 将csv文件和图像文件进行合并
    data_df = movie_df.merge(poster_df, on='imdbId', how='inner')
    print('data_df\n',data_df.head(2))
    data_df.drop_duplicates(subset='imdbId', inplace=True)
    print('一共有{}条数据。'.format(len(data_df)))

    # 查看各电影的电影类型总数
    print('各类型电影的数量\n',data_df.groupby('Single_Genre').size().sort_values(ascending=False).head(3))
    # 可视化各类型电影数量
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sns.countplot(x='Single_Genre', data=data_df)
    plt.title('各类型电影数量统计')
    plt.xticks(rotation='vertical')
    plt.xlabel('电影类型')
    plt.ylabel('数量')
    # plt.show()
    # 减少分类，将电影类型分为三类
    cond = (data_df['Single_Genre'] != 'Comedy') & (data_df['Single_Genre'] != 'Drama')
    data_df.loc[cond, 'Single_Genre'] = 'Other'
    plt.subplot(1,2,2)
    sns.countplot(x='Single_Genre', data=data_df)
    plt.title('合并后各类型电影数量统计')
    plt.xticks(rotation='vertical')
    plt.xlabel('电影类型')
    plt.ylabel('数量')
    # plt.show()
    print('处理后数据预览\n{}'.format(data_df.head()))
    # print(data_df['Single_Genre']=='Other')
    print('\n===================== 任务2. 特征表示 =====================')
    # 对数据集中每张图片进行特征提取
    n_feat_dim = 100
    n_sample = len(data_df)

    # 初识化特征矩阵
    X = np.zeros((n_sample, n_feat_dim))
    print('X.shape\n', X.shape)
    for i, r_data in data_df.iterrows():
        if (i + 1) % 100 ==0:
            print('正在提取特征, 已完成{}张海报'.format(i + 1))
        img_path = os.path.join(poster_path, r_data['img_path'])
        hist_feat =extract_hist_feat(img_path, n_feat_dim)
        X[i:] = hist_feat.copy() # 之前写成这样X[i:]报错
        # print(X.shape)
        # print(type(X))
    print()
    # 获取标签名称
    target_name = data_df['Single_Genre'].values
    # print(target_name)
    from sklearn.preprocessing import LabelEncoder
    lab_enc = LabelEncoder()
    y = lab_enc.fit_transform(target_name)
    # print(y)
    # print('lab_enc = LabelEncoder()\n',lab_enc.classes_)
    for i in range (len(lab_enc.classes_)):
        print('{}对应的标签{}'.format(lab_enc.classes_[i],i))
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 4, random_state=0)
    print('\n共有{}条训练数据'.format(len(X_train)))
    print('共有{}条测试数据'.format(len(X_test)))
    print('\n===================== 任务3. 数据模型创建及验证 =====================')
    mode_name_dict ={'KNN':     (KNeighborsClassifier(), {'n_neighbors': [5, 10, 15]}),
                     'LR':      (LogisticRegression(),{'C': [0.1, 1, 10]}),
                     'SVM':     (SVC(),{'C': [100, 1000, 10000]}),
                     'DT':      (DecisionTreeClassifier(),{'max_depth': [100, 150,200]}),
                     'GNB':     (GaussianNB(),None),
                     'RF':      (RandomForestClassifier(),{'n_estimators': [10, 20, 30]}),
                     'GBDT':    (GradientBoostingClassifier(),{'learning_rate': [0.01, 0.1, 0.5, 1]})
                     }

    results_df = pd.DataFrame(columns=['Accuracy %'], index=list(mode_name_dict.keys()))
    results_df.index.name = 'Mode'
    models = []
    for model_name, model_config in mode_name_dict.items():
        print('************************{}模型*******************************'.format(model_name))
        acc, model = train_model(X_train, y_train,
                          X_test, y_test,
                          model_config)
        models.append(model)
        results_df.loc[model_name] = acc * 100
    # print(reslut_df)
    print('models\n', models)
    # 将结果保存到csv文件中
    results_df.to_csv('./pre_results.csv')
    # 将结果可视化
    results_df.plot(kind='bar')
    plt.ylabel('Accuracy')
    plt.savefig('./pre_results')
    plt.tight_layout()
    # plt.show()

    # 保存最优模型
    best_model_index = results_df.reset_index()['Accuracy %'].argmax()
    print('best_model_index\n',best_model_index)
    best_model = models[best_model_index]
    print('best_model\n', best_model)
    save_mode_path = './predictor.pkl'
    with open(save_mode_path, 'wb') as f:
        pickle.dump(best_model, f)

    # 任务四：数据预测
    print('\n===================== 任务4. 数据预测 =====================')
    # 加载保存的模型
    with open(save_mode_path, 'rb') as f:
        predictor = pickle.load(f)

    # 进行预测
    imdb_id = 2544
    img_path = os.path.join(poster_path, str(imdb_id)+ '.jpg')
    poster_feat = extract_hist_feat(img_path, n_feat_dim)
    pre_resultes = predictor.predict(poster_feat.reshape(1, -1))
    pred_genre = lab_enc.inverse_transform(pre_resultes)

    true_genre = data_df[data_df['imdbId'] == imdb_id]['Single_Genre'].values

    plt.figure()
    plt.imshow(io.imread(img_path))
    plt.grid(False)
    plt.title('实际类型:{},预测类型:{}\n'.format(true_genre, pred_genre))
    plt.show()

if __name__ == '__main__':
    main()