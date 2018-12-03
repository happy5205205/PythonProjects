"""
    时间：2018年4月25日
    作者：张鹏
    版本：2.0
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, exposure
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 解决matplotlib中中文字出实现乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 指定数据集路径
dataset_path = './data'
csv_filpath = os.path.join(dataset_path, 'MovieGenre.csv')
poster_path = os.path.join(dataset_path, 'SampleMoviePosters')

def extract_hist_feat(img_path, nbins= 50, as_grey=True):
    """
        提取每个图片的直方图作为颜色特征
        img_path：图片路径
        nbins：直方图bin的个数，即特征的维度
    """
    image_data = io.imread(img_path, as_grey=as_grey)
    if as_grey:
        # 灰度图片
        # 直方图均衡化
        eq_image_data = exposure.equalize_hist(image_data)
        # 提取直方图特征
        hist_feat, _= exposure.histogram(eq_image_data, nbins=nbins)
        # 统一直方图的频率（归一化特征），避免因为图片的尺寸不同导致直方图统计个数不同
        norm_hist_feat = hist_feat / sum(hist_feat)
    else:
        # 彩色图片
        # 每个通道提取直方图，然后再合并

        # RGB三通道直方图均衡化
        r_eq_image_data = exposure.equalize_hist(image_data[:,:,0])
        g_eq_image_data = exposure.equalize_hist(image_data[:,:,1])
        b_eq_image_data = exposure.equalize_hist(image_data[:,:,2])

        # RGB三通道提取直方图特征和没通道归一化特征
        # R通道
        r_hist_feat, _ = exposure.histogram(r_eq_image_data, nbins=nbins)
        norm_r_hist_feat = r_hist_feat / sum(r_hist_feat)
        # G通道
        g_hist_feat, _ = exposure.histogram(g_eq_image_data, nbins=nbins)
        norm_g_hist_feat = g_hist_feat / sum(g_hist_feat)
        # B通道
        b_hist_feat, _ = exposure.histogram(b_eq_image_data, nbins=nbins)
        norm_b_hist_feat = b_hist_feat / sum(b_hist_feat)
        norm_hist_feat = np.concatenate((norm_r_hist_feat, norm_g_hist_feat, norm_b_hist_feat))
    return norm_hist_feat

def model_train():
    pass

def main():
    """
        主函数
    """
    # 任务1 ： 数据集的查看及处理
    print('\n===================== 任务1. 数据查看及处理 =====================')
    movie_df = pd.read_csv(csv_filpath, encoding='ISO-8859-1',
                           usecols = ['imdbId', 'Title', 'IMDB Score', 'Genre'])

    # 数据分割，取第一个分类作为预测
    movie_df['single_gener'] = movie_df['Genre'].str.split('|', expand=True)[0]
    # print(movie_df['single_gener'])
    # 将数据文件CSV与海报文件路径经行合并
    # 创建海报DataFrame
    poster_df = pd.DataFrame(columns=['imdbId', 'img_path'])
    poster_df['img_path'] = os.listdir(poster_path)
    poster_df['imdbId'] = poster_df['img_path'].str[: -4].astype('int')
    # print(poster_df.head())
    data_df = movie_df.merge(poster_df, on='imdbId', how='inner')
    # 删除重复项
    data_df.drop_duplicates(subset=['imdbId'], inplace=True)
    # print(data_df.head())
    print('SCV文件的数据有{}条，有{}张海报，合并后的数据{}条。'.format(len(movie_df),
                                                  len(poster_df), len(data_df)))
    print('处理前的电影分类数量\n',data_df.groupby('single_gener')
          .size().sort_values(ascending=False))
    # 查看每一类数据的个数直方图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='single_gener', data=data_df)
    plt.title('处理前的电影分类数量')
    plt.xticks(rotation='vertical')
    plt.xlabel('电影类别')
    plt.ylabel('数量')
    plt.tight_layout()

    # 其中一些电影类型数据量过少不利于预测，所以将电影数据类型数据量少的合并为Other
    # 电影类型为：Drama，Comedy，Other
    # data_df.loc[(data_df['single_gener'] != 'Drama') &
    #             (data_df['single_gener'] != 'Comedy'), 'single_gener'] = 'Other'
    cond = (data_df['single_gener'] != 'Drama') & (data_df['single_gener'] != 'Comedy')
    data_df.loc[cond, 'single_gener'] = 'Other'
    print('处理后的电影分类数量\n',data_df.groupby('single_gener').size().sort_values(ascending=False))
    plt.subplot(1, 2, 2)
    sns.countplot(x='single_gener', data=data_df)
    plt.title('处理后的电影分类数量')
    plt.xticks(rotation='vertical')
    plt.xlabel('电影类别')
    plt.ylabel('数量')
    plt.tight_layout()
    # plt.show()
    # plt.savefig('./Total Number of Movie.jpg')
    # data_df.reset_index(inplace=True)
    print('处理后的数据预览前5条,如下所示：\n{}'.format(data_df.head()))

    print('\n===================== 任务2. 特征表示 =====================')
    n_feat_dim = 100
    n_sample = len(data_df)
    # 初始化特征矩阵
    X = np.zeros((n_sample,n_feat_dim))
    print('X是{}的矩阵'.format(X.shape))

    for i, r_data in data_df.iterrows():
        if (i+1) % 100 ==0:
            print('正在提取特征，已完成{}张图片的提取。'.format(i+1))
        img_path = os.path.join(poster_path, r_data['img_path'])
        hist_feat = extract_hist_feat(img_path,n_feat_dim)
        # 赋值到矩阵中
        X[i:] = hist_feat.copy()
    # 标签处理
    from sklearn.preprocessing import LabelEncoder
    lab_enc = LabelEncoder()
    y = lab_enc.fit_transform(data_df['single_gener'].values)
    for i in range (len(lab_enc.classes_)):
        print('{}对应的标签为{}'.format(lab_enc.classes_[i], i))
    # 数据分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)
    print('训练数据{}条\n测试数据{}条'.format(len(X_train), len(X_test)))
    # 特征归一化
    scaler = MinMaxScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    print('\n===================== 任务3. 数据建模及验证 =====================')
    activations = ['relu', 'logistic', 'tanh']
    alphas = [0.01, 0.1, 1, 10]
    results_df = pd.DataFrame(columns=alphas, index=activations)
    for activation in activations:
        print('激活函数：{}\n'.format(activations))
        for alpha in alphas:
            ann_model = MLPClassifier(hidden_layer_sizes=[100, 100], activation=activation,
                                      alpha=alpha, solver='lbfgs', random_state=0)
            ann_model.fit(X_train_scaler, y_train)
            print('alpha:{}'.format(alpha))
            acc = ann_model.score(X_test_scaler, y_test)
            print('准确率为：{:.3f}'.format(acc))

            results_df.loc[activation, alpha] = acc
    # 保存结果
    results_df.to_csv('./results_pre.csv')

    # 可视化结果
    results_df.plot(kind = 'bar')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('./results_pre.jpg')
    plt.show()



if __name__ == '__main__':
    main()