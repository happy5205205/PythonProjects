"""
    时间：2018年4月27日
    作者：张鹏
    1.项目描述：
           豆瓣电影为互联网用户提供了一个可以分享电影评论及观点的平台。用户在提供影评的同时还可以为电影打分。
    2.数据集的描述：
          Kaggle提供的数据集。数据集（DMSC.csv）为CSV文件。该数据集包含了28部电影的200万条影评。可以用于文本分类，
    聚类，情感分析等。
        数据字典
        ID：评论的ID（从0开始）
        Movie_Name_EN：电影的英文名称
        Movie_Name_CN：电影的中文名称
        Crawl_Date：爬取日期
        Number：评论个数
        Username：发表评论账户的用户名
        Date：评论的日期
        Star：评论时用户的打分（1~5）
        Comment：评论内容
        Like：评论被点赞的次数
    3.项目任务：
        3.1 数据查看
        3.2 数据预处理
        3.3 文本特征处理
        3.4 建模及预测
"""
import nltk
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib 中中文乱码解决问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 指定数据集路径
dataset_path = './data'
datafile = os.path.join(dataset_path, 'DMSC.csv')
# 停用词表路径
stop_words_path = './stop_words'

# 加载数据
raw_data = pd.read_csv(datafile)

# 数据查看
print('数据集有{}条记录。'.format(len(raw_data)))

# 电影名称
print('数据包含{}部电影。'.format(len(raw_data['Movie_Name_CN'].unique())))

print(raw_data['Movie_Name_CN'].unique())

# 电影平均得分
movie_mean_score = raw_data.groupby('Movie_Name_CN')['Star'].mean().sort_values(ascending=False)
# print(movie_mean_score)
movie_mean_score.plot(kind = 'bar')
plt.tight_layout()
# plt.show()

# 数据预处理
# 去除空值
cln_data = raw_data.dropna().copy()

# 创建新的一列，如果打分>=3赋值为1表示为正面评分，否则赋值为0表示负面评分
cln_data['Positively_Rated'] = np.where(cln_data['Star'] >= 3, 1, 0)
# 数据预览
print(cln_data.head())

# 加载停用词表
stop_words_1 = [line.rstrip() for line in open(os.path.join(stop_words_path, '中文停用词库.txt'), 'r', encoding='utf-8')]
stop_words_2 = [line.rstrip() for line in open(os.path.join(stop_words_path, '哈工大停用词表.txt'), 'r', encoding='utf-8')]
stop_words_3 = [line.rstrip() for line in open(os.path.join(stop_words_path, '四川大学机器智能实验室停用词库.txt'), 'r', encoding='utf-8')]
stop_words = stop_words_1 + stop_words_2 + stop_words_3
print(stop_words[0:10])

# 处理文本数据
import re
import jieba.posseg as pseg

def proc_text(raw_line):
    """
        处理文本数据
        返回分词结果
    """

    # 1 使用正则表达式去除非中文中的字符
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_line)

    # 2 结巴分词+词性标注
    word_list = pseg.cut(chinese_only)

    # 3 去除停用词，保留有意义的词性
    # 动词，形容词，副词
    used_flags = ['v', 'a', 'ad']
    meaningful_word = []
    for word, flag in word_list:
        if (word not in stop_words) and (flag in used_flags):
            meaningful_word.append(word)

    return ' '.join(meaningful_word)

# 测试一条记录
# test_text = cln_data.loc[5, 'Comment']
# print('原文本：', test_text)
# test_text_res = proc_text(test_text)
# print('\n处理后的文本：', test_text_res)

# 处理数据集中所有的文本
cln_data['Words'] = cln_data['Comment'].apply(proc_text)
# cln_data['Words'] = proc_text(l for l in cln_data['Comment'])
print(cln_data.head(2))

# 将处理后的数据集保存
saved_data = cln_data[['Words', 'Positively_Rated']].copy()
saved_data.dropna(subset=['Words'], inplace = True)
saved_data.to_csv(os.path.join(dataset_path, 'douban_cln_data.csv'), encoding='utf-8', index=False)

# 分割文本集与测试集
from sklearn.model_selection import train_test_split
X_train_data, X_test_data, y_train, y_test = train_test_split(saved_data['Words'], saved_data['Positively_Rated'],
                                                              test_size=1/4, random_state=0)
print('X_test_data.shape:',X_test_data.shape)
print('X_test_data的第一条记录为：\n', X_train_data.iloc[1])
print('\n训练样本{}条，测试样本{}条'.format(len(X_train_data), len(X_test_data)))

# 文本特征提取
from sklearn.feature_extraction.text import TfidfVectorizer

# max_features 指定语料库中频率最高的词
n_dim = 10000
vectorizer = TfidfVectorizer(max_features=n_dim)
X_train = vectorizer.fit_transform(X_train_data.values)
X_test = vectorizer.transform(X_test_data.values)

print('X_train.shape',X_train.shape)
print('特征维度：', len(vectorizer.get_feature_names()))
print('语料库中top{}的词：'.format(n_dim))
print('vectorizer.get_feature_names()',vectorizer.get_feature_names())

# 建模及预测
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(C = 100)
lr_model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
prediction = lr_model.predict(X_test)
print('AUC:', roc_auc_score(y_test, prediction))



