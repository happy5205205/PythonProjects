"""
    时间：2018年4月27日
    作者：张鹏
    版本：python版本
    日期:     2017/10
    实战案例6：豆瓣影评数据分析

    配套的jupyter版本
"""

# 导入必要的包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import jieba.posseg as pseg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

# 解决matplotlib中中文显示乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号‘-’显示为方块的问题

# 指定数据路径
dataset_path = './data'
datafile = os.path.join(dataset_path, 'DMSC.csv')
# 停用词表的路径
stop_words_path = './stop_words'

# 加载停用词表
stopword1 = [line.strip() for line in open(os.path.join(stop_words_path, '中文停用词库.txt'), 'r',
                                      encoding='utf-8')]
stopword2 = [line.strip() for line in open(os.path.join(stop_words_path, '哈工大停用词表.txt'), 'r',
                                      encoding='utf-8')]
stopword3 = [line.strip() for line in open(os.path.join(stop_words_path, '四川大学机器智能实验室停用词库.txt'), 'r',
                                      encoding='utf-8')]

stopword = stopword1 + stopword2 +stopword3

def pro_text(raw_line):
    """
        处理文本数据
        返回分词结果
    """
    # 1.使用正则表达式去除非中文结果
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_line)

    # 结巴分词+词性标注
    word_list = pseg.cut(chinese_only)

    # 去除停用词，保留有意义的词性
    # 动词，形容词，副词
    used_flags = ['v', 'a', 'ad']
    meaningful_word = []
    for word, flag in word_list:
        if (word not in stopword) and (flag in used_flags):
            meaningful_word.append(word)
    return ' '.join(meaningful_word)
def main():
    """
        主函数
    """
    # 加载数据
    raw_data = pd.read_csv(datafile)

    # 任务1. 数据查看
    print('\n===================== 任务1. 数据查看 =====================')
    print('共有{}条数据。'.format(len(raw_data)))
    # 电影名称
    print('一共有{}部电影'.format(len(raw_data['Movie_Name_CN'].unique())))
    print('电影名称如下：\n{}'.format(raw_data['Movie_Name_CN'].unique()))

    # 电影平均得分
    movie_mean_score = raw_data.groupby['Movie_Name_CN']['Star'].mean().sort_values(ascending=False)
    movie_mean_score.plot(kind = 'bar')
    plt.tight_layout()

    #任务2. 数据预处理
    print('\n===================== 任务2. 数据预处理 =====================')
    # 去除空值
    cln_data = raw_data.dropna().copy()
    # 建立新的一列， 如果打分>=分，为正面评价1， 否则为负面评价0
    cln_data['Positively_Rated'] = np.where(cln_data['Star'] >=3, 1, 0)
    # 数据预览
    print('数据预览前五条：\n{}'.format(cln_data.head()))

    # 处理后的数据保存
    save_data = cln_data[['Word' , 'Positively_Rated']]
    save_data.dropna(subset=['Word'], inplace=True)
    save_data.to_csv(os.path.join(dataset_path, 'douban_cln_data.csv'), encoding='utf-8',
                     index=False)

    # 分割训练数据集和测试数据集
    X_train_data, X_test_data, y_train, y_test =train_test_split(save_data['Word'],
                                                                 save_data['Positively_Rated'],
                                                                 test_size=1/4,
                                                                 random_state=0
                                                                 )
     # 任务3. 文本特征提取
    print('\n===================== 任务3. 文本特征提取 =====================')
    # max_features 指点语料库中频率最高的词
    n_dim = 10000
    vectorizer = TfidfVectorizer(max_features=n_dim)
    X_train = vectorizer.fit_transform(X_train_data.values)
    X_test = vectorizer.transform(X_test_data.values)
    print('特征维度：{}'.format(len(vectorizer.get_feature_names())))
    print('语料词中top的词:\n{}'.format(n_dim))
    print(vectorizer.get_feature_names())

    # 任务4. 建模及预测
    print('\n===================== 任务4. 建模及预测 =====================')
    lr_model = LogisticRegression(C=100)
    lr_model.fit_transform(X_train, y_train)
    prediction = lr_model.predict(X_test)
    print('AUC值：', roc_auc_score(y_test, prediction))

    if __name__ == '__main__':
        main()





