# _*_ coding: utf-8 _*_
"""
    时间：2018年12月24日
    作者：张鹏
    文件名：utils.py
    功能：配置文件
"""
import zipfile
import pandas as pd
import jieba.posseg as pseg
import re
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from PythonAI.TextData.NewsClassic import config

# 加载停用词
stopwords1 = [line.rstrip() for line in open('./data/stop_words/中文停用词库.txt', 'r', encoding='utf-8')]
stopwords2 = [line.rstrip() for line in open('./data/stop_words/哈工大停用词表.txt', 'r', encoding='utf-8')]
stopwords3 = [line.rstrip() for line in open('./data/stop_words/四川大学机器智能实验室停用词库.txt', 'r',
                                             encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3

def unzip_dataset_file():
    """
        解压数据集zip文件
    """
    if not os.path.exists(config.dataset_path):
        with zipfile.ZipFile(config.dataset_file) as f:
            print('正在解压')
            f.extractall(path='./data')
            print('解压完成')

def get_dict_from_category_file():
    """
        从类别文件中提取出类别名称
        返回：
            category_dict 字典类型， {url: 类别名称}
    """
    category_dict = {}
    with open(config.categories_file, 'r', encoding='GB2312') as f:
        lines = f.read().splitlines()

        i = 0
        while i < len(lines):
            category_dict[lines[i + 1]] = lines[i]
            i += 2
    return category_dict

def process_raw_text(category_dict):
    """
        读取数据文件，将url装换成对应的类别，并保存结果
        参数：
            category_dict
    """
    filename_list = os.listdir(config.dataset_path)

    category_list = []
    text_list = []
    for filename in filename_list:
        raw_text_filepath = os.path.join(config.dataset_path, filename)
        with open(raw_text_filepath, 'r',errors='ignore', encoding='GB2312') as f:
            lines = f.read().splitlines()
            i = 0
            while i < len(lines):
                # 通过url获取类别
                url_line = lines[i + 1]
                url_prefix = url_line[url_line.index('http://'): url_line.index('.com/') + 5]
                category_label = category_dict[url_prefix]
                category_list.append(category_label)

                if config.classify_type == 'title':
                    # 以新闻标题分类
                    title_line = lines[i + 3]
                    raw_text = title_line[title_line.index('<contenttitle>') + 14 : title_line.index('</contenttitle>')]
                    text_list.append(raw_text)
                else:
                    # 以新闻内容分类
                    content_line = lines[i + 4]
                    raw_text = content_line[content_line.index('<content>') + 9 : content_line.index('</contenttitle>')]
                    text_list.append(raw_text)
                if (len(category_list + 1) % 5000) == 0:
                    # 每处理5000条记录，查看最后一条的处理结果
                    print('处理结果：{}------{}'.format(category_list[-1], text_list[-1]))

    raw_text_df = pd.DataFrame()
    raw_text_df['category'] = category_list
    raw_text_df['text'] = text_list
    raw_text_df.to_csv(config.proc_raw_text_csv_file, index=False, encoding='uft-8')


def preprocess_text(raw_text):
    """
        文本预处理操作
        参数
            - raw_text 原始文本
        返回
            - proc_text 处理后的文本
    """
    # 1. 使用正则表达式去除非中文字符
    filter_pattern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_pattern.sub('', raw_text)

    # 2.结巴分词+词性标注
    words_lst = pseg.cut(chinese_only)

    # 去除停用词
    meaningful_word = []
    for word, flag in words_lst:
        if word not in stopwords:
            meaningful_word.append(word)
    process_text = ' '.join(meaningful_word)

    return process_text


def prepare_data():
    # 解压数据集
    unzip_dataset_file()

    # 读取类别描述文件构建dict。用于后续处理
    category_dict = get_dict_from_category_file()

    # 读取数据文件，将其url转换为对应的类别，并保存结果
    if not os.path.exists(config.proc_raw_text_csv_file):
        print('处理原始数据。。。。')
        process_raw_text(category_dict)
        print('完成并保存结果至',config.proc_raw_text_csv_file)
    if not os.path.exists(config.cln_text_csv_file):
        print('数据清洗。。。。')
        # 读取保存的文件
        raw_text_df = pd.read_csv(config.proc_raw_text_csv_file)

        # 清洗原始数据
        # 去除确实数据
        raw_text_df.dropna(inplace=True)
        # 处理文本数据
        raw_text_df['text'] = raw_text_df['text'].apply(preprocess_text)
        # 过滤空字符串
        cln_text_df = raw_text_df[raw_text_df['text'] !='']
        # 去除重复的结果数据
        cln_text_df = cln_text_df.drop_duplicates()

        # 保存处理好的文本数据
        cln_text_df.to_csv(config.cln_text_csv_file, index=None, encoding='utf-8')
        print('完成并保存至', config.cln_text_csv_file)
    else:
        cln_text_df = pd.read_csv(config.cln_text_csv_file)
    return cln_text_df


def get_top_categories(all_data):
    """
        获取top类别
        参数
            -all_data 全部数据
        返回：
        -   top_categories top类别
    """
    sorted_category = all_data.groupby('category').size().sort_values(ascending=False)
    print('各类样本数量')
    print(sorted_category)

    # 选取top的category作为最终分类
    top_categories = sorted_category.iloc[:config.n_category].index.tolist()
    print('top{}分类：'.format(config.n_category))
    print(top_categories)

    return top_categories


def do_feature_engineering(train_data, test_data):
    """
        特征工程获取文本特征

        参数
            - train_data 训练样本
            - test_data  测试样本
        返回
            - train_X 训练特征
            - test_X  测试特征
    """
    train_proc_text = train_data['text'].values
    test_proc_text = train_data['text'].values

    # TF-IDF 特征提取
    tfidf_vectorizer = TfidfVectorizer(max_features=config.n_common_words)
    train_tfidf_feat = tfidf_vectorizer.fit_transform(train_proc_text).toarray()
    test_tfidf_feat = tfidf_vectorizer.transform(test_proc_text).toarray()

    # 词袋模型
    count_vectorizer = CountVectorizer(max_features=config.n_common_words)
    train_count_feat = count_vectorizer.fit_transform(train_proc_text).toarray()
    test_count_feat = count_vectorizer.transform(test_proc_text).toarray()

    # 合并特征
    train_X = np.hstack((train_tfidf_feat, train_count_feat))
    test_X = np.hstack((test_tfidf_feat, test_count_feat))

    return train_X, test_X