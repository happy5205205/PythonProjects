# _*_ coding: utf-8 _*_
"""
    时间：2018年12月17日
    作者：张鹏
    结巴分词
"""
# 1.情感分析
# 简单的例子

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

text1 = 'I like the movie so much!'
text2 = 'That is a good movie.'
text3 = 'This is a great one.'
text4 = 'That is a really bad movie.'
text5 = 'This is a terrible movie.'

def proc_text(text):
    """
        预处理文本
        :return:
    """
    # 分词
    raw_words = nltk.word_tokenize(text)

    # 词形归一化
    wordent_lematizer = WordNetLemmatizer()
    words = [wordent_lematizer.lemmatize(raw_word) for raw_word in raw_words ]

    # 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    # True 表示该词在文本之中，为了使用nltk分类器
    return {word : True for word in filtered_words}

# 构造训练样本
train_data = [[proc_text(text1), 1],
              [proc_text(text2), 1],
              [proc_text(text3), 1],
              [proc_text(text4), 0],
              [proc_text(text5), 0]
              ]
# print(train_data)

# 训练模型
nb_model = NaiveBayesClassifier.train(train_data)

# 测试模型
text6 = 'That is a good movie'
# print(nb_model.classify(proc_text(text6)))

# 文本相似度
import nltk
from nltk import FreqDist

text = text1 + text2 + text3 + text4 + text5
print(type(text))
words = nltk.word_tokenize(text)
freq_dist = FreqDist(words)
# print(freq_dist['is'])
# print(freq_dist)

# 取出常用的n=5个单词
n = 5
# 构造“常用单词列表”
most_common_words = freq_dist.most_common(n)
print(most_common_words)

def lookup_pos(most_common_words):
    """
        查看常用单词的位置
        :param most_common_words:
        :return:
    """
    result = {}
    pos = 0
    for word in most_common_words:
        result[word[0]] = pos
        pos += 1
    return result

# 记录位置

std_pos_dict = lookup_pos(most_common_words)
print(std_pos_dict)

# 新文本
new_text = 'That one is a good movie. This is so good!'

# 初始化向量
freq_vec = [0] * n
# 分词
new_words = nltk.word_tokenize(new_text)

# 在常用单词列表上面统计词频
for new_word in new_words:
    if new_word in list(std_pos_dict.keys()):
        freq_vec[std_pos_dict[new_word]] += 1
print('freq_vec',freq_vec)

# new_text = 'That one is a good movie.'

# 3. 文本分类及TF-IDF
# 3.1 NLTK中的TF-IDF
from nltk.text import TextCollection

# 构建TextCollection对象
tc = TextCollection([text1, text2, text3, text4, text5])
new_text = 'That one is a good movie. This is so good!'
word = 'That'
tf_idf_val = tc.tf_idf(word, new_text)
print('{}的tf_idf的值为{}'.format(word, tf_idf_val))

# 3.2 sklearn中的tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
feat = vectorizer.fit_transform([text1, text2, text3, text4, text5])
feat.toarray()
vectorizer.get_feature_names()
feat_array = feat.toarray()
print('feat_array', feat_array)
print('feat_array.shape'.format(feat_array.shape))
print(vectorizer.transform([new_text]).toarray)

# 中文TF-IDF
import os
import jieba.posseg as pseg
import re

ch_text1 = '5月9日，新华社发文称，五一前后住建部约谈了成都、太原等12个城市，主要集中在东北、中西部地区以及政策利好的海南。'
ch_text2 = '响应国家号召，被约谈城市纷纷出台举措应对楼市过热问题。'
ch_text3 = '成都由此将限购对象从自然人升级为家庭，西安对碧桂园等42家房企进行约谈，昆明、贵阳、长春则升级了限售政策。'
ch_text4 = '5月份楼市调控的高潮出现在19日，住建部再次重申坚持房地产调控目标不动摇、力度不放松，并从住房发展规划、住房和土地供应、资金管控、市场监管、落实主体责任等方面做出了明确要求。'
ch_text5 = '调控的目的只有一个   ，那就是稳定楼市。'

ch_texts = [ch_text1, ch_text2, ch_text3, ch_text4, ch_text5]

stop_words_path = './stop_words/'

stopwords1 = [line.rstrip() for line in open(os.path.join(stop_words_path, '中文停用词库.txt'), 'r',
                                             encoding='utf-8')]
stopwords2 = [line.rstrip() for line in open(os.path.join(stop_words_path, '哈工大停用词表.txt'), 'r',
                                             encoding='utf-8')]
stopwords3 = [line.rstrip() for line in
              open(os.path.join(stop_words_path, '四川大学机器智能实验室停用词库.txt'), 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3

print(len(stopwords))

def text_pro(raw_line):
    """
        处理文本数据
        返回分词结果
    """
    # 使用正则表达式去除非中文字符
    filter_parttern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_parttern.sub('',raw_line)

    # 结巴分词+词性标注
    word_list = pseg.cut(chinese_only)

    # 去除停用词，保留有意义的词性
    # 动词，形容词，副词
    used_flags = ['v', 'a', 'ad']
    meaningful_words = []
    for word, flag in word_list:
        if word not in stopwords:
            meaningful_words.append(word)
    return ' '.join(meaningful_words)

corups = [text_pro(ch_text) for ch_text in ch_texts]

ch_vectorizer = TfidfVectorizer()
ch_feats = ch_vectorizer.fit_transform(corups)
ch_vectorizer.get_feature_names()
print(ch_feats.toarray()[0,:])