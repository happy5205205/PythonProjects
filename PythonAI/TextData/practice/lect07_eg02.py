# _*_ coding:utf-8 _*_
"""
    时间：2018年12月10ri
    作者：张鹏
"""
# 文本预处理

# 1、nltk基本操作
import nltk
# nltk.download()

from nltk.corpus import brown

# # 引用布朗大学的语料库
# print(brown.categories)
# print(brown.categories())

# 查看brown大学语料库
# print('一共优{}条句子'.format(len(brown.sents())))
# print('一共有{}个单词'.format(len(brown.words())))
# print(type(brown.words()))
# print(brown.words()[:10])

# 词频统计
from nltk import FreqDist
dist = FreqDist(brown.words())
# print(dist)
# print(type(dist))
# print('非重复单词个数{}个'.format(len(dist)))
# print('前十个单词：{}'.format(list(dist.keys())[:10]))
# print('the出现的次数：{}'.format(dist['the']))

# 找出长度大于5，且出现频词大于500的单词
freq_words = [w for w in brown.words() if len(w) > 5 and dist[w] > 500]
# print('找出长度大于5，且出现频词大于500的单词',freq_words[:10])

# 分词
# 2.1 NLTK英文分词
sentence = "Python is a widely used high-level programming language for general-purpose programming."
tokens = nltk.word_tokenize(sentence)
print('NLTK英文分词', sentence)

# 2.2 分句
texts = 'Python is a widely used high-level programming language for general-purpose programming, created by Guido van Rossum and first released in 1991. An interpreted language, Python has a design philosophy that emphasizes code readability (notably using whitespace indentation to delimit code blocks rather than curly brackets or keywords), and a syntax that allows programmers to express concepts in fewer lines of code than might be used in languages such as C++ or Java.[23][24] The language provides constructs intended to enable writing clear programs on both a small and large scale.'
sentences = nltk.sent_tokenize(texts)
# print(len(sentences))
# print(sentences)

# 2.3中文结巴分词
import jieba
seg_list = jieba.cut('欢迎来到小象学院', cut_all=True)
print('全模式：' + '/'.join(seg_list))
seg_list = jieba.cut('欢迎来到小象学院', cut_all=False)
print('精准模式' + '/'.join(seg_list))

# 3 词行归一化
# 3.1 词干提取

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
# print(porter_stemmer.stem('looked'))
# print(porter_stemmer.stem('went'))

input1 = 'List listed lists listing listings'
words1 = input1.lower().split(' ')
x = [porter_stemmer.stem(w) for w in words1]
# print(x)

from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
# print(snowball_stemmer.stem('looked'))
# print(snowball_stemmer.stem('looking'))

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
# print(lancaster_stemmer.stem('looked'))
# print(lancaster_stemmer.stem('looking'))

# 3.2 词形归并(lemmatization)
from nltk.stem import WordNetLemmatizer
wordnet_lematizer = WordNetLemmatizer()
# print(wordnet_lematizer.lemmatize('cats'))
# print(wordnet_lematizer.lemmatize('boxes'))
# print(wordnet_lematizer.lemmatize('are'))
# print(wordnet_lematizer.lemmatize('went'))

# 指明词性可以更准确地进行lemma
# lemmatize 默认为名词
# print(wordnet_lematizer.lemmatize('are', pos='v'))
# print(wordnet_lematizer.lemmatize('went', pos='v'))

# 4 词性标注
# words = nltk.word_tokenize('Python is a widely used programming language.')
# # print(words)
# print(nltk.pos_tag(words)) # 需要下载 averaged_perceptron_tagger

# 5.去除停用词
# from nltk.corpus import stopwords
# filtered_words = [word for word in words if word not in stopwords.words('english')]
# print('原始：',words)
# print('去除停用词', filtered_words)

# 典型文本处理流程
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# 原始文本
raw_text = 'Life is like a box of chocolates. You never know what you\'re gonna get.'

# 分词
raw_words = nltk.word_tokenize(raw_text)

# 词形归一化
wordnet_lematizer = WordNetLemmatizer()
words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]

# 去除停用词
filtered_words = [word for word in words if word not in stopwords.words('english')]

print('原始文本{}'.format(raw_text))
print('预处理结果{}'.format(filtered_words))
