"""
    时间：2018年4月26日
    作者：张鹏
    内容：1. NLTK基本操作
        2. 分词
        3. 词形归一化
        4. 词性标注 (Part-Of-Speech)
        5. 去除停用词
        6. 典型的文本预处理流程
"""

# 1. NLTK基本操作
# import nltk
# nltk.download()

import nltk
from nltk.corpus import brown  # 需要下载brow语料库
# 引入布朗大学的语料库

# 查看语料库包含的类别
print('查看brown语料库包含的类别\n',brown.categories())

# 查看brown语料库
# print('查看brown语料库共有句子{}条：'.format(len(brown.sents())))
# print('查看brown语料库共有单词{}个：'.format(len(brown.words())))

print('查看brown语料库前十个单单词是：{}'.format(brown.words()[:10]))

# 词频统计
from nltk import FreqDist
dist = FreqDist(brown.words())
print('dist:',dist)
print('非重复单词个数：', len(dist))
print('前十个单词：', list(dist.keys())[:10])
print('the出现的次数：', dist['the'])

# 找出长度大于5且出现次数大于500次
ferq_wordas = [w for w in dist.keys() if len(w)> 5 and dist[w] >500]
print('找出长度大于5且出现次数大于500次', ferq_wordas)

# 2. 分词
# 2.1 NLTK英文分词
scetence = "Python is a widely used high-level programming language for general-purpose programming."
tokens = nltk.word_tokenize(scetence) # 需要下载puntkt分词模型
print('NLTK英文分词', scetence)
# 2.2 分句
texts = 'Python is a widely used high-level programming language for general-purpose programming,' \
        ' created by Guido van Rossum and first released in 1991. An interpreted language,' \
        ' Python has a design philosophy that emphasizes code readability' \
        ' (notably using whitespace indentation to delimit code blocks rather ' \
        'than curly brackets or keywords), and a syntax that allows programmers to express ' \
        'concepts in fewer lines of code than might be used in languages such as C++ or Java.[23][24] ' \
        'The language provides constructs intended to enable writing clear programs on both a small and large scale.'
scetences = nltk.sent_tokenize(texts)
print('NLTK英文分句长度：', len(scetences))
print('NLTK英文分句:\n', scetences)
# 2.3 中文结巴分词
# 安装 pip install jieba
import jieba
seg_list = jieba.cut("欢迎来到小象学院", cut_all=True)
print('全模式:' + ' / '.join(seg_list))             # 全模式

seg_list = jieba.cut("欢迎来到小象学院", cut_all=False)
print('精准模式:' + ' / '.join(seg_list))           # 精准模式

# 3 词形归一化

# 3.3 词干提取（stemming）
# Posterstemmer
from nltk.stem.porter import PorterStemmer
poster_stemmer = PorterStemmer()
print(poster_stemmer.stem('looked'))
print(poster_stemmer.stem('went'))

input1 = 'List listed lists listing listings'
words1 =input1.lower().split(' ')
print('Posterstemmer词干提取:', [poster_stemmer.stem(w) for w in words1])

# SnowballStemmer
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('english')
print(snowball_stemmer.stem('looking'))
print(snowball_stemmer.stem('looked'))

# 3.2 词性归并（lemmatization）
from nltk.stem import WordNetLemmatizer # 需要下载wordent语料库
wordnet_lemmatizer = WordNetLemmatizer()
print('归并前cats，归并后', wordnet_lemmatizer.lemmatize('cats'))
print('归并前boxes，归并后', wordnet_lemmatizer.lemmatize('boxes'))
print('归并前are，归并后', wordnet_lemmatizer.lemmatize('are'))
print('归并前went，归并后', wordnet_lemmatizer.lemmatize('went'))

# 指明词性可以更准确的进行lemma
# lemmatize 默认为名词
print('指明词性可以更准确的进行lemma:',wordnet_lemmatizer.lemmatize('are', pos='v'))
print('指明词性可以更准确的进行lemma',wordnet_lemmatizer.lemmatize('went', pos='v'))

# 4. 词性标注 (Part-Of-Speech)
import nltk
words = nltk.word_tokenize('Python is a widely used programming language.')
print('词性标注 (Part-Of-Speech):\n',nltk.pos_tag(words)) # 需要下载 averaged_perceptron_tagger

# 5. 去除停用词
from nltk.corpus import stopwords
filtered_words = [word for word in words if word not in stopwords.words('english')]
print('原始词：',words)
print('去除停用词后：', filtered_words)




