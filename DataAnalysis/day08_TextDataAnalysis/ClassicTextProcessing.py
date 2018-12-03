"""
    时间：2018年4月26日
    作者：张鹏
    内容：典型文本预处理流程
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# 原始文本
raw_text = 'Life is like a box of chocolates. You never know what you\'re gonna get.'

# 分词
raw_words = nltk.word_tokenize(raw_text)

# 词性归一化
wordent_lematizer = WordNetLemmatizer()
words = [wordent_lematizer.lemmatize(raw_word) for raw_word in raw_words]

# 去除停用词
filtered_words = [word for word in words if word not in stopwords.words('english')]

print('原始文本：', raw_text)
print('预处理文本：', filtered_words)