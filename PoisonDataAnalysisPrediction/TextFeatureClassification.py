"""
    时间：2018年4月26日
    作者：张鹏
    内容：文本特征分类
"""

print('\n===================== 1. 情感分析 =====================')
# 1. 情感分析
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

text1 = 'I like the movie so much!'
text2 = 'That is a good movie.'
text3 = 'This is a great one.'
text4 = 'That is a really bad movie.'
text5 = 'This is a terrible movie.'

def pro_text(text):
    """
        预处理文本
    """
    # 分词
    raw_words = nltk.word_tokenize(text)

    # 词形归一化
    wordnet_lemmtizer = WordNetLemmatizer()
    words = [wordnet_lemmtizer.lemmatize(raw_word) for raw_word in raw_words]

    # 去除停用词
    filtered_words = [word for word in words if word not in stopwords.words('english')]

    # 表示该词在文本中，为了使用nltk中的分类器
    return {word:True for word in filtered_words}
# 构造样本
train_data = [
              [pro_text(text1), 1],
              [pro_text(text2), 1],
              [pro_text(text3), 1],
              [pro_text(text4), 0],
              [pro_text(text5), 0]
              ]

print('train_data',train_data)

# 训练模型
nb_model = NaiveBayesClassifier.train(train_data)

# 测试模型
text6 = 'that is a bad one'
print('测试结果：',nb_model.classify(pro_text(text6)))


print('\n===================== 2. 文本相似度 =====================')
# 2. 文本相似度
import nltk
from nltk import FreqDist

text1 = 'I like the movie so much '
text2 = 'That is a good movie '
text3 = 'This is a great one '
text4 = 'That is a really bad movie '
text5 = 'This is a terrible movie'

text = text1 + text2 + text3 + text4 + text5
words = nltk.word_tokenize(text)
freq_dist = FreqDist(words)
print(freq_dist['That'])

# 取出常用的n=5个单词
n = 5

# 构造常用的单词列表
most_common_words = freq_dist.most_common(n)
print('构造常用的单词列表\n',most_common_words)

def lookup_pos(most_common_words):
    """
        查找常用单词的位置
    """
    result = {}
    pos = 0
    for word in most_common_words:
        result[word[0]]=pos
        pos += 1
    return result
# 记住位置
std_pos_dict = lookup_pos(most_common_words)
print('std_pos_dict', std_pos_dict)

# 新文本
new_text = 'That one is a good movie. This is so good!'

# 初始化向量
freq_vec = [0] * n

# 分词
new_words = nltk.word_tokenize(new_text)

# 在常用单词列表上面计算词频统计
for new_word in new_words:
    if new_word in list(std_pos_dict.keys()):
        freq_vec[std_pos_dict[new_word]] +=1

print('freq_vec',freq_vec)

print('\n===================== 3. 文本分类及TF-IDF =====================')
# 3. 文本分类及TF-IDF

# 3.1 NLTK中的TF-IDF
from nltk.text import TextCollection

text1 = 'I like the movie so much '
text2 = 'That is a good movie '
text3 = 'This is a great one '
text4 = 'That is a really bad movie '
text5 = 'This is a terrible movie'

# 创建TextCollection对象
tc = TextCollection([text1, text2, text3, text4, text5])
new_text = 'That one is a good movie. This is so good!'
word = 'That'
tf_idf_val = tc.tf_idf(word, new_text)
pro_text('{}的TF-IDF值为：{}'.format(word, tf_idf_val))

# 3.1 sklearn中的TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
feat = vectorizer.fit_transform([text1, text2, text3, text4, text5])
print(vectorizer.get_feature_names())
feat_arrary = feat.toarray()
print(feat_arrary.shape)
print(feat_arrary[0, :])
print(vectorizer.transform([new_text]).toarray())

# 3.3 中文中的TF-IDF
ch_text1 = ' 非常失望，剧本完全敷衍了事，主线剧情没突破大家可以理解，可所有的人物都缺乏动机，' \
           '正邪之间、妇联内部都没什么火花。团结-分裂-团结的三段式虽然老套但其实也可以利用积' \
           '攒下来的形象魅力搞出意思，但剧本写得非常肤浅、平面。场面上调度混乱呆板，满屏的铁甲审美疲劳。' \
           '只有笑点算得上差强人意。'
ch_text2 = ' 2015年度最失望作品。以为面面俱到，实则画蛇添足；以为主题深刻，实则老调重弹；以为推陈出新，' \
           '实则俗不可耐；以为场面很high，实则high劲不足。气！上一集的趣味全无，这集的笑点明显刻意到心虚。' \
           '全片没有任何片段给我有紧张激动的时候，太弱了，跟奥创一样。'
ch_text3 = ' 《铁人2》中勾引钢铁侠，《妇联1》中勾引鹰眼，《美队2》中勾引美国队长，' \
           '在《妇联2》中终于……跟绿巨人表白了，黑寡妇用实际行动告诉了我们什么叫忠贞不二；' \
           '而且为了治疗不孕不育连作战武器都变成了两支验孕棒(坚决相信快银没有死，后面还得回来)'
ch_text4 = ' 虽然从头打到尾，但是真的很无聊啊。'
ch_text5 = ' 剧情不如第一集好玩了，全靠密集笑点在提神。僧多粥少的直接后果就是每部寡姐都要换着队友谈恋爱，' \
           '这特么比打斗还辛苦啊，真心求放过～～～（结尾彩蛋还以为是洛基呢，结果我呸！）'

ch_texts = [ch_text1, ch_text2, ch_text3, ch_text4, ch_text5]

import jieba

corpus = []
for ch_text in ch_texts:
    corpus.append(' '.join(jieba.cut(ch_text, cut_all=False)))
# # 或者
# corpus = ['/'.join(jieba.cut(ch_text, cut_all=False)) for ch_text in ch_texts]

# print(corpus)

ch_vectorizer = TfidfVectorizer()
ch_feats = ch_vectorizer.fit_transform(corpus)
print(len((ch_vectorizer.get_feature_names())))
print(ch_feats.toarray()[0, :])
