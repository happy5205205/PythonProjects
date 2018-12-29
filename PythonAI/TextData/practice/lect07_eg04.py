# _*_ coding: utf_8 _*_

"""
    时间：2018年12月24日
    作者：张鹏
"""
# 词袋模型
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
import jieba.posseg as pseg

# 加载停用词表
stop_words_path = './stop_words/'

stopwords1 = [line.rstrip() for line in open(os.path.join(stop_words_path, '中文停用词库.txt'), 'r',
                                             encoding='utf-8')]
stopwords2 = [line.rstrip() for line in open(os.path.join(stop_words_path, '哈工大停用词表.txt'), 'r',
                                             encoding='utf-8')]
stopwords3 = [line.rstrip() for line in
              open(os.path.join(stop_words_path, '四川大学机器智能实验室停用词库.txt'), 'r', encoding='utf-8')]
stopwords = stopwords1 + stopwords2 + stopwords3

def proc_text(raw_line):
    """
        处理文本数据
        返回分词结果
    """
    # 1 使用正则表达式去除非中文字符
    filter_parttern = re.compile('[^\u4E00-\u9FD5]+')
    chinese_only = filter_parttern.sub('', raw_line)

    # 2 结巴分词和词性标注
    word_list = pseg.cut(chinese_only)

    # 3 去除停用词，保留有意义的词性
    # 动词、形容词、副词、名词
    used_flags = ['v', 'a', 'ad', 'n']
    meaningful_words = []
    for word, flag in word_list:
        if (word not in stopwords) and (flag in used_flags):
            meaningful_words.append(word)

    return ''.join(meaningful_words)


count_vectorizer = CountVectorizer()

ch_text1 = '5月9日，新华社发文称，五一前后住建部约谈了成都、太原等12个城市，主要集中在东北、中西部地区以及政策利好的海南。'
ch_text2 = '响应国家号召，被约谈城市纷纷出台举措应对楼市过热问题。'
ch_text3 = '成都由此将限购对象从自然人升级为家庭，西安对碧桂园等42家房企进行约谈，昆明、贵阳、长春则升级了限售政策。'
ch_text4 = '5月份楼市调控的高潮出现在19日，住建部再次重申坚持房地产调控目标不动摇、力度不放松，并从住房发展规划、住房和土地供应、资金管控、市场监管、落实主体责任等方面做出了明确要求。'
ch_text5 = '调控的目的只有一个，那就是稳定楼市。'

ch_texts = [ch_text1, ch_text2, ch_text3, ch_text4, ch_text5]

corups = [proc_text(ch_text) for ch_text in ch_texts]

X = count_vectorizer.fit_transform(corups)

X.toarray()

new_text = '''面对调控与市场的双面性，人们最关心的是如何做出购房选择。预判楼市如同看待股市，总有两种不同声音。

　　一种声音认为，楼市要涨，再不上车就“晚”了；一种声音认为，楼市求稳，要学会慢慢的“飞”。

　　面对两种声音，较真儿的人总喜欢论出个是非胜负，坦然的人更愿意踏实努力尽力而为。

　　回头看，刚需的人总要上车，被炒的房终会入市。

　　十九大报告已经明确“要加快建立多主体供应、多渠道保障，租购并举的住房制度”。'''

new_pro_text = proc_text(new_text)
proc_text('new_pro_text', new_pro_text)

new = count_vectorizer.transform([new_pro_text]).toarray()
proc_text(new)