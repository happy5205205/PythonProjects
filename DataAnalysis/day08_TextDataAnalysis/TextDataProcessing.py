"""
    时间：2018年4月25日
    作者：张鹏
    内容：Python文本数据处理
            1. 文本基本操作
            2. 文本清洗
            3. 正则表达式
"""

import nltk


# 1. 文本基本操作
text1 = 'Python is a widely used high-level programming language for ' \
        'general-purpose programming, created by Guido van Rossum and first released in 1991.'
print('text1字符：{}'.format(len(text1)))

# 获取单词
text2 = text1.split(' ')
print('单词个数：{}\n'.format(text2))
# 找出含有3个字母或者长度超过3的单词
print('找出含有3个字母或者长度超过3的单词:\n',[w for w in text2 if len(w) > 3])
# 找出首写字母大写的单词
print('找出首写字母大写的单词:\n', [w for w in text2 if w.istitle()])
# 找出以字母s结尾的单词
print('找出以字母s结尾的单词:\n', [w for w in text2 if w.endswith('s')])
# 找出不重复的单词
text3 = 'TO be or not to be'
text4 = text3.split(' ')
print('单词个数：{}\n不重复单词个数：{}'.format(len(text4),len(set(text4))))
print('不重复单词\n{}'.format(set(text4)))
# 忽略大小写统计
print('忽略大小写统计,不重复单词:\n{}'.format(set([w.lower() for w in text4])))
print('忽略大小写统计,不重复单词个数:\n{}'.format(len(set([w.lower() for w in text4]))))

# 2. 文本清洗
text5 = '            A quick brown fox jumped over the lazy dog.  '
print('split分割：\n{}'.format(text5.split(' ')))
print('strip分割：\n{}'.format(text5.strip(' ')))

text6 = text5.strip(' ')
print('text6:', text6)
# 去掉末尾的换行符
text7 = 'This is a line\n'
print('text7:',text7)
print('11111111111111111111111111')
print('去掉末尾的换行符:', text7.rstrip())
print('11111111111111111111111111')

# 3. 正则表达式
text8 = '"Ethics are built right into the ideals and objectives' \
        ' of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr @UN @UN_Women'
text9 = text8.split(' ')
# 找出特定文本，比如：以#开头
print('找出特定文本，比如：以#开头:\n', [w for w in text9 if w.startswith('#')])
# 以@开头的文本
print('以@开头的文本:\n', [w for w in text9 if w.startswith('@')])
# 根据@后面的字符的样式查找文本
# 样式符合的规则：包含字母，或则数字，或者下划线
import re
print('正则表达式：\n{}'.format([w for w in text9 if re.search('@[A-Za-z0-9_]+', w)]))

text10 = 'ouagadougou'
print(re.findall('[aeiou]', text10))
print(re.findall('[^aeiou]', text10))


