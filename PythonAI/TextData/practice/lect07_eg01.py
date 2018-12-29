# _*_ coding: utf-8 _*_
"""
    时间：2018年12月10日
    作者：张鹏

"""

# 1 文本的基本操作

text1 = 'Python is a widely used high-level programming ' \
        'language for general-purpose programming, created ' \
        'by Guido van Rossum and first released in 1991.'

print('字符串的个数{}'.format(len(text1)))

# 获取单词
text2 = text1.split(' ')
print('单词个数{}'.format(len(text2)))

# 找出长度超过3的单词
text3 = [w for w in text2 if len(w) > 3]
print('找出长度超过3的单词',text3)

# 找出手写字母大写的单词
text4 = [w for w in text2 if w.istitle()]
print('找出手写字母大写的单词',text4)

# 找出以s结尾的单词
text5 = [w for w in text2 if w.endswith('s')]
print('找出以s结尾的单词',text5)

# 找不不重复的单词
text6 = 'TO be or not to be'
text7 = text6.split(' ')
print('单词个数：{}'.format(len(text7)))
print('不重复单词个数{}'.format(len(set(text7))))

# 忽略大小写
print('忽略大小写{}'.format(set([w.lower() for w in text7])))


# 2. 文本清洗
text8 = '            A quick brown fox jumped over the lazy dog.  '
text8.split(' ')
text9 = text8.strip()
print(text9)

# 去掉末尾的换行符
text10 = 'This is a line\n'
print('去掉末尾换行符{}'.format(text10.rstrip()))

# 正则表达式
text11 = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr @UN @UN_Women'
text12 = text11.split(' ')

# 查找特定文本
# 以#开头的
print('以#开头的', [w for w in text12 if w.startswith('#')])
print('以@开头的', [w for w in text12 if w.startswith('@')])

# 根据@后的字符样式查找文本
# 样式符合规则：包含字母、或数字、或者下划线
import re
print([w for w in text12 if re.search('@[A-Za-z0-9_]+', w)])

text13 = 'ouagadougou'
print(re.findall('[aeiou]', text13))
print(re.findall('[^aeiou]', text13))
