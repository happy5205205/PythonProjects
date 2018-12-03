"""
    数据分析中常用的python技巧
    时间：2018年6月27日
    作者：张鹏
"""

# # 普通表达式
# import math
# def get_log(x):
#     """
#         x:为输入
#         Y:为输出
#     """
#     if x>0:
#         y = math.log(x)
#     else:
#         y=float('nan')
#     return y
#
# y1= get_log(0)
# print(y1)
#
# #  条件表达式
# x = int(input('Enter you number '))
# y2 = math.log(x) if x > 0 else float('nan')
# print(y2)

# 2 列表推导式
l=[]
for i in range(1000):
    if i % 2 == 0:
       l.append(i)

# print(l)

y = [i for i in range(1000) if i % 2 !=0]
# print(y)

# list
l = [1,'a',2,'b']
print('11111',l[0])
print('11111',l[1])
print('11111',l[-1])
print('22',l[:2])
print(len(l))
l[0]= 333
print(len(l))
l.append(555)
print(l)
for i in l:
    print(i)
# 通过索引遍历list
for i in range(len(l)):
    print(l[i])

print([l[i] for i in range(len(l)) ])

print([1,2]+[3,4])
print(i in [1,2])


# tuple
t = (1,'a',2,'b')

c, b, _, _= t
print('unpack', c)

# dictionary 字典
d = {'小象学院': 'http://www.chinahadoop.cn/',
    '百度': 'https://www.baidu.com/',
    '阿里巴巴': 'https://www.alibaba.com/',
    '腾讯': 'https://www.tencent.com/'}

for i in d.keys():
    print(i)

for i in d.values():
    print(i)
for i ,j in d.items():
    print(i,":",j)

print(len(d.keys()))

# {i,":",j for i, j in d.items()}

# set
my_set = {1, 2, 3}
print(my_set)
my_set1 = set([1,2,3,4,3,1])
print(my_set1)
my_set.add(3)
print(my_set)
my_set.add(4)
print(my_set)
my_set.update([7,8,9])
print(my_set)


import collections
c1 = collections.Counter(['a','b','c',1,2,1,2,2,2])
c2 = collections.Counter({'a':2,'b':3,'c':1})
c3 = collections.Counter(a=1,b=2,c=3)
print(c1)
print(c2)
print(c3)

# 更新内容
# 注意这是做加法不是替换
c1.update({'a':5, 'b':5,1:5,'c':10,2:1})
print(c1)
d = {'a':10, 'b':5,1:50,'c':10,2:100}
# 访问内容
print('a=', c1['a'])
print(d['a'])
print('a=', c1['e'])
# element方法
# for element in c1.elements():
    # print(element)

# most_common
print(c1.most_common())

# 5. defaultdict
# 统计每个字母出现的次数
s = 'chainshoop'
# 使用 collection中的counter方法
num = collections.Counter(s).most_common()
print(num)

# 使用dict
counter = {}
for c in s:
    if c not in counter:
        counter[c] = 1
    else:
        counter[c] +=1
print(counter)
print(counter.items())

# 使用defaultdict
counter2 = collections.defaultdict(int)
for c in s :
    counter2[c] += 1
print(counter2)
print(counter2.items())

# 记录相同元素的列表
colors = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = counter3 = collections.defaultdict(list)
for k, v in colors:
    d[k].append(v)
print(d.items())
print(colors[1][0])

# map()函数

print('示例1获取两个列表对应位置的最小值')
l1 = [1, 3, 7, 4, 9]
l2 = [2, 4, 6, 8, 10]
mins = map(min, l1, l2)
print(mins)
for mins in mins:
    print(mins)

print('示例2：对列表中的数据进行平方根操作')
import math
m = map(math.sqrt, l1)
print(m)
print(list(m))

m3 = map(math.log, l1)
print('m3',list(m3))

# 匿名函数
my_func = lambda a, b, c : a*b+c
print(my_func)
print(my_func(1,2,3))

# 结合map使用
l1 = [1, 3, 7, 4, 9]
l2 = [2, 4, 6, 8, 10]
print('map结合lambda使用')
res = map(lambda x,y: x+y, l1,l2)
print(list(res))

# python 操作csv文件
import csv
with open('./data/grades.csv') as csvfile:
    grades_data = list(csv.DictReader(csvfile))
print('记录个数：', len(grades_data))
print('前2条记录：', grades_data[:2])
print('列名：', list(grades_data[0].keys()))

arg = sum(float(row['assignment2_grade'])  for row in grades_data)/len('assignment2_grade')
print(arg)

print(set([row['assignment1_submission'][:7] for row in grades_data]))
