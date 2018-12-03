#-*- coding:UTF-8 -*-

'''
if True:
    print("风热")·
else:
   print("456")
if True:
    print("Answer")
    print("True")
else:
    print("Answer")
    print("False")
    '''

#字符串使用
'''
str = 'hello world'
print(str)
print(str[0])
print(str[2:5])
print(str[2:])
print(str[::])
print(str*2)
print(str+"你好世界")
'''

#列表使用
list = ['abc',789,2.34,'john',70.2]
tinylist =[123,'jack']
print(list)
print(list[1])
print(list[::])
print(list[2:])
print(list*2)
print(list[::])
print(list+tinylist)
print(list)
list[0] = 'python'
print(list)
print(len(list))
list.append(9999)
print(list)

#元组：元组是类似于列表中的序列数据类型，一个元组由数个逗号分隔的值组成。
#列表和元组之间的区别是：列表是方括号[]，列表的长度和元素是可以改变的，而元组是圆括号（），不能被更新
# tuple = ('abc',789,2.33,'jack',1.2)
# print(tuple)
# tinytuple=(123,'john')
# print(tuple[1],[3])
# print(tuple[1:3])
# print(tuple[2:])
# print(tinytuple*2)
# print(tuple+tinytuple)
# tuple[1]='java'  #元组元素不可被改变
# print(tuple)

#字典
#python字典是一种哈希表型。由“键-值”对组成
#键可以是任何python类型，但通常是数字或字符串
#值可以是任意的python对象
#字典是由花括号{}，可以分配值，并用方括号[]访问，例如：
# dict = {}
# dict['one'] = 'this is one'
# dict[0] = 'this is two'
# tinydict={'name':'jack','age':14,'sex':'boy'}
# print(dict)
# print(dict['one'])
# print(dict[0])
# print(tinydict)
# print(tinydict.keys())
# print(tinydict.values())
# print(type(dict))


#set
# a = {1,4,7,3,6,9}
# b = {2,5,8,1,4,7}
# print(a)
# a.add(10)
# print(a)
# a.remove(1)
# print(a)
# c = a.union(b)
# print(c)

#2的3次方
# c =pow(2,3)
# print(c)

#深拷贝
# dict1 = {1:10,2:20}
# print(dict1)
# dict2 = dict1
# dict1[1]= 30
# print(dict1)
# print(dict2)#引用变量
# dict3 = dict1.copy()#深拷贝
# dict3[1] = 100
# print(dict3)
# print(dict2)
# print(dict1)

