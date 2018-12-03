# 元组

t = (1, 'a', 2, 'b')
print(type(t))
print(t)

#元组的内容不能修改，否则会报错
# t[0] = 3

# 遍历tuple
print('遍历list(for循环)：')
for item in t:
    print(item)

# 通过索引遍历tuple
print('遍历tuple(while循环)：')
i = 0
while i != len(t):
    print(t[i])
    i += 1

# 解包 unpack
a, b, c, d = t
print('unpack: ', c)

# 确保unpack接收的变量个数和tuple的长度相同，否则报错,
# 假如长度为四，你只想两个参数，其余用下划线表示 a, b, _, _ = t
# 经常出现在函数返回值的赋值时
# a, b, c = t
