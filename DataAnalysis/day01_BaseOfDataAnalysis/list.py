# 1、list
l = [1, 'a', 2, 'b']
print(type(l))
print('修改前：', l)

# 修改list的内容
l[0] = 3
print('修改后：', l)

# 末尾添加元素
l.append(4)
print('添加后：', l)

# 遍历list
print('遍历list(for循环)：')
for item in l:
    print(item)

# 通过索引遍历list
print('遍历list(while循环)：')
i = 0
while i != len(l):
    print(l[i])
    i += 1

# 列表合并
print('列表合并(+)：', [1, 2] + [3, 4])

# 列表重复
print('列表重复(*)：', [1] * 5)

# 判断元素是否在列表中
print('判断元素存在(in)：', 1 in [1, 2])