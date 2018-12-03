import math

print('示例1，获取两个列表对应位置上的最小值：')
l1 = [1, 4, 7, 3, 6]
l2 = [2, 5, 8, 10, 12]
mins = map(min, l1, l2)
print(mins)

# map()函数操作时，直到访问数据时才会执行
for item in mins:
    print(item)
#
print('示例2，对列表中的元素进行平方根操作：')
squared = map(math.sqrt, l2)
print(squared)
print(list(squared))