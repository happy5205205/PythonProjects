my_fun = lambda a, b, c: a * b
print(my_fun)
print(my_fun(1, 2, 3))

#结合map
print('lambda结合map')
l1 = [1, 3, 5, 7, 9]
l2 = [2, 4, 6, 8, 10]

res = map(lambda x, y :x * y, l1, l2)
print(tuple(res))