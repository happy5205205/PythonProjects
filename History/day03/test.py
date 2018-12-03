def t1():
    func1 = [lambda x: x*i for i in range(10)]
    # print(func1)
    result1 = [f1(2) for f1 in func1]
    print(result1)

# my_func = lambda a, b, c: a * b
# print(my_func)
# print(my_func(1, 2, 3))


def t2():
    func2 = [lambda x, i=i: x*i for i in range(10)]
    result2 = [f2(2) for f2 in func2]
    print(result2)


def t3():
    func3 = (lambda x: x*i for i in range(10))
    result3 = [f3(2) for f3 in func3]
    print(result3)

t1()
t2()
t3()
func1 = (lambda x: x*i for i in range(10))
ll = [f(1) for f in func1]
print(type(ll))