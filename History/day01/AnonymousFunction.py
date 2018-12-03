#可以使用lambda关键字来创建小的匿名函数。这写函数被称为匿名，因为他们不是以标准方式通过是要def关键字声明
#Lambda形式可以采取任何数量的参数，但是形式上只有一个放回值。他们不能包含命令和多个表达式
#匿名函数不能直接调用打印，因为需要lambd表达式
#lambda函数都有自己的命名空间，并且不能访问变量高于在其参数列表和那些在全局命名空间的变量

# #定义
# sum = lambda arg1,arg2:arg1 + arg2  #lambda表达式
# #调用
# print("Value of total: ",sum(10,20))
# print("Value of total:",sum(30,20))

#返回多个值
# tup = lambda x,y:(x+1,y+1)
# c = tup(2,3)
# print(c[::])
# (a,b) = tup(1,4)
# print(a,b)
# print(c[::])

#利用lambda可以实现了类似于scala中的高阶函数效果

def outfunction(func,x,y):
    c = func(x,y)
    print(c)
outfunction(lambda x,y:x+y,1,2)