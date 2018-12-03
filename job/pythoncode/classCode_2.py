# _*_ coding: utf-8 _*_

"""
    类的继承
        面向对象的编程带来的主要好处之一是代码的重用，实现这种重用的方法之一是通过继承机制。
        通过继承创建的新类称为子类或派生类，被继承的类称为基类、父类或超类。
    继承语法
        class 派生类名(基类名)
        class A:        # 定义类 A
            .....

        class B:         # 定义类 B
            .....

        class C(A, B):   # 继承类 A 和 B
            .....
    使用issubclass()或者isinstance()方法来检测。
        issubclass() - 布尔函数判断一个类是另一个类的子类或者子孙类，语法：issubclass(sub,sup)
        isinstance(obj, Class) 布尔函数如果obj是Class类的实例对象或者是一个Class子类的实例对象则返回true。

    方法重写
        如果你的父类方法的功能不能满足你的需求，你可以在子类重写你父类的方法：

"""
class Parent: # 定义父类
    parentAttr = 100

    def __init__(self):
        print('调用父类构造函数')

    def parentMethod(self):
        print('调用父类方法')

    def myMethod(self):
        print('父类的方法会被子类重写')

    def setAttr(self, attr):
        Parent.parentAttr = attr

    def getAttr(self):
        print('获得父类的属性{}'.format(Parent.parentAttr))

class Child(Parent):
    def __init__(self):
        print('调用子构造函数')

    def childMethod(self):
        print('调用子类方法')

    def myMethod(self):
        print('子类继承父类之后重写父类方法')

c = Child()
p = Parent()
c.childMethod()
c.parentMethod()
c.getAttr()
c.setAttr(200)
c.getAttr()
p.myMethod()
c.myMethod()