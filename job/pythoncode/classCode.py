"""
    Python从设计之初就已经是一门面向对象的语言，正因为如此，在Python中创建一个类和对象是很容易的。
    本章节我们将详细介绍Python的面向对象编程。
    如果你以前没有接触过面向对象的编程语言，那你可能需要先了解一些面向对象语言的一些基本特征，在头脑里头形成一个基本的面向对象的概念，这样有助于你更容易的学习Python的面向对象编程。
    接下来我们先来简单的了解下面向对象的一些基本特征。

    面向对象技术简介
        1.类(Class): 用来描述具有相同的属性和方法的对象的集合。
            它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。
        2.类变量：类变量在整个实例化的对象中是公用的。类变量定义在类中且在函数体之外。
            类变量通常不作为实例变量使用。数据成员：类变量或者实例变量, 用于处理类及其实例对象的相关的数据。
        3.方法重写：如果从父类继承的方法不能满足子类的需求，可以对其进行改写，
            这个过程叫方法的覆盖（override），也称为方法的重写。
        4.实例变量：定义在方法中的变量，只作用于当前实例的类。
        5.继承：即一个派生类（derived class）继承基类（base class）的字段和方法。
            继承也允许把一个派生类的对象作为一个基类对象对待。例如，有这样一个设计：一个Dog类型的对象派生自Animal类，这是模拟"是一个（is-a）"关系（例图，Dog是一个Animal）。
            实例化：创建一个类的实例，类的具体对象。
        6.方法：类中定义的函数。
        7.对象：通过类定义的数据结构实例。对象包括两个数据成员（类变量和实例变量）和方法。
"""

# _*_ coding: utf-8 _*_

class Employee(object):

    '所有员工的基类'

    empCount = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.empCount += 1

    def displayCount(self):
        print('Total Employee {}'.format(Employee.empCount))

    def dispalyEmployee(self):
        print('Name:{},Salary:{}'.format(self.name, self.salary))

"创建 Employee 类的第一个对象"
emp1 = Employee('张三', 350000)
"创建 Employee 类的第二个对象"
emp2 = Employee('李四', 23000)

emp1.dispalyEmployee()
emp2.dispalyEmployee()
print('Total Employee {}'.format(Employee.empCount))

# __dict__ : 类的属性（包含一个字典，由类的数据属性组成）
# __doc__ :类的文档字符串
# __name__: 类名
# __module__: 类定义所在的模块（类的全名是'__main__.className'，如果类位于一个导入模块mymod中，那么className.__module__ 等于 mymod）
# __bases__ : 类的所有父类构成元素（包含了一个由所有父类组成的元组）

print ("Employee.__doc__:", Employee.__doc__)
print ("Employee.__name__:", Employee.__name__)
print ("Employee.__module__:", Employee.__module__)
print ("Employee.__bases__:", Employee.__bases__)
print ("Employee.__dict__:", Employee.__dict__)