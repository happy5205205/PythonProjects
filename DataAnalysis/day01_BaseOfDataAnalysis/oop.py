class Person:
    def __init__(self, name, company= '自由职业'):
        self.name = name
        self.company = company

    def set_name(self, name):
        self.name = name

    def set_copamy(self, company):
        self.company = company

p = Person('小张')
print('{}的职业是{}'.format(p.name, p.company))

name = p.set_name('小明')
company = p.set_copamy('数据分析师')
print('{}的职业是{}'.format(p.name, p.company))