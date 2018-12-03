#coding = utf-8
# import day01_BaseOfDataAnalysis.Test_module_a
# day01_BaseOfDataAnalysis.Test_module_a.print_func("haha")

from day01.Test_module_a import *
sum_func(2,3)

from day01.Test_module_a import print_func as pf
pf("xixi")

# print('in module b: ',__name__)