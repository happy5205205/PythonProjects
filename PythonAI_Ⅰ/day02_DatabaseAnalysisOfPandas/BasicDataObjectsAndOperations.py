# -*- coding: utf-8 -*-
"""
    时间：2018年7月9日
    作者：张鹏
"""
# Series
# 创建series
import pandas as pd
countries = ['中国', '美国', '澳大利亚']
countries_S = pd.Series(countries)
print(countries)
print(countries_S)

countries_dic = {'HC': '中国', 'US': '美国', 'AU': '澳大利亚'}
countries_dic_s = pd.Series(countries_dic)

countries_dic_s.index.name = 'Code'
countries_dic_s.name = 'Country'
print(countries_dic_s)
print(countries_dic_s.index)
print(countries_dic_s.name)

# 处理缺失数据
countries = ['中国', '美国', '澳大利亚', None]
print(pd.Series(countries))
numbers = [4, 5, 6, None]
print(pd.Series(numbers))

# Series索引
countries_dic = {'CH': '中国', 'US': '美国', 'AU': '澳大利亚'}
countries_dic_s = pd.Series(countries_dic)
print('***********************************************\n')
print(countries_dic_s)
# 通过索引判断数据是否存在
# Series可看作定长有序字典
print('CH' in countries_dic_s)
print('NZ' in countries_dic_s)
print('***********************************************\n')
print(countries_dic_s.iloc[1])
print(countries_dic_s.loc['US'])
print(countries_dic_s['US'])
print('***********************************************\n')
print(countries_dic_s.iloc[[0, 2]])
print(countries_dic_s.loc[['CH', 'AU']])

# 向量化操作
import numpy as np
s = pd.Series(np.random.randint(0, 5, 10))
print('***********************************************\n')
print(s.head())
print(len(s))

print(np.sum(s))
print('***********************************************\n')
for label, values in s.iteritems():
    s.loc[label] = values +2
print(s.head())
print('***********************************************\n')
s = pd.Series(np.random.randint(0, 5, 10))
print(s.head())
print('***********************************************\n')
s += 2
print(s.head())
