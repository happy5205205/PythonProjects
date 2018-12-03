"""
    数据存储
"""
import os
import pandas as pd

df = pd.read_csv('./data/test.csv', encoding='gb18030')
print(df.head())

