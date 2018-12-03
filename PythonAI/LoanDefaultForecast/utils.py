"""
    作者：张鹏
    时间：2018年11月21日
"""
import matplotlib.pyplot as plt
import seaborn as sns
import PythonAI_Ⅱ.LoanDefaultForecast.config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time
import numpy as np

def clean_date(raw_data):
    """
        数据清洗
        参数
            - raw_data 原始数据
        返回
            - cln_date 清洗后的数据原始数据
    """
    # 替换数据中的空数据为'moderate'
    # 数据中只有Saving accounts和Checking account存在空值
    cln_data = raw_data.fillna('moderate')
    return cln_data

def inspect_dataset(train_data, test_data):
    """
        查看数据集
        参数
            - train_data：   训练数据集
            - test_data：    测试数据集
    """
    print('\n===================== 数据查看 =====================')
    print('训练集有{}条数据'.format(len(train_data)))
    print('测试集有{}条数据'.format(len(test_data)))

    # 可视化各类别的数量统计图
    plt.figure(figsize=(10, 5))
