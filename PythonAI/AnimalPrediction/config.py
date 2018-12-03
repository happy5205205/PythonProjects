# _*_coding: utf-8 _*_
"""
    时间：2018年11月28日
    作者：张鹏
    功能：动物识别数据
    文件名：config.py
"""
import os

# 指定数据集路径

data_path = './data'

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.mkdir(output_path)

# 类别型特征
category_cols = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'tail', 'domestic', 'catsize']

# 数值型特征
num_cols = ['legs']

# 标签列
label_col = ['class_type']

# 需要读取的列
all_cols = category_cols + num_cols + label_col