# _*_ coding: utf-8 _*_
"""
    作者：张鹏
    时间：2018年12月2日
    文件名：config.py
    功能：配置文件
"""
# 以新闻标题分类。还是以新闻内容分类
classify_type = 'title'

# 数据集zip文件
dataset_file = './data/SogouCS.reduced.zip'

# 数据集解压路劲
dataset_path = './data/SogouCS.reduced'

# 类别描述文件
categories_file = './data/categories.txt'

# 处理数据CSV文件
proc_raw_text_csv_file = './data/proc/raw_text.csv'

# 清洗数据CSV文件
cln_text_csv_file = './data/proc/cln_text.csv'

# top n_category 的类别
n_category = 5

# 词频个数
n_common_words = 500