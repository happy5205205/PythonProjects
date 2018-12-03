# _*_ coding:utf-8 _*_
"""
    作者：张鹏
    版本：v1.0
    日期 2018-11-20 16：59;
    文件命：config.py
    功能 配置文件
"""

import os

# 指定文件路径
dataset_path = '../date'

# 结果保存路径
output_path = '../output'
if not os.path.exists(output_path):
    os.mkdir(output_path)

feat_cols = ['Age','Sex', 'Job', 'Housing','Saving accounts',
             'Checking account', 'Credit amount', 'Duration', 'Purpose']

label_col = ['Risk']

# 类别数据列中，类别整型转换字典
sex_dict = {'male': 0, 'female': 1}

housing_dict = {'free': 0, 'rent': 1, 'own': 2}

saving_dict = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3}

checking_dict = {'little': 0, 'moderate': 1, 'rich': 2}

purpose_dict = {
    'radio/TV':             0,
    'education':            1,
    'furniture/equipment':  2,
    'car':                  3,
    'business':             4,
    'domestic appliances':  5,
    'repairs':              6,
    'vacation/others':      7
}

risk_dict = {
    'bad':  0,
    'good': 1
}


