# -*- coding: utf-8 -*-
"""
    作者：zhangp
    版本：1.0
    时间2018年6月30日

    实战案例1-1：中国五大城市PM2.5数据分析 (1)
    任务：
        - 五城市污染状态
        - 五城市每个区空气质量的月度差异

    数据集来源：https://www.kaggle.com/uciml/pm25-data-for-five-chinese-cities

    案例文档：lect01_proj_readme.pdf
"""

import os
import numpy as np
import csv
from PythonAI_Ⅰ.day01_WorkEnvironmentAndAnalysisBasis import config


def load_data(data_file, usecols):
    """
        加载数据文件
        data_file : 数据路径
        usecoles : 所在的列
        返回 data_arr : 数据的多维数据表示
    """
    data = []
    with open(data_file, 'r') as csv_file:
        data_reader = csv.DictReader(csv_file)
        # 对数据进行清洗
        for row in data_reader:
            # 取出每一行的数据，组成一个列表放入数据列表中
            row_data = []
            # 注意csv模块读入的数据全部为字符串类型
            for col in usecols:
                str_val = row[col]
                # 数据类型转换成float，如果NA，则返回nan
                row_data.append(float(str_val) if str_val != 'NA' else np.nan)
            # 如果行数据中不包含nan才保存该记录
            if not any(np.isnan(row_data)):
                data.append(row_data)
    # 将数据转换成多为的 ndarray
    data_arr = np.array(data)
    return data_arr


def get_polluted_perc(data_arr):
    """
        获取污染占比的小时数
        规则：
            重度污染（heavy）  PM2.5 > 150
            中度污染（middle） 75<PM2.5 <= 150
            轻度污染（light）  35<PM2.5 <= 75
            优质空气（good）   PM2.5< 35
        参数：
            data_arr : 数据的多维数组表示
        返回：
            polluted_perc_list : 污染小时数百分比例表
    """
    # 将每个地区的PM2.5平均后作为该城市小时的PM值
    # 按行取平均值
    hour_var = np.mean(data_arr[:, 2:], axis=1)
    # 总小时数
    n_huor = hour_var.shape[0]
    # 重度污染小时数
    n_heavy_hour = hour_var[hour_var > 150].shape[0]
    # 中度污染小时数
    n_medium_hour = hour_var[(hour_var > 75) & (hour_var <= 150)].shape[0]
    # 轻度污染小时数
    n_light_hour = hour_var[(hour_var > 35) & (hour_var <= 75)].shape[0]
    # 优质空气
    n_good_hour = hour_var[hour_var <= 35].shape[0]

    polluted_perc_list = [n_heavy_hour/n_huor, n_medium_hour/n_huor,
                          n_light_hour/n_huor, n_good_hour/n_huor]
    return polluted_perc_list


def get_avg_pm_per_month(data_arr):
    """
        获取每个区每月的平局PM值
        参数：
            - data_arr ：数据的多维数组表示
        返回
            - results_arr : 多维数组结果
    """
    results = []
    # 获取年份
    years = np.unique(data_arr[:, 0])
    for year in years:
        # 获取当前年份
        year_data_arr = data_arr[data_arr[:, 0] == year]
        # 获取数据的月份
        moth_listr = np.unique(data_arr[:, 1])
        for moth in moth_listr:
            # 获取月的所有数据
            month_data_arr = year_data_arr[year_data_arr[:,1] == moth]
            # 计算当前月份的平均值
            mean_vals = np.mean(month_data_arr[:,2:],axis=0).tolist()
            # 格式化字符串
            row_data = ['{:.0f}-{:02.0f}'.format(year, moth)] + mean_vals
            results.append(row_data)
    results_arr = np.array(results)
    return (results_arr)


def save_stats_to_csv(results_arr, save_file, headers):
    """
    将统计结果保存至csv文件中
    - results_arr: 多维数组结果
    - save_file: 保存文件路径
    - headers: cvs表头
    """
    with open(save_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in results_arr:
            writer.writerow(row)


def main():
    """
        主函数
    """
    print('----------------数据预览和数据处理----------------')
    for city_name, (file_name, cols) in config.data_config_dict.items():
        date_file = os.path.join(config.data_path, file_name)
        usecols = config.common_cols + ['PM_' + col for col in cols]
        data_arr = load_data(date_file, usecols)
        print('{}共有{}条有效数据'.format(city_name, data_arr.shape[0]))
        # print('{}共有{}条数据'.format(city_name, len(data_arr))
        print('{}的前十条预览为\n'.format(city_name))
        print(data_arr[:3])

        print('----------------数据分析----------------')
        # 五个城市的污染状态，统计污染小时的占比
        polluted_statu_list = []
        polluted_perc_list = get_polluted_perc(data_arr)
        polluted_statu_list.append([city_name]+polluted_perc_list)
        print('{}的污染小时数占比{}'.format(city_name,polluted_statu_list))

        # 每个城市每个区空气质量的月度差异，分析计算每个月每个区的平均PM值
        results_arr = get_avg_pm_per_month(data_arr)
        print('{}城市每月平均PM值预览前10'.format(city_name))
        print(results_arr[:10])

        print('----------------结果展示----------------')
        # 保存月度统计到CSV
        save_filename = city_name + '_moth_stats.csv'
        save_file = os.path.join(config.output_path, save_filename)
        save_stats_to_csv(results_arr, save_file, headers=['month'] + cols)
        print('月统计结果已保存至{}'.format(save_file))
        print()
    # 污染状态结果保存
    save_file = os.path.join(config.output_path, 'polluted_percentage.csv')
    with open(save_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['city', 'heavy', 'middle', 'light', 'good'])
        for row in polluted_statu_list:
            writer.writerow(row)
    print('污染状态结果已保存至{}'.format(save_file))

if __name__ == '__main__':
    main()
