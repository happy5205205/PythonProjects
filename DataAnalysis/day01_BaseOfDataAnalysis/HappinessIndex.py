'''
    作者：张鹏
    时间：2018-02-01
    任务：3.1 查看幸福指数分布情况
*        3.2 统计分析区域的幸福指数
*        3.3 比较两年间的排名变化情况
'''

import os
import csv
import pandas as pd
import numpy as np


# 指定数据集路径
dataset_path = '../data'
report_2015_datafile = os.path.join(dataset_path, '2015.csv')
# print(report_2015_datafile)
report_2016_datafile = os.path.join(dataset_path, '2016.csv')
# 读入数据
def load_data(data_file):
    """
        读取数据文件，加载数据。
        返回列表，其中列表中的每个元素为一个元组，包括Country, Region, Happiness Rank和Happiness Score
    """
    with open(data_file, 'r') as csvfile:
        data_reader = csv.DictReader(csvfile)

        report_2015_data = [(row['Country'], row['Region'],
                             row['Happiness Rank'], row['Happiness Score'])
                            for row in data_reader]
        return report_2015_data

# 调用函数
report_2015_data = load_data(report_2015_datafile)
report_2016_data = load_data(report_2016_datafile)
# 数据预览
print('2015年报告，前10条记录预览：')
print(report_2015_data[:10])

print('2016年报告，前10条记录预览：')
print(report_2016_data[:10])

# 查看幸福指数,使用列表推导式
happnicess_2015_socre = [float(item[3]) for item in report_2015_data]
happnicess_2016_socre = [float(item[3]) for item in report_2016_data]
print('2015年报告，前5条记录幸福指数：',happnicess_2015_socre[:5])
print('2016年报告，前5条记录幸福指数：',happnicess_2016_socre[:5])
# print('2015年报告，前10条记录幸福指数：', [float(item[3]) for item in report_2015_data][:10])
# print('2016年报告，前10条记录幸福指数：', [float(item[3]) for item in report_2016_data][:10])

# 使用numpy.histogram查看数据的直方图分布
hist_2015, hist_edge_2015 = np.histogram(happnicess_2015_socre)
hist_2016, hist_edge_2016 = np.histogram(happnicess_2016_socre)

print('2015年报告，幸福指数直方图分布：{}；直方图边界：{}。'.format(hist_2015, hist_edge_2015))
print('2016年报告，幸福指数直方图分布：{}；直方图边界：{}。'.format(hist_2016, hist_edge_2016))

def get_region_happniness_scores(report_date):
    '''
        获取区域包含国家的幸福指数
    '''

    region_score_dict = {}
    for item in report_date:
        region = item[1]
        score = float(item[3])
        if region in region_score_dict:
            region_score_dict[region].append(score)
        else:
            region_score_dict[region] = [score]
    return region_score_dict

region_2015_score_dict = get_region_happniness_scores(report_2015_data)
print(region_2015_score_dict)
region_2016_score_dict = get_region_happniness_scores(report_2016_data)
# 遍历字典数据
print('2015年报告')
for region, score in region_2015_score_dict.items():
    print('{}：最大值：{}，最小值{}，平均值：{}，中间值：{}'.format(region, np.max(score),
                                                 np.min(score), np.mean(score), np.median(score)))
print('**************************************************************************************************')
print('2016年报告')
for region, score in region_2016_score_dict.items():
    print('{}：最大值：{}，最小值{}，平均值：{}，中间值：{}'.format(region, np.max(score),
                                                 np.min(score), np.mean(score), np.median(score)))

# 比较两年间的排名变化情况

# 将数据构成字典，key是国家，value是排名
county_2015_dict = {item[0]: int(item[2]) for item in report_2015_data}
county_2016_dict = {item[0]: int(item[2]) for item in report_2016_data}
# print(county_2015_dict)
ser_2015 = pd.Series(county_2015_dict)
# print(ser_2015)
ser_2016 = pd.Series(county_2016_dict)
# print(ser_2016)

# 将两年的记录相减，即得出排名变化情况
# 注意Series在进行计算时，是按照index的顺序进行计算的，所以不需要担心顺序问题
# NaN表示某一年的记录缺失，无法计算
ser_change = ser_2016 - ser_2015
print('2015-2016排名变化：')
print(ser_change)

# 查看中国幸福指数排名
print('中国幸福指数排名{}'.format(ser_change['China']))

# 幸福指数上升最快的国家
print('幸福指数上升最快的国家{},指数为：{}'.format(ser_change.argmax(), ser_change[ser_change.argmax()]))

# 幸福指数下降最快的国家
print('幸福指数下降最快的国家{},指数为：{}'.format(ser_change.argmin(),ser_change[ser_change.argmin()]))




