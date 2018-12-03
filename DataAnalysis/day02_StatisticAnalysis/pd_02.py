import pandas as pd

# 1、数据合并
staff_df = pd.DataFrame([{'姓名': '张三', '部门': '研发部'},
                        {'姓名': '李四', '部门': '财务部'},
                        {'姓名': '赵六', '部门': '市场部'}])
student_df = pd.DataFrame([{'姓名': '张三', '专业': '计算机'},
                        {'姓名': '李四', '专业': '会计'},
                        {'姓名': '王五', '专业': '市场营销'}])
# print(staff_df)
# print()
# print(student_df)
# print('***************************************')

# print(pd.merge(staff_df, student_df, how='outer', on='姓名'))
# 或者
# staff_df.merge(student_df, how='outer', on='姓名')
# print('***************************************')

# print(pd.merge(staff_df, student_df, how='inner', on='姓名'))
# 或者
# staff_df.merge(student_df, how='inner', on='姓名')
# print('***************************************')

# print(pd.merge(staff_df, student_df, how='left', on='姓名'))
# 或者
# staff_df.merge(student_df, how='left', on='姓名')
# print('***************************************')

# print(pd.merge(staff_df, student_df, how='right', on='姓名'))
# 或者
# staff_df.merge(student_df, how='right', on='姓名')

# 也可以按索引进行合并
# staff_df.set_index('姓名', inplace= True)
# student_df.set_index('姓名', inplace= True)
# print(staff_df)
# print(student_df)
# print(pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True))

staff_df.reset_index(inplace=True)
student_df.reset_index(inplace=True)
# print(staff_df)
# print(student_df)

staff_df.rename(columns={'姓名': '职工姓名'}, inplace=True)
student_df.rename(columns={'姓名': '学生姓名'}, inplace=True)
# print(staff_df)
# print(student_df)
# print(pd.merge(staff_df, student_df, how='left', left_on='职工姓名', right_on='学生姓名'))

# 如果两个数据中包含有相同的列名（不是要合并的列）时，merge会自动加后缀作为区别
staff_df['地址'] = ['天津', '北京', '上海']
student_df['地址'] = ['天津', '上海', '广州']
# print(staff_df)
# print(student_df)
# print(pd.merge(staff_df, student_df, how='left', left_on='职工姓名', right_on='学生姓名'))
# 也可指定后缀名称
# print(pd.merge(staff_df, student_df, how='left', left_on='职工姓名', right_on='学生姓名', suffixes=('_家庭', '_公司')))

# 也可以指定多列进行合并，找出同一个人的工作地址和家乡地址相同的记录
# print(staff_df.merge(student_df, how='inner', left_on=['职工姓名', '地址'], right_on=['学生姓名', '地址']))

# apply使用
# 获取姓
# print(staff_df['职工姓名'].apply(lambda x : x[0]))
# 获取名
# print(staff_df['员工姓名'].apply(lambda x: x[1:]))

# 结果合并
staff_df.loc[:, '姓'] = staff_df['职工姓名'].apply(lambda x: x[0])
# staff_df['姓'] = staff_df['职工姓名'].apply(lambda x : x[0])
staff_df.loc[:, '名'] = staff_df['职工姓名'].apply(lambda x: x[1:])
# staff_df['名'] = staff_df['职工姓名'].apply(lambda x: x[1:])
# print(staff_df)

# 2. 数据分组
report_data = pd.read_csv('..//data//2015.csv')
# print(report_data.head())
print('***************************************')
# 按地区进行分组
# groupd = report_data.groupby('Region')
# print(groupd.head())
# 每个地区的幸福指数的平均值
# print(groupd['Happiness Score'].mean())
# 每个地区有多少个城市
# print(groupd.size())

# 迭代对象
# for group, frame in groupd:
#     mean_sorce = frame['Happiness Score'].mean()
#     min_socre = frame['Happiness Score'].min()
#     max_sorce = frame['Happiness Score'].max()
    # print('{}的平均幸福指数为：{}，其中高幸福指数为：{}，最低幸福指数为：{}'
    #       .format(group, mean_sorce, max_sorce, min_socre))

# 自定义函数进行分组
# 按照幸福指数排名进行划分，1-10, 10-20, >20
# 如果自定义函数，操作针对的是index
# report_data2 = report_data.set_index('Happiness Rank')
# print(report_data2.head())
# def get_rank_group(rank):
#     # rank_group = ''
#     if rank <= 10:
#         rank_group = '0 -- 10'
#     elif rank <= 20:
#         rank_group = '10 -- 20'
#     else:
#         rank_group = '> 20'
#     return rank_group

# grouped = report_data2.groupby(get_rank_group)
# print(grouped2.head())
# for group, frame in grouped:
    # print('{}分组的数据个数：{}'.format(group, len(frame)))

# 实际项目中，通常可以先人为构造出一个分组列，然后再进行groupby

# 按照score的整数部分进行分组
# 按照幸福指数排名进行划分，1-10, 10-20, >20
# 如果自定义函数，操作针对的是index
report_data['score gtoup'] = report_data['Happiness Score'].apply(lambda x: int(x))
grouped = report_data.groupby('score gtoup')
for group, frame in grouped:
    print('幸福整数部分为：{}，有{}个人'.format(group, len(frame)))

import numpy as np

print(grouped.agg({'Happiness Score': np.mean, 'Happiness Rank': np.max}))

print(grouped['Happiness Score'].agg([np.mean, np.max, np.min, np.std]))