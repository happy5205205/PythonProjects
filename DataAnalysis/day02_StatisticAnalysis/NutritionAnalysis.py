'''
    时间：2018/02/08
    作者：张鹏
    任务：按单品类型分析查看数据
         按菜单类型分析查看数据
         查看分析单品及菜单的份量
'''


import os
import pandas as pd
import pylint


# 指定数据路径
path = './data'
date_file = os.path.join(path, 'menu.csv')

#需要与查看的指定列
used_cols = ['Calories', 'Calories from Fat', 'Total Fat', 'Cholesterol', 'Sugars']

def inspect_date(df_date):
    '''
        查看原始数据的基本信息
    '''
    print('*****************************数据预览*****************************')
    print(df_date.head())
    print('*****************************查看数据基本信息*****************************')
    print(df_date.info())
    print('*****************************数据基本统计信息*****************************')
    print(df_date.describe())


def main():
     # 读取数据
    menu_data = pd.read_csv(date_file)
     # 查看数据基本信息
    inspect_date(menu_data)
    # 任务1. 按单品类型分析查看数据
    print('\n===================== 任务1. 按单品类型分析查看数据 =====================')
    print('\n===================== 营养成分最高的单品: =====================')
    max_idxs = [menu_data[clo].argmax() for clo in used_cols]
    for col, max_idxs in zip(used_cols, max_idxs):
        print('{}单品中最高的是:{}'.format(col, menu_data.iloc[max_idxs]['Item']))

    min_idxs = [menu_data[clo].argmin() for clo in used_cols]
    for col, min_idxs in zip(used_cols, min_idxs):
        print('{}单品中最低的是:{}'.format(col, menu_data.iloc[min_idxs]['Item']))

    # 任务2. 按菜单类型分析查看数据
    print('\n===================== 任务2. 按菜单类型分析查看数据 =====================')
    print('\n===================== 菜单类型的单品数目分布: =====================')
    cat_grouped = menu_data.groupby('Category')
    print(cat_grouped.size().sort_values(ascending=False))

    # 查看菜单类型的营养分布平均值
    print(cat_grouped[used_cols].mean())

    # 营养成分最高的菜单类型
    max_idxs = [cat_grouped[col].mean().argmax() for col in used_cols]
    for col, max_idxs in zip(used_cols, max_idxs):
        print('{}营养成分最高的菜单类型是：{}'.format(col, max_idxs))

    print()
    min_idxs = [cat_grouped[col].mean().argmin() for col in used_cols]
    for col, min_idx in zip(used_cols, min_idxs):
        print('{}营养成分最低的菜单类型是：{}'.format(col, min_idx))

    # 任务3. 查看分析单品及菜单的份量
    print('\n===================== 任务3. 查看分析单品及菜单的份量 =====================')
    # 过滤数据，只保留包含 'g'的单品
    serving_size = menu_data[menu_data['Serving Size'].str.contains('g')].copy()
    # print(Serving_Size.head())
    def proc_size_str(size_str):
        '''
            处理Serving Size的字符串返回g
        '''
        start_idx = size_str.index('(') + 1
        later_idx = size_str.index('g')
        size_val = size_str[start_idx: later_idx]
        return float(size_val)
    serving_size['size'] = serving_size['Serving Size'].apply(proc_size_str)
    print(serving_size.head())

    # 份量最多的单品
    max_idxs = serving_size['size'].argmax()
    print('份量最多的单品是{}，有{}g'.format(serving_size.iloc[max_idxs]['Item'], serving_size['size'].max()))
    min_idxs = serving_size['size'].argmin()
    print('份量最少的单品是{}，有{}g'.format(serving_size.iloc[min_idxs]['Item'], serving_size['size'].min()))

    serving_size_ground = serving_size.groupby('Category')
    print('份量中含量最多{}，有{}g'.format(serving_size_ground['size'].mean().argmax(),
                                  serving_size_ground['size'].mean().max()))
    print('份量中含量最少{}，有{}g'.format(serving_size_ground['size'].mean().argmin(),
                                  serving_size_ground['size'].mean().min()))
    # 将数据写入文件，保存文件
    serving_size.to_csv('./serving_size.csv', index=False)
    # pylint.run_pylint()
if __name__ == '__main__':
    main()