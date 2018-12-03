'''
    时间：2018/2/9
    作者：张鹏
    功能：单变量分析
         变量间的关系分析
'''

import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import os
import uni_varable_plot

# 解决matplotlib显示中文问题
# 仅适用于Windons
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号‘-’显示为方块问题

# 指定数据路径
data_path = './data'
data_file = os.path.join(data_path, 'pokemon.csv')


def main():
    pokemon_data = pd.read_csv(data_file)
    # print(pokemon_data.head())
    print('*****************************任务一：单变量分析*****************************')
    uni_varable_plot.plot_type1(pokemon_data)
    uni_varable_plot.plot_Egg_Group(pokemon_data)
    uni_varable_plot.plot_other(pokemon_data)
    uni_varable_plot.plot_count_num(pokemon_data)

    print('*****************************任务二：变量间的任务分析*****************************')
    numeric_cols = ['Total', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Pr_Male',
                    'Height_m', 'Weight_kg', 'Catch_Rate']
    plt.figure(figsize=(12, 6))
    sns.pairplot(pokemon_data.dropna(), vars=numeric_cols, size=1)
    plt.title('两变量之间的相关系')
    plt.savefig('./变量间关系图.jpg')
    plt.show()

    # 计算两变量之间的相关系，观察变量之间的关系
    corr_df = pokemon_data[numeric_cols].corr()
    # print(corr_df)
    plt.figure()
    # # 关闭格子线
    ax = plt.gca()
    ax.grid(False)
    plt.imshow(corr_df, cmap='jet')
    plt.title('两变量之间的相关系')
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation= 'vertical')
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./变量关系热力图.jpg')
    plt.show()

if __name__ == '__main__':
    main()