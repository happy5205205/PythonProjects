'''
    作者：张鹏
    功能：模拟值塞子
    版本：1.0
    3.0：数据可视化
    4.0:直方图可视化结果
    5.0:科学计算
    时间：01/24/18
'''

import random
import matplotlib.pyplot as plt
import numpy as np

#解决中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    '''
        主函数
    '''
    total_time = 1000
    roll_arr1 = np.random.randint(1, 7, size = total_time)
    roll_arr2 = np.random.randint(1, 7, size = total_time)

    result_arr = roll_arr1 + roll_arr2
    # print(result_arr)
    hist, bin = np.histogram(result_arr, bins=range(2, 14))
    print(hist)
    print(bin)

    #数据可视化
    # plt.hist(roll_list,bins=range(2,14), edgecolor='black', linewidth=1)
    plt.hist(result_arr,bins=range(2,14), normed=1, edgecolor='black',
             linewidth=1, rwidth = 0.8)
    #设置X轴坐标显示位置

    tick_lable = ['2点', '3点', '4点', '5点', '6点',
                  '7点', '8点', '9点', '10点', '11点', '12点']
    tick_pos = np.arange(2, 13) + 0.5 #向右移动0。5的距离
    print(tick_pos)
    plt.xticks(tick_pos, tick_lable)
    plt.title('骰子点数统计')
    plt.xlabel('点数和')
    plt.ylabel('次数')
    plt.show()
if __name__ == '__main__':
    main()


