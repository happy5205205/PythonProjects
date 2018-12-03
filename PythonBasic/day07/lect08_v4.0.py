'''
    作者：张鹏
    功能：模拟值塞子
    版本：1.0
    3.0：数据可视化
    4.0:直方图可视化结果
    时间：01/24/18
'''

import random
import matplotlib.pyplot as plt


#解决中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def roll_dice():
    '''
        模拟扔塞子
    '''
    roll = random.randint(1,6)
    return roll


def main():
    '''
        主函数
    '''
    #保存两个塞子和的点数
    roll_list = []

    total_time = int(input('请输入次数：'))
    for i in range(total_time):
        roll1 = roll_dice()
        roll2 = roll_dice()
        roll_list.append(roll1 + roll2)


    #数据可视化
    # plt.hist(roll_list,bins=range(2,14), edgecolor='black', linewidth=1)
    plt.hist(roll_list,bins=range(2,14), normed=1, edgecolor='black', linewidth=1)
    plt.title('骰子点数统计')
    plt.xlabel('点数和')
    plt.ylabel('次数')
    plt.show()
if __name__ == '__main__':
    main()


