'''
    作者：张鹏
    功能：模拟值塞子
    版本：1.0
    3.0：数据可视化
    时间：01/24/18
'''

import random
import matplotlib.pyplot as plt


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
    #初始化点数列表
    result_list = [0] * 11
    roll_list = list(range(2,13))
    #转换成字典
    roll_dict = dict(zip(roll_list , result_list))
    # print('1111111111',roll_dict)

    #保存两个塞子的点数
    roll1_list = []
    roll2_list = []

    total_time = int(input('请输入次数：'))
    for i in range(total_time):
        roll1 = roll_dice()
        roll2 = roll_dice()
        roll1_list.append(roll1)
        roll2_list.append(roll2)

        for j in range(2,13):
            if j == roll1 + roll2:
                roll_dict[j] += 1
    for i, result in roll_dict.items():
        print('点数为：{}，出现的次数为：{}，概率为{}'.format(i +1, result, result / total_time))

    #数据可视化
    x = range(1, total_time + 1)
    plt.scatter(x, roll1_list, c = 'red', alpha=0.5)
    plt.scatter(x, roll2_list, c = 'blue', alpha=0.5)
    plt.show()
if __name__ == '__main__':
    main()


