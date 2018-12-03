'''
    作者：张鹏
    功能：模拟值塞子
    版本：1.0
    时间：01/24/18
'''

import random


def roll_dict():
    '''
        模拟扔塞子
    '''
    roll = random.randint(1,6)
    return roll


def main():
    '''
        主函数
    '''
    #初始化列表
    result_list = [0] * 6
    # print(result_list)
    total_time = int(input('请输入次数：'))
    for i in range(total_time):
        for j in range(1,7):
            if j == roll_dict():
                result_list[j - 1] += 1
    for i , result in enumerate(result_list):
        print('点数为：{}，出现的次数为：{}，概率为{}'.format(i +1 , result,result / total_time))
if __name__ == '__main__':
    main()


