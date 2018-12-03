# coding:utf-8
"""
    对于一个给定的序列 a = [a1, a2, a3, … , an]，请设计一个算法，用于输出这个序列的全部排列方式。
    例如：a = [1, 2, 3]
    则输出为：
            [1, 2, 3]
            [1, 3, 2]
            [2, 1, 3]
            [2, 3, 1]
            [3, 2, 1]
            [3, 1, 2]
"""


def func(list, start):
    '''

        如果对于无重复元素的序列，可以用如下方法递归实现

    '''
    if start == len(list):
        print(list)
    for i in range(start,len(list)):
        list[i], list[start]=list[start],list[i]
        func(list,start+1)
        list[i], list[start] = list[start], list[i]



def main():
    list = [1, 2, 3]
    func(list, 0)
if __name__ =='__main__':
    main()