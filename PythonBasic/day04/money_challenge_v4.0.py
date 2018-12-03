'''
    作者：张鹏
    功能：52周存钱挑战
    版本：1.0

    日期：05/08/2017
'''
import math


# saving = 0  #全局变量

def saving_money_n_week(total_wek,money_per_week,increase_money):
    money_list = [] #记录每周存款数的列表
    # global saving #假如saving是全局变量记得函数内用global声明
    for i in range(total_wek):

        money_list.append(money_per_week)
        saving = math.fsum(money_list)
        # print('第{}周，存入金额为{}，总计为:{}'.format(i+1,money_per_week,saving))

        #更新下周存入数
        money_per_week += increase_money
    # print('函数内的saving',saving)
    return saving
def main():
    '''
        主函数
    '''
    money_per_week = float(input('请输入每周存款金额数：'))    #每周存入金额
    increase_money = float(input('请输入递增金额数：'))       #递增金额
    total_wek = int(input('请输入存钱周数：'))                 #总共周数
    #调用函数
    saving = saving_money_n_week(total_wek,money_per_week,increase_money)
    print('总的存款金额：',saving)

if __name__ == '__main__':
    main()