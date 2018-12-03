'''
    作者：张鹏
    功能：52周存钱挑战
    版本：1.0

    日期：05/08/2017
'''
import math
import datetime as dt


def saving_money_n_week(total_wek,money_per_week,increase_money):
    money_list = [] #记录每周存款数的列表
    saving_money_list = []
    for i in range(total_wek):

        money_list.append(money_per_week)
        saving = math.fsum(money_list)
        # print('第{}周，存入金额为{}，总计为:{}'.format(i+1,money_per_week,saving))
        saving_money_list.append(saving)
        #更新下周存入数
        money_per_week += increase_money
    # print('函数内的saving',saving)
    return saving_money_list


def main():
    '''
        主函数
    '''
    money_per_week = float(input('请输入每周存款金额数：'))    #每周存入金额
    increase_money = float(input('请输入递增金额数：'))       #递增金额
    total_wek = int(input('请输入存钱周数：'))                 #总共周数
    #调用函数
    saving_money_list = saving_money_n_week(total_wek,money_per_week,increase_money)
    # print('总的存款金额：',saving)
    input_year_str = input('请输入日期（yyyy-mm-dd）：')
    print(input_year_str)
    input_year = dt.datetime.strptime(input_year_str,format('%Y-%m-%d'))#解析日期
    print(input_year)
    #获得第几周
    week_num = list(input_year.isocalendar())[1]  #等价week_num = input_year.isocalendar()[1]
    print(week_num)

    print('第{}周的的存款金额是{}'.format(week_num,saving_money_list[week_num-1]))
if __name__ == '__main__':
    main()