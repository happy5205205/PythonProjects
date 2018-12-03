'''
    作者：张鹏
    功能：52周存钱挑战
    版本：1.0
    日期：05/08/2017
'''


def main():
    '''
        主函数
    '''
    money_per_week = 10   #每周存入金额
    i = 1                 #记录周数
    increase_money = 10   #递增金额
    total_wek = 52        #总共周数
    saving = 0            #账户累计

    while i <= total_wek:
        #存入金额数
        # saving = saving + money_per_week
        saving += money_per_week
        print('第{}周，存入金额为{}，总计为:{}'.format(i,money_per_week,saving))

        #更新下周存入数
        money_per_week += increase_money
        i += 1

if __name__ == '__main__':
    main()