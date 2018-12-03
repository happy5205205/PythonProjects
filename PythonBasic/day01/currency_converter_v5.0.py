'''
    作者：张鹏
    版本：3.5.4
    时间：01/17/2018
    2.0:新增判断是不是美元或者是人名币，进行对应的转换计算
    3.0:程序可以一直运行，直至用户选择退出
    4.0:将汇率兑换功能封装到函数中
    5.0:使程序结构化

'''

'''
税率兑换函数
'''
# def convert_currency(im,er):
#     out_money = im * er
#     return out_money


def main ():
    '''
    主函数
    :return:
    '''

    #汇率
    USD_VS_RMB = 6.77

    currency_str_value = input('请输入带有单位的货币金额：')

    #活的货币种类
    unit = currency_str_value[-3:]

    if unit == 'CNY':
        exchange_rate = 1/USD_VS_RMB
        type_name = '美元'
    elif unit == 'USD':
        exchange_rate = USD_VS_RMB
        type_name = '人民币'
    else:
        exchange_rate = -1

    if exchange_rate != -1:
        in_money = eval(currency_str_value[:-3])
        #使用lambda定义函数
        convert_currency2 = lambda x : x*exchange_rate
        #调用lamba函数
        out_money = convert_currency2(in_money)
        #调用函数
        # out_money = convert_currency(in_money,exchange_rate)
        print('转换成',type_name,'后的金额:',out_money)
    else:
        print('不支持该种货币！')

if __name__ == '__main__':
    main()

