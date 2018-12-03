'''
    作者：张鹏
    版本：3.5.4
    时间：01/16/2018
   2.0:新增判断是不是美元或者是人名币，进行对应的转换计算

'''

#汇率
USD_VS_RMB = 6.77

#人民币的输入,input输入是字符串
currency_str_value =input('请输入带单位的金额：')
#获取货币单位
unit = currency_str_value[-3:]

if unit == 'CNY' or unit == 'cny':
    #输入的是人民币
    rmb_str_value = currency_str_value[:-3]
    # 将字符串转换成数字
    rmb_value = eval(rmb_str_value)
    # print('111111',rmb_value)
    #汇率计算
    usd_value = rmb_value / USD_VS_RMB
    #输出结果
    print('美元（USD）金额是： ',usd_value)

elif unit == 'USD' or unit == 'usd':

    usd_str_value = currency_str_value[:-3]

    usd_value =eval(usd_str_value)

    rmb_value = usd_value * USD_VS_RMB

    print('人民币（USD）金额是： ',rmb_value)
else:
    print("请输入正确的带单位金额")
