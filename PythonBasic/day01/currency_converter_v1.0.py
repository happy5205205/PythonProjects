'''
    作者：张鹏
    版本：3.5.4
    时间：01/16/2018
'''

#汇率
USD_VS_RMB = 6.77

#人民币的输入,input输入是字符串
rmb_str_value = input('请输入人民币金额（CNY）：')

#将字符串转换成数字
rmb_value = eval(rmb_str_value)

#汇率计算
usd_value = rmb_value / USD_VS_RMB

#计算输出
print('美元（USD）金额是： ',usd_value)