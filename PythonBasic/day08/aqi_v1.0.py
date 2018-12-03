'''
    作者：张鹏
    时间：01/25/18
    版本1.0
    功能：计算AQI
'''


def cal_pm_iaqi(pm_val):
    pass


def cal_co_iaqi(co_val):
    pass


def cal_api(param_list):
    pm_val = param_list[0]
    co_val = param_list[2]

    #调用函数计算每个的iaqi
    pm_iaqi = cal_pm_iaqi(pm_val)
    co_iaqi = cal_co_iaqi(co_val)

    aqi_val= []
    aqi_val.append(pm_val)
    aqi_val.append(co_val)

    return max(aqi_val)





def main():
    '''
        主函数
    '''
    print('请输入以下信息，用空格分开')
    input_str = input('(1)PM2.5： （2）CO：')
    str_list = input_str.split(' ')
    print(str_list)
    pm_val = float(str_list[0])
    co_val = float(str_list[1])
    param_list = []
    param_list.append(pm_val)
    param_list.append(co_val)
    # print('11111111', param_list)
    #调用函数
    aqi = cal_api(param_list)
    print('空气质量为：｛｝'.format(aqi))



if __name__ == '__main__':
    main()


