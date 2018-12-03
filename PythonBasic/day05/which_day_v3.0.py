'''
    作者：zhangpeng
    时间：01/22/2018
    版本1.0：判断是第几天
'''


from datetime import datetime


def is_leap_year(year):
    '''
        判断是否闰年
    '''
    is_leap = False
    if (year % 400 == 0) or ((year % 4 == 0) and year % 100 != 0):
        is_leap = True
    return is_leap


def main():
    '''
        主函数
    '''
    input_year_str = input('请输入年月日（yyyy-mm-dd）:')
    input_year = datetime.strptime(input_year_str,format('%Y-%m-%d'))
    # print(input_year)
    year = input_year.year
    month = input_year.month
    day = input_year.day

    #创建集合
    _30_days_month_set = {4,6,9,11}
    _31_days_month_set = {1,3,5,7,8,10,12}

    #初始化
    days = 0
    days += day
    for i in range(1,month):
        if i in _30_days_month_set:
            days += 30
        elif i in _31_days_month_set:
            days += 31
        else:
            days += 28
    if ((year % 400 ==0) or ((year % 4 ==0) and year % 100 != 0)) and (month > 2):
        days += 1
    print('这是{}年的第{}天'.format(year,days))



if __name__ == '__main__':
    main()
