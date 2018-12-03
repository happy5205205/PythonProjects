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

    days_in_month_list = [31,28,31,30,31,30,31,31,30,31,30,31]#每个月多少天放入一个元组
    a = days_in_month_list[:month-1]
    print(a)
    # if is_leap_year(year):
    #     days_in_month_list[1] = 29
    #     days = sum(days_in_month_list[:month-1]) + day
    #     print('这是第{}年的第{}天'.format(year,days))
    # else:
    #     days = sum(days_in_month_list[:month-1]) + day
    #     print('这是第{}年的第{}天'.format(year,days))
    #与上面if。。else。。等价
    if is_leap_year(year):
        days_in_month_list[1] = 29
    days = sum(days_in_month_list[:month-1]) + day
    print('这是第{}年的第{}天'.format(year,days))

if __name__ == '__main__':
    main()
