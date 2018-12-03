'''
    作者：zhangpeng
    时间：01/22/2018
    版本1.0：判断是第几天
'''


from datetime import datetime


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

    days_in_month_tup = (31,28,31,30,31,30,31,31,30,31,30,31)#每个月多少天放入一个元组
    a = days_in_month_tup[:month-1]
    print(a)

    #判断是否闰年
    if (year % 400 == 0) or ((year % 4 == 0) and year % 100 != 0):
        if month > 2:
            total = sum(days_in_month_tup[:month-1]) + day + 1
            print(total)
    else:
        total1 = sum(days_in_month_tup[:month-1]) + day
        print(total1)
if __name__ == '__main__':
    main()
