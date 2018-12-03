import datetime as dt
import time as tm

#从1970年1月1日算起

print('当前时间：', tm.time())

dt_now = dt.datetime.fromtimestamp((tm.time()))
print(dt_now)
print('{}年{}月{}日'.format(dt_now.year, dt_now.month, dt_now.day))

#日期计算
delta = dt.timedelta(days= 100)
print(delta)
print('今天的前100天：', dt.date.today() - delta)
#判断日期大小
print(dt.date.today()> dt.date.today() - delta)
