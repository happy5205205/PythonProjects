import time
import datetime
t = time.time() #获取当前时间戳
print(t)
print(datetime.datetime.fromtimestamp(t))


for i in range(10):
    print(i)