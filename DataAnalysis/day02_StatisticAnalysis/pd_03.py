import pandas as pd
import numpy as np
car = pd.read_csv('..//data//cars.csv')
# print(car.head())
# 我们想要比较不同年份的不同厂商的车，在电池方面的不同
# print(car.pivot_table(values='(kW)', index= 'YEAR', columns='Make', aggfunc=np.mean))

# 我们想要比较不同年份的不同厂商的车，在电池方面的不同
# 可以使用多个聚合函数
print(car.pivot_table(values= '(kW)', index= 'YEAR', columns= 'Make', aggfunc=[np.mean, np.min]))

# 我们想要比较不同年份的不同厂商的车，在电池方面的不同
# 可以使用多个聚合函数
print(car.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean, np.min], margins=True))