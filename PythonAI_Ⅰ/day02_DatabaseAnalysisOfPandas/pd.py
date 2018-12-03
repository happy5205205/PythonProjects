import pandas as pd

# s = pd.Series(10)
# print(s)

df = {'one':[9,8,7,6], 'two':[3,2,1,0]}
a = pd.DataFrame(df)
print(a)
print()
# a[1]
print(a.iloc[1])