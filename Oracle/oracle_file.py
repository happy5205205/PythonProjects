"""
    连接到oracle数据库
"""
import cx_Oracle
import pandas as pd

conn = cx_Oracle.connect('hr/hr@127.0.0.1:1521/orcl')
cursor = conn.cursor()
sql = 'select * from employees -- where employee_id = 100'
cursor.execute(sql)
data = cursor.fetchall()
df = pd.Series(data)
print(df)
print(type(df))
# print(data)
# print(len(data))
cursor.close()
conn.commit()
conn.close()
