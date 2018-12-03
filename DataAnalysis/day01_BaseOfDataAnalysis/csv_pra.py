import csv

with open('grades.csv') as csvfile:
    grades_data = list(csv.DictReader(csvfile))

print('记录个数：', len(grades_data))
print('前2条记录：', grades_data[:2])
print('列名：', list(grades_data[0].keys()))

avg_assign1 = sum(float(row['assignment1_grade']) for row in grades_data) / len(grades_data)
print('assignment1平均分数：', avg_assign1)


avg_assign = sum(float(row['assignment1_grade'])for row in grades_data) / len(grades_data)

date = set(row['assignment1_submission'][:7] for row in grades_data)
print(date)
l = [1,2,3,4]
l.append(5)
print(l)