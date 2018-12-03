#whiie循环
# count = 4
# while count <5:
#     print(count,"is less than 5")
#     count = count + 1
# else:
#     print(count,"is not less than 5")

#for循环
#求素数
# for num in range(10,20):
#     # print(num,end='  ')
#     for i in range(2,num):
#         if num%i ==0:
#             j=num/i
#             print('%d equals %d*%d'%(num,i,j))
#             break
#     else:
#         print(num,'is a prime number')

#遍历集合
r = range(10,20)
for num in r:
    print(num ,end='  ')
r = {1,2,3,4,5}
for num2 in r:
    print(num2)
r = ['aaa',3,3.4,'c']
for num3 in r:
    print(num3)
