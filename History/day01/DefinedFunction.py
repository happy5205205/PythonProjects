#定义函数
# def changeme(mylist):
#     "this change a passed list into this function"
#     mylist.append([1,2,3,4]);
#     print("value inside the function :" ,mylist)
#     return (mylist)
# #调用函数
# mylist = [10,20,20,30,40];
# changeme(mylist)

#默认参数
#有默认值的参数后面不能在跟无默认值的参数
# def printinfo(name,age = 35):
#     "this prints a passed info into this function"
#     print("Name:",name)
#     print("Age:",age)
#     return;
# #调用
# # name = "jack"
# # printinfo(name)
# #如果调换了参数顺序，则必须把参数名都带上
# printinfo(age = 50,name ='marry')

#可变参数
def printinfo1(arg,*vartuple):
    "this prints a variable passed arguments"
    print("Output is :")
    print(arg)
    for var in vartuple:
        print(var)
    print(type(vartuple))
    return;
# #调用
# # printinfo1(10);
printinfo1(70,20,50)
# print(type(vartuple))