#!/usr/bin/env/python
#-*- coding:utf-8 -*-

#读取文本
# file_handler = open(r'D:\users.txt','a+')
# file_handler.seek(0)
# line = file_handler.readline()
# while line:
#     print(line)
#     line = file_handler.readline()
# file_handler.close()

#其它文件的IO函数的使用
#！/usr/bin/env/python
#coding = utf-8
# fileHandler = open(r'D:\users.txt','a+')
# fileHandler.seek(0)

#读取整个文件
# contents = fileHandler.read()
# print(contents)

#读取所有行，在逐行输出
# fileHandler.seek(0)
# lines = fileHandler.readline()
# for line in lines:
#     print(line)
# print(fileHandler.tell())
# fileHandler.close()


#!/usr/bin/env/python
#coding = utf-8
fileHandler = open(r'D:\users.txt','a+')
fileHandler.write("\r\n")
fileHandler.write("thank you")
fileHandler.seek(0)
contens = fileHandler.read()
print(contens)
