'''
    作者：张鹏
    时间：01/23/2018
    版本：1.0
    功能：判断密码强弱
    2.0版本：循环的终止
    3.0:保存密码及强度到文件中
    4.0：读取文件的密码
'''


def main():
    #文件读取三种方式
    f = open('password_3.0.txt','r')
    # 1.read 把整个文件内容作为一行字符串输出
    # read = f.read()
    # print(read)

    # 2.readline 每次只输出一行
    # line = f.readline()
    # print(line)

    # 3. readlines把整个文件内容作为一行字符串作为列表输出
    # lines = f.readlines()
    # print(lines)

    for i in f:
        print(i)

    f.close()

if __name__ == '__main__':
    main()



