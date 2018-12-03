'''
    作者：张鹏
    时间：01/23/2018
    版本：1.0
    功能：判断密码强弱
    2.0版本：循环的终止
    3.0:保存密码及强度到文件中
    6.0:定义文件操作类
'''


class FileTool:
    '''
        文件操作类
    '''
    def __init__(self,filepath):
        self.filepath = filepath

    def wirte_to_file(self,lines):
        '''
            读取文件
        '''
        f = open(self.filepath,'a')
        f.write(lines)
        f.close()

    def open_from_file(self):
        '''
            打开文件
        '''
        f = open(self.filepath,'r')
        read = f.read()
        return read
        f.close()


class PasswordTool:
    '''
        密码工具类
    '''

    def __init__(self,password):
        self.password = password
        self.strength_level =0

    def process_password(self):
         #规则一：长度大于8位
        if len(self.password) >= 8:
            self.strength_level += 1
        else:
            print('密码长度至少为8位')

        #规则二：包含数字
        if self.check_number_exist():
            self.strength_level += 1
        else:
            print('密码要包含数字')

        #规则三：包含字母
        if self.check_letter_exist():
            self.strength_level += 1
        else:
            print('密码要包含字母！')


    def check_number_exist(self):
        '''
            判断字符串中是否含有数字
        '''
        has_number = False
        for c in self.password:
            if c.isnumeric():
                has_number = True
                break
        return has_number

    def check_letter_exist(self):
        '''
            判断字符串是否含有字母
        '''
        has_letter = False
        for c in self.password:
            if c.isalpha():
                has_letter = True
                break
        return has_letter

def main():
    '''
        主函数
    '''
    try_time = 5
    file_path = 'password_6.0.txt'
    # 实例化文件操作类
    filetool = FileTool(file_path)
    # while (try_time > 0) and (try_time <= 5):
    while try_time > 0:
        password = input('请输入密码：')
        #实例化密码工具对象类
        password_tool = PasswordTool(password)
        password_tool.process_password()

        # #将密码强度写入文本
        # strength_level_dit = {1:'弱',2:'一般',3:'强'}
        # for i in strength_level_dit.keys():
        #     if i == password_tool.strength_level:
        #         f = open('password_5.0.txt','a')
        #         f.write('密码：{}，强度：{}\n'.format(password,strength_level_dit[password_tool.strength_level]))
        #         f.close()


        #写操作
        strength_level_dit = {1:'弱',2:'一般',3:'强'}
        for i in strength_level_dit.keys():
            if i == password_tool.strength_level:
                lines = '密码：{}，强度：{}\n'.format(password,strength_level_dit[password_tool.strength_level])
                filetool.wirte_to_file(lines)


        if password_tool.strength_level == 3:
            print('***恭喜，密码强度合格,设置成功！***')
            break
        else:
            print('****密码不合格！*****')
            try_time -= 1

        print()

    if try_time == 0:
        print('输入次数过多，请稍后尝试！')

    #读文本操作
    read = filetool.open_from_file()
    print(read)

if __name__ == '__main__':
    main()