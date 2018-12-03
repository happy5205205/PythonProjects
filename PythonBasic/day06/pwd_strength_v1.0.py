'''
    作者：张鹏
    时间：01/23/2018
    版本：1.0
    功能：判断密码强弱
'''


def check_number_exist(password):
    '''
        判断字符串中是否含有数字
    '''
    for c in password:
        if c.isnumeric():
            return True
    return False


def check_letter_exist(password):
    '''
        判断字符串是否含有字母
    '''
    for c in password:
        if c.isalpha():
            return True
    return False



def main():
    '''
        主函数
    '''
    password = input('请输入密码：')
    strength_level = 0

    #规则一：长度大于8位
    if len(password) >= 8:
        strength_level += 1
    else:
        print('密码长度至少为8位')

    #规则二：包含数字
    if check_number_exist(password):
        strength_level += 1
    else:
        print('密码要包含数字')

    #规则三：包含字母
    if check_letter_exist(password):
        strength_level += 1
    else:
        print('密码要包含字母')

    if strength_level == 3:
        print('恭喜，密码强度合格')
    else:
        print('密码不合格')
if __name__ == '__main__':
    main()