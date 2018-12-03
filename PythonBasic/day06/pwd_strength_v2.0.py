'''
    作者：张鹏
    时间：01/23/2018
    版本：1.0
    功能：判断密码强弱
    2.0版本：循环的终止
'''


#
def check_number_exist(password):
    '''
        判断字符串中是否含有数字
    '''
    has_number = False
    for c in password:
        if c.isnumeric():
            has_number = True
            break
    return has_number


def check_letter_exist(password):
    '''
        判断字符串是否含有字母
    '''
    has_letter = False
    for c in password:
        if c.isalpha():
            has_letter = True
            break
    return has_letter



def main():
    '''
        主函数
    '''
    try_time = 5
    # while (try_time > 0) and (try_time <= 5):
    while try_time > 0:
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
            print('密码要包含字母！')

        if strength_level == 3:
            print('***恭喜，密码强度合格,设置成功！***')
            break
        else:
            print('****密码不合格！*****')
            try_time = try_time - 1
        print()

    if try_time == 0:
        print('输入次数过多，请稍后尝试！')


if __name__ == '__main__':
    main()



