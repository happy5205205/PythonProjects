'''
    作者：张鹏
    时间：01/18/2018
    功能：计算bmr
'''

def current_bmr(input_str):
    try:
        str_list = input_str.split(' ')
        gender = str_list[0]
        weight = float(str_list[1])
        heigth = float(str_list[2])
        age = int(str_list[3])

        if gender == '男':
            bmr = (13.7 * weight) + (5.0 * heigth) - (6.8 * age) + 66
        elif gender == '女':
            bmr = (9.6 * weight) + (1.8 * heigth) - (4.7 * age) + 655
        else:
            bmr = -1

        if bmr != -1:
            print('基础代谢率: {}大卡'.format(bmr))#字符串格式化输出
        else:
            print(' 输入有误请核对！'
                  '\n **性别：',gender,
                  '\n **身高：',heigth,
                  '\n **体重：',weight,
                  '\n ** 版本年龄：',age,
                  '\n 请重新输入！')

    except IndexError:
        print('请输入完整的信息')
    except ValueError:
        print('请输入正确的值')
    except:
        print('程序异常')


def main():
    '''
    主函数
    '''

    y_or_n = input('是否退出程序y/n:')
    while y_or_n != 'y':
        print('请输入以下信息，用空格分割')
        input_str = input('性别 体重（kg） 身高(cm) 年龄')
        current_bmr(input_str)
        y_or_n = input('是否退出程序y/n:')
    print('程序已退出')

if __name__ == '__main__':
    main()