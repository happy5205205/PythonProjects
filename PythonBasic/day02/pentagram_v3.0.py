'''
    作者：张鹏
    时间：01/17/2018
    1.0：功能绘制五角星
    3.0:使用迭代
'''

import turtle


# def draw_pentagram(size):
#     #绘制五角星
#     count = 1
#     while count <= 6:
#         turtle.forward(size)
#         turtle.right(144)
#         count = count + 1


def draw_recursive_pentagram(size):
    #迭代函数
    count = 1
    while count <= 5:
        turtle.forward(size)
        turtle.right(144)
        count = count + 1
    #绘制完五角星 更新参数
    size += 50
    if size <= 300:
        draw_recursive_pentagram(size)


def main():
    '''
    主函数
    '''
    #计数器
    turtle.penup()
    turtle.backward(200)
    turtle.pendown()
    turtle.pensize(2)
    turtle.pencolor('blue')

    size = 100
    draw_recursive_pentagram(size)
    turtle.exitonclick()

if __name__ == '__main__':
    main()