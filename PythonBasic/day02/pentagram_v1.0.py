'''
    作者：张鹏
    时间：01/17/2018
    1.0：功能绘制五角星
'''
import turtle

def draw_pentagram(size):
    #绘制五角星
    count = 1
    while count <= 5:
        turtle.forward(size)
        turtle.right(144)
        count = count + 1
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
    while size <=300:
        draw_pentagram(size)
        size += 50

    turtle.exitonclick()

if __name__ == '__main__':
    main()