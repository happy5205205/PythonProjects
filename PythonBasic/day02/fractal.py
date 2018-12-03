'''
    作者：张鹏
    时间：01/17/2018
    1.0：功能绘制五角星
    3.0:使用迭代
    4.0:使用递归绘制分形树
'''
import turtle


def draw_branch(branch_legth):
    '''
    绘制分形数
    '''
    if branch_legth > 5:
        #绘制右侧树枝
        turtle.forward(branch_legth)
        print('向前',branch_legth)
        turtle.right(20)
        print('右转20')
        draw_branch(branch_legth-15)#自己调用自己

        #绘制左侧树枝
        turtle.left(40)
        print('左转 40')
        draw_branch(branch_legth - 15)#自己调用自己

        #返回之前的树枝
        turtle.right(20)
        print('右转20')
        turtle.backward(branch_legth)
        print('回退',branch_legth)


def main():
    '''
    主函数
    '''
    #计数器

    turtle.left(90)
    turtle.penup()
    turtle.backward(200)
    turtle.pendown()
    turtle.pensize(2)
    turtle.pencolor('blue')

    draw_branch(100)
    turtle.exitonclick()

if __name__ == '__main__':
    main()