#1、定时任务

# ! /usr/bin/env python
# coding=utf-8
import time, os, sched


# 第一个参数确定任务的时间，返回从某个特定的时间到现在经历的秒数
# 第二个参数以某种人为的方式衡量时间   
schedule = sched.scheduler(time.time, time.sleep)


def perform_command(cmd, inc):
    #    os.system(cmd)
    print(time.time())
    print('zhixing写入数据库', time.time() - tt)
    global tt
    tt = time.time()


def timming_exe(cmd, inc=60):
    # enter用来安排某事件的发生时间，从现在起第n秒开始启动      
    schedule.enter(inc, 0, perform_command, (cmd, inc))
    #  # 持续运行，直到计划时间队列变成空为止       
    schedule.run()


if __name__ == '__main__':
    tt = time.time()
    print("show time after 5 seconds:", tt)
    timming_exe("echo %time%", 5)

#2、利用sched实现周期调用

# ! /usr/bin/env python
# coding=utf-8
import time, os, sched

# 第一个参数确定任务的时间，返回从某个特定的时间到现在经历的秒数
# 第二个参数以某种人为的方式衡量时间   
schedule = sched.scheduler(time.time, time.sleep)


def perform_command(cmd, inc):
    # 安排inc秒后再次运行自己，即周期运行       
    schedule.enter(inc, 0, perform_command, (cmd, inc))
    #    os.system(cmd)
    print(time.time())
    print('zhixing写入数据库', time.time() - tt)
    global tt
    tt = time.time()


def timming_exe(cmd, inc=60):
    # enter用来安排某事件的发生时间，从现在起第n秒开始启动      
    schedule.enter(inc, 0, perform_command, (cmd, inc))
    #  # 持续运行，直到计划时间队列变成空为止       
    schedule.run()


if __name__ == '__main__':
    tt = time.time()
    print("show time after 5 seconds:", tt)
    timming_exe("echo %time%", 5)