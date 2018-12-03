import numpy as np

# l = np.array([1, 2, 3]) - np.array([4, 5, 6])
# print(type(l))
# print(list(l))
# l1 = np.arange(0, 30, 2)
# print(l1)
# print(l1.reshape(5, 3))

x = np.random.randint(0,10,(1,10))
# print(x)
# print(x.shape)
# print(x.reshape(-1,1))

# a1 = [[1],[2],[3]]
# print(type(a1))
# print(np.shape(np.array(a1)))
#
# a2 = [[1,2,3]]
# print(np.shape(np.array(a2)))
#
# a3 = [1,2,3]
# print(np.shape(np.array(a3)))


def cumulation (a, b):

    c = a*b
    return c

def main():

    a = int(input('please enter  number for a:'))
    b = int(input('please enter number for b:'))
    c = cumulation(a,b)
    print('{}和{}的乘积为：{}'.format(a,b,c))
if __name__ == '__main__':
    main()