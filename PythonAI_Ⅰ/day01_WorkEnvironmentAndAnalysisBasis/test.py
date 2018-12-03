import pylint
import numpy as np

# print([i for i in range(10)])
#
# print({x.upper() for x in 'abcd'})
# pylint.run_epylint()

# r = [1,2,3]
r = np.arange(36).reshape((4, 9))
print(r)
print(r.ndim)
print(r.shape)
print(r.dtype)
print(np.zeros((5,5)).ndim)
# print('r: \n', r)
# print('r[2, 2]: \n', r[2, 2])
# print('r[3, 3:6]: \n', r[3, 3:6])
# print(r[3:5,1:])
# print('ppppp')
# print(np.mean(r[:,2:],axis=1))
# print(np.mean(r[:,2:],axis=1).shape[0])
