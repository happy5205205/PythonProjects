# 使用gridspec和直方图绘制一个复杂分析图
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

x = np.random.random(size=10000)
y = np.random.normal(loc=0., scale=1., size=10000)

plt.figure()
gspec = gridspec.GridSpec(3, 3)

top_hist = plt.subplot(gspec[0, 1:])
side_hist = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])

lower_right.scatter(x, y)
top_hist.hist(x, bins=100, normed=True)
side_hist.hist(y, bins=100, orientation='horizontal', normed=True)
side_hist.invert_xaxis()
plt.show()