"""
    practice1
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

random_image = np.random.random([20, 20])
plt.imshow(random_image,cmap='gray')
plt.colorbar()
plt.show()