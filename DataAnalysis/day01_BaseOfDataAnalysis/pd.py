import pandas as pd
import numpy as np
countries = ['中国', '美国', '澳大利亚']
countries_s = pd.Series(countries)
print(type(countries_s))
print(countries_s)


s = pd.Series(np.random.randint(0, 10, 50))
print(s)
print(len(s))