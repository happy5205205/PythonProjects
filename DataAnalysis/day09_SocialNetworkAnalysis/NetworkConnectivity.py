"""
    时间：2018年5月29日
    内容：文本网络学习之网络连通性
    作者：张鹏
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


G = nx.Graph()
G.add_edges_from([
    ('A','K'),
    ('A','B'),
    ('A','C'),
    ('B','C'),
    ('B','K'),
    ('C','E'),
    ('C','F'),
    ('C','E'),
    ('D','E'),
    ('E','F'),
    ('E','H'),
    ('F','G'),
    ('I','J')
])
# plt.figure()
# nx.draw_networkx(G)
# # plt.show()
# 1 整体聚类系数
print('整体聚类系数',nx.transitivity(G))

# 2 局部聚类系数
print('节点F的聚类系数：', nx.clustering(G, 'F'))
print('节点A的聚类系数：', nx.clustering(G, 'A'))
print('节点J的聚类系数：', nx.clustering(G, 'J'))

# 3 平均聚类系数
print('平均聚类系数:', nx.average_clustering(G))

# 4 节点间距离
# 去掉AC边
G.remove_edge('A', 'C')
plt.figure()
nx.draw_networkx(G)
plt.show()
print("nx.shortest_path(G, 'A', 'H'):",nx.shortest_path(G, 'A', 'H'))
print("nx.shortest_path_length(G, 'A', 'H'):",nx.shortest_path_length(G, 'A', 'H'))