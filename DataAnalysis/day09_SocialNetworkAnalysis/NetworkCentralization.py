"""
    时间：2018年5月31日
    作者：张鹏
    内容：网络中心势
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
G = nx.convert_node_labels_to_integers(G, first_label=1)
plt.figure()
nx.draw_networkx(G)
# plt.show()

# 1 度中心性
deg_cent = nx.degree_centrality(G)
print(deg_cent)
print(deg_cent[1])

print('节点1的度：', nx.degree(G, 1))
print('网络节点个数：', len(nx.nodes(G)))
print('节点1的度中心性{}/({}-1) = {}'.format(nx.degree(G, 1),
      len(nx.nodes(G)),
      nx.degree(G, 1) / (len(nx.nodes(G)) - 1)))

# 2 接近中心性
close_cent = nx.closeness_centrality(G)
print(close_cent)
print('节点1的接近中心性：', close_cent[1])
print('网络节点个数：', len(nx.nodes(G)))
print('节点1与其他节点的最短路径和：', sum(nx.shortest_path_length(G,1).values()))

# 3 中介中心性
btwn_cent = nx.betweenness_centrality(G, normalized=True, endpoints=False)
import operator
# 按字典的值排序
print('按字典的值排序', sorted(btwn_cent.items(),key=operator.itemgetter(1), reverse=True)[:5])

print('nx.pagerank(G)\n',nx.pagerank(G))
