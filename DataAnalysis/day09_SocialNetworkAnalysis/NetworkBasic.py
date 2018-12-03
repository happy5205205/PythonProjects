"""
    时间：2018年5月12日
    作者：张鹏
    内容：文本网络基础学习
"""
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1、创建图

# 1.1 添加节点
G = nx.Graph()
# 添加单个节点
G.add_node(1)
# 添加节点列表
G.add_nodes_from([2,3])
print('查看节点：',G.nodes())
# 1.2 添加边
# 添加指定节点的边
G.add_edge(1,2)
# 以元组形式添加
e = (2, 3)
G.add_edge(*e)
# 添加多条边
G.add_edges_from([(1,2),(1,3)])
# 查看图信息
print('共有节点个数：', G.number_of_nodes())
print('查看节点：', G.nodes())
print('共有边个数：', G.number_of_edges())
print('查看那些边：', G.edges())
# 查看相邻节点
print(G.neighbors(2))
# 移除边
G.remove_edge(1, 3)
print('边个数：',G.number_of_edges())
print(G.edges())

# 节点和边可以是任意数据类型，通常为数字和字符
G.add_node('a')
G.add_node('bv')
G.add_edge('a', 1)

# 简单可视化
# plt.figure()
# nx.draw_networkx(G)
# plt.show()

# 2 其它方式创建图
# 2.1邻接列表
G2 = nx.read_adjlist('./data/G_adjlist.txt', nodetype=int)
print('list(G2.edges())\n',list(G2.edges()))

# 简单可视化
# plt.figure()
# nx.draw_networkx(G2)
# plt.show()

# 2.2邻节矩阵
G_mat = np.array([[0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                  [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                 ])
G3 = nx.Graph(G_mat)
print('list(G3.edges())\n',list(G3.edges()))

# 简单可视化
# plt.figure()
# nx.draw_networkx(G3)
# plt.show()

# 边列表
G4 = nx.read_edgelist('./data/G_edgelist.txt', data=[('weigth', int)])
print('list(G4.edges(data=True))\n',list(G4.edges(data=True)))

# # 简单可视化
# plt.figure()
# nx.draw_networkx(G4)
# plt.show()

# 2.4DateFrame
G_df = pd.read_csv('./data/G_edgelist.txt', delim_whitespace=True,
                   header =None, names= ['n1', 'n2', 'weight'])
print('G_df\n',G_df)

G5 = nx.from_pandas_dataframe(G_df,'n1','n2',edge_attr='weight')
print('list(G5.edges(data=True))\n',list(G5.edges(data=True)))
# 简单可视化
# plt.figure()
# nx.draw_networkx(G5)
# plt.show()

# 3 访问边
G6 =nx.Graph()
G6.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
plt.figure()
nx.draw_networkx(G6)
plt.show()

# 返回节点和邻居节点
for n, nbrs in G6.adjacency_iter():
    print(n, nbrs)

#遍历边
# 找出权重小于0.5的边
for n, nbrs in G6.adjacency_iter():
    for nbrs, edge_attr in nbrs.items():
        data = edge_attr['weight']
        if data < 0.5:
            print('{}, {}, {:.3f}'.format(n, nbrs, data))

# 4 为图添加属性
# 4.1 图属性
G7 = nx.Graph(day = 'Friday')
print('G7.graph\n',G7.graph)

G7.graph['day'] = 'Monday'
print('G7.graph["day"]\n',G7.graph)

# 4.2 节点属性
G7.add_node(1, time='5pm')
G7.add_nodes_from([3], time='3pm')
print('节点属性',G7.node)
G7.node[1]['room'] = 714
print('G7.nodes',G7.nodes())

# data=True表示连属性一起输出
G7.nodes(data=True)
print('G7.nodes(data=True)\n',G7.nodes(data=True))

# 4.3 边属性
G7.add_edge(1, 2, weight=4.7)
G7.add_edges_from([(3,4), (4,5)], color='red')
G7.add_edges_from([(1, 2, {'color' : 'blue'}), (2, 3, {'weight':8})])
print('边属性：\n',G7.edges(data=True))

# 5 网络类型

# 无向图
G8 = nx.Graph()
G8.add_edge('A','B')
G8.add_edge('B','C')
plt.figure()
nx.draw_networkx(G8)
plt.show()

# 有向图
G9 = nx.DiGraph()
G9.add_edge('A', 'B')
G9.add_edge('B', 'C')

plt.figure()
nx.draw_networkx(G9)
plt.show()

# 权重网络
G10 = nx.Graph()
G10.add_edge('A','B', weight=6)
G10.add_edge('B','C', weight=13)
G10.edges(data=True)
print('G10.edges(data=True)\n', G10.edges(data=True))
plt.figure()
nx.draw_networkx(G10)
plt.show()

# 网络符号
G11 = nx.Graph()
G11.add_edge('A', 'B', sign='+')
G11.add_edge('B', 'C', sign='-')
plt.figure()
nx.draw_networkx(G11)
plt.show()

G12 = nx.Graph()
G12.add_edge('A', 'B', relation='friend')
G12.add_edge('B', 'C', relation='coworker')
G12.add_edge('C', 'D', relation='family')
G12.add_edge('E', 'F', relation='neighbor')
print('网络符号:',G12.edges(data=True))
plt.figure()
nx.draw_networkx(G12)
plt.show()

# 多重图
G13 = nx.MultiGraph()
G13.add_edge('A', 'B', relation='friend')
G13.add_edge('A', 'B', relation='neighbor')
G13.add_edge('G', 'F', relation='family')
G13.add_edge('G', 'F', relation='coworker')
print('多重图:',G13.edges(data=True))
plt.figure()
nx.draw_networkx(G13)
plt.show()


# 6 网络的属性访问

# 6.1 边属性的访问
G14 = nx.Graph()
G14.add_edge('A','B',weight=6,relation='family')
G14.add_edge('B','C',weight=13,relation='friend')
# 列出所有的边
print('G14.edges()\n',G14.edges())
# 列出边，并且带属性
print('G14.edges()\n',G14.edges(data=True))
# 指定输出属性
print('指定输出属性:', G14.edges(data='relation'))
# 边（A B）的属性字典
print('边（A B）的属性字典:',G14.edge['A']['B'])
# 边（B,C）指定输出属性
print('边（B,C）指定输出属性:', G14.edge['B']['C']['weight'])
# 无向图中节点顺序无关
print('边（B,C）指定输出属性:', G14.edge['C']['B']['weight'])

# 有向权重网络
G14 = nx.DiGraph()
G14.add_edge('A', 'B', weight=6, relation='family')
G14.add_edge('C', 'B', weight=13, relation='friend')

print("G14.edge['C']['B']['weight']",G14.edge['C']['B']['weight'])
# 注意有向图的节点顺序
# print("G14.edge['B']['C']['weight']",G14.edge['B']['C']['weight'])

# 边属性的访问
G15 = nx.Graph()
G15.add_edge('A', 'B', weight=6, relation='family')
G15.add_edge('B', 'C', weight=13, relation='friend')
# 为节点添加属性
G15.add_node('A', role= 'trader')
G15.add_node('B', role= 'trader')
G15.add_node('C', role = 'manager')

# 列出所有节点
print('列出所有节点:',G15.nodes())
print('列出所有节点带属性:',G15.nodes(data=True))
print('访问单个节点带属性:',G15.node['A']['role'])
