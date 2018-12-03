"""
    时间：2018年6月19日
    内容：网络可视化
"""
# 网络可视化
import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gpickle('./data/major_us_cities')
# print(G.nodes(data=True))
# print(G.edges(data=True))

plt.figure(figsize=(10, 9))
nx.draw_networkx(G)
# plt.show()

# 网络布局

[x for x in nx.__dir__() if x.endswith('_layout')]
plt.figure(figsize=(10,9))
pos = nx.random_layout(G)
nx.draw_networkx(G, pos)


plt.figure(figsize=(10,9))
pos = nx.circular_layout(G)
nx.draw_networkx(G, pos)


# 自定义布局
plt.figure(figsize=(10,9))
pos = nx.get_node_attributes(G,'location')
nx.draw_networkx(G, pos)


plt.figure(figsize=(10,9))
nx.draw_networkx(G, pos, alpha=0.7, with_labels=False, edge_color='0.4')
plt.axis('off')
plt.tight_layout()
# plt.show()

plt.figure(figsize=(10,9))
node_color = [G.degree(v) for v in G]
node_size = [0.0005 * nx.get_node_attributes(G, 'population')[v] for v in G]
edge_width = [0.005 * G[u][v]['weight'] for u,v in G.edges()]
nx.draw_networkx(G,pos,node_size=node_size,
                 node_color=node_color,alpha=0.7,with_labels=False,
                 width=edge_width,edge_color='0.4',cmap=plt.cm.Blues)
plt.axis('off')
plt.tight_layout()


plt.figure(figsize=(10,9))
node_color = [G.degree(v) for v in G]
node_size = [0.0005 * nx.get_node_attributes(G, 'population')[v] for v in G]
edge_width = [0.005 * G[u][v]['weight'] for u,v in G.edges()]
nx.draw_networkx(G,pos,node_size=node_size,
                 node_color=node_color,alpha=0.7,with_labels=False,
                 width=edge_width,edge_color='0.4',cmap=plt.cm.Blues)
greater_than_770 = [x for x in G.edges(data=True) if x[2]['weight']>770]
nx.draw_networkx_edges(G,pos,edgelist=greater_than_770,edge_color='r',alpha=0.4,width=6)
nx.draw_networkx_labels(G,pos,labels={'Los Angeles, CA':'LA','WeChat York, NY':'NYC'},
                        font_size=18,font_color='w')
plt.axis('off')
plt.tight_layout()
plt.show()

