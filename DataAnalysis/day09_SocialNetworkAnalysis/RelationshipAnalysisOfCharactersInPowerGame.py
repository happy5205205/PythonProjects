#  -*- coding: utf-8 -*-
"""
    作者：张鹏
    版本：v1.0
    时间：2018年6月21日
    实战案例：“权力游戏”人物关系分析
"""
# 引入必要的包
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

#  解决matplotlib中文乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 指定数据集
dataset_path = './data'

def get_top_records(series_list, top_n=10, show_figure=False):
    """
    取出每本书最重要的top_n个任务
         top_n=最重要的top_n
        show_figure:是否显示图片
      """
    for i, seriers in enumerate(series_list):
        print('第{}几本书重要的{}个人物：'.format(i + 1, top_n))
        top_characters = seriers.sort_values(ascending=False)[:top_n]
        print(top_characters)

        if show_figure:
            plt.figure(figsize=(10, 9))
            top_characters.plot(kind='bar',title='第{}本书'.format(i + 1))
            plt.tight_layout()
            plt.show()
        print()


def main():
    """
        主函数
       """
    # 任务1：查看数据
    print('\n===================== 任务1. 数据查看 =====================')
    # 加载数据
    book1_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book1-edges.csv'))
    book2_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book2-edges.csv'))
    book3_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book3-edges.csv'))
    book4_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book4-edges.csv'))
    book5_df = pd.read_csv(os.path.join(dataset_path, 'asoiaf-book5-edges.csv'))

    print(book1_df.head())
    # 任务2：构建网络
    print('\n===================== 任务2. 构建网络 =====================')
    # dataFrame构建网络
    G_book1 = nx.from_pandas_dataframe(book1_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book2 = nx.from_pandas_dataframe(book2_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book3 = nx.from_pandas_dataframe(book3_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book4 = nx.from_pandas_dataframe(book4_df, 'Source', 'Target', edge_attr=['weight', 'book'])
    G_book5 = nx.from_pandas_dataframe(book5_df, 'Source', 'Target', edge_attr=['weight', 'book'])

    G_books = [G_book1, G_book2, G_book3, G_book4, G_book5]

    # 查看网络的边
    print('第一个图的边：')
    print(G_book1.edges(data=True))
    # 简单可视化
    plt.figure(figsize=(10,9))
    nx.draw_networkx(G_book1)
    plt.show()

    # 任务3. 网络分析
    print('\n===================== 任务3. 网络分析 =====================')
    print('Degree Centrality')
    # 计算每个网络的degree centrality
    #并将计算结果构建成Series
    clo_cen_list = [nx.closeness_centrality(G_book) for G_book in G_books]
    clo_cen_series_list = [pd.Series(clo_cen) for clo_cen in clo_cen_list]
    get_top_records(clo_cen_series_list, show_figure=True)

    print('Betweenness Centralisty')
    # 计算每个网络的Betweenness Centralisty
    bet_cen_list = [nx.betweenness_centrality(G_book) for G_book in G_books]
    bet_cen_series_list = [pd.Series(bet_cen) for bet_cen in bet_cen_list]
    get_top_records(bet_cen_series_list,show_figure=True)

    print('Page Rank')
    # 计算每个网络的Page Rank
    # 并将计算结果构建成Series
    page_rank_list = [nx.pagerank(G_book) for G_book in G_books]
    page_rank_series = [pd.Series(page_rank) for page_rank in page_rank_list]
    get_top_records(page_rank_series, show_figure=True)

    print('\n===================== 任务3.2 各指标的相关性 =====================')
    cor_df = pd.DataFrame(columns=['Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality', 'Page Rank'])
    cor_df['Degree Centrality'] = pd.Series(nx.degree_centrality(G_book1))
    cor_df['Closeness Centrality'] = pd.Series(nx.closeness_centrality(G_book1))
    cor_df['Betweenness Centrality'] = pd.Series(nx.betweenness_centrality(G_book1))
    cor_df['Page Rank'] = pd.Series(nx.pagerank(G_book1))
    print(cor_df.corr())

    print('\n===================== 任务3.3 人物重要性的趋势 =====================')
    trend_df = pd.DataFrame(columns=['Book1', 'Book2', 'Book3', 'Book4', 'Book5'])
    trend_df['Book1'] = pd.Series(nx.degree_centrality(G_book1))
    trend_df['Book2'] = pd.Series(nx.degree_centrality(G_book2))
    trend_df['Book3'] = pd.Series(nx.degree_centrality(G_book3))
    trend_df['Book4'] = pd.Series(nx.degree_centrality(G_book4))
    trend_df['Book5'] = pd.Series(nx.degree_centrality(G_book5))
    trend_df.fillna(0,inplace=True)

    # 第一本书中最重要top10任务趋势
    top_10_from_book1 = trend_df.sort_values('Book1', ascending=False)[:10]
    top_10_from_book1.T.plot(figsize=(10,9))
    plt.tight_layout()
    plt.savefig('./role_trend.jpg')
    plt.show()
    print('\n===================== 任务3.4 网络可视化 =====================')
    plt.figure(figsize=(10,9))
    # 节点的颜色有节点的度决定
    node_color = [G_book1.degree(v) for v in G_book1]
    # 节点的大小由degree centrality决定
    node_size = [10000 * nx.degree_centrality(G_book1)[v] for v in G_book1]
    # 边的宽度有权重决定
    edge_width = [0.2 * G_book1[u][v]['weight']for u, v in G_book1.edges()]

    # 使用spring布局
    pos = nx.spring_layout(G_book1)
    nx.draw_networkx(G_book1, pos, node_size=node_size,node_color=node_color,
                     alpha = 0.7, with_labels=False,width =edge_width)

    # 取出第一本书的top10人物
    top10_in_book1 = top_10_from_book1.index.values.tolist()
    # 构建label
    lables = {role:role for role in top10_in_book1}

    nx.draw_networkx_labels(G_book1, pos, labels=lables, font_size=10)

    plt.figure(figsize=(10,9))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./book1_network.jpg')
    plt.show()

if __name__ == '__main__':
    main()