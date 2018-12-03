import pandas as pd
# 1、创建DataFrame
country1 = pd.Series({'Name' : '中国',
                      'Language': 'chinese',
                      'Area': '9.975M km2',
                      'Happiness Rank': 79})

country2 = pd.Series({'Name' : '美国',
                      'Language': 'English (US)',
                      'Area': '9.837M km2',
                      'Happiness Rank': 14})

country3 = pd.Series({'Name' : '澳大利亚',
                      'Language': 'English (AU)',
                      'Area': '7.692M km2',
                      'Happiness Rank': 9})

df = pd.DataFrame([country1, country2, country3], index=['CH', 'US', 'UA'])
# print(df)
# print(type(df))

# 2、Dataframe索引

#行索引
# print('loc:')
# print(df.loc['CH'])
# print(type(df.loc['CH']))

# print(df.iloc[1])

#列索引
# print(df['Area'])
# print(type(df['Area']))

# 获取不连续的列数据
# print(df[['Name', 'Area']]) #注意两个中括号

# 获取不连续的行数据
# print(df.iloc[[1, 2]])

# 混合索引
# 注意写法上的区别
# print('先取出行，再取出列：')
# print(df.iloc[1][2])
# print(df.iloc[0]['Area'])
# print(df.loc['UA']['Area'])

# print('先取出列，再取出行：')
# print(df['Area'][0])
# print(df['Area'].iloc[0])
# print(df['Area'].loc['UA'])

# 行列转换
# print(df.T)


# 3、删除数据
# print(df.drop(['UA']))
# 注意drop操作只是将修改后的数据copy一份，而不会对原始数据进行修改
# print(df)

# 如果使用了inplace=True，会在原始数据上进行修改，同时不会返回一个copy
# print(df.drop(['UA'], inplace= True)) #默认是False

#  如果需要删除列，需要指定axis=1
# print(df.drop(['Area'], axis=1))
# print(df)#原始数据还没有删除

# 也可直接使用del关键字
# del df['Name']
# print(df)

# 4. DataFrame的操作与加载
# 注意从DataFrame中取出的数据进行操作后，会对原始数据产生影响，这种操作不够安全
# ranks = df['Happiness Rank']
# print('最初的：')
# print(ranks)
# print('加2之后的：')
# ranks += 2
# print(ranks)
# print(df)

# 注意从DataFrame中取出的数据进行操作后，会对原始数据产生影响
# 安全的操作是使用copy()
# ranks = df['Happiness Rank'].copy()
# ranks += 2
# print(ranks)
# print(df)

# 加载csv文件数据
reprot_2015_df = pd.read_csv('../data/2015.csv')
# # print(reprot_2015_df.head())
# print(reprot_2015_df.info())

# 使用index_col指定索引列
# 使用usecols指定需要读取的列
reprot_2016_df = pd.read_csv('../data/2016.csv',
                             index_col = 'Country',
                             usecols = ['Country', 'Happiness Rank', 'Happiness Score', 'Region']
                             )
# 数据预览
# print(reprot_2016_df.head())
# print('列名(column)：', reprot_2016_df.columns)
# print('行名(index)：', reprot_2016_df.index)

# 注意index是不可变的
# reprot_2016_df.index[0] = '丹麦'#会报错

# 重置index
# 注意inplace加与不加的区别
# print(reprot_2016_df.reset_index().head())#不加对原始数据不造成影响
#加了更改原始数据,分开写
reprot_2016_df.reset_index(inplace=True)
# print(reprot_2016_df.head())

# 重命名列名
# reprot_2016_df.rename(columns={'Region': '地区', 'Hapiness Rank': '排名', 'Hapiness Score': '幸福指数'})
# print(reprot_2016_df.head())

# 重命名列名，注意inplace的使用
reprot_2016_df.rename(columns={'Country' : '国家','Region': '地区', 'Happiness Rank': '排名', 'Happiness Score': '幸福指数'},
                     inplace=True)
# print(reprot_2016_df.head())

# 5. Boolean Mask

# 过滤 Western Europe 地区的国家
# only_western_europe = reprot_2016_df['地区'] == 'Western Europe'
# print(reprot_2016_df['地区'] == 'Western Europe')
# print(reprot_2016_df[only_western_europe])

# 过滤 Western Europe 地区的国家
# 并且排名在10之外
only_western_europe_10 = (reprot_2016_df['地区'] == 'Western Europe') & (reprot_2016_df['排名'] > 10)
# print(only_western_europe_10)

# 叠加 boolean mask 得到最终结果
# print(reprot_2016_df[only_western_europe_10])

# 6. 层级索引
# print(reprot_2015_df.head())
# 设置层级索引
report_2015_df2 = reprot_2015_df.set_index(['Region', 'Country'])
# print(report_2015_df2.head(20))

# level0 索引
# print(report_2015_df2.loc['Western Europe'])

# 两层索引
# print(report_2015_df2.loc['Western Europe', 'Switzerland'])

# 交换分层顺序
# print(report_2015_df2.swaplevel())

# 排序分层
# print(report_2015_df2.sort_index(level=0))

# 7. 数据清洗
log_data = pd.read_csv('log.csv')
# print(log_data)

log_data.set_index(['time', 'user'], inplace =True)
log_data.sort_index(inplace = True)
# print(log_data)

#空数据用0填充
# print(log_data.fillna(0))
# #删除空数据
# print(log_data.dropna())
# # 按之前的数据进行填充
# print(log_data.ffill())
# # 按之后的数据进行填充
print(log_data.bfill())