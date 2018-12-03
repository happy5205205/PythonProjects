# -*- coding:utf-8 -*-
critics = {
    'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5,
                  'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
    'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0,
                     'The Night Listener': 3.0, 'You, Me and Dupree': 3.5},
    'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5,
                         'The Night Listener': 4.0},
    'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0,
                     'You, Me and Dupree': 2.5},
    'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0,
                     'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
    'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0,
                      'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
    'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0},
    'Yu': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5,
           'You, Me and Dupree': 2.5, 'The Night Listener': 3.0}
    }

from math import sqrt

print
u"距离与相关系数：它们之间是相反的，若距离越短（距离的数值越小），则相似度越大（相似度的数值越大）"


# 欧几里得距离
def sim_distance(prefs, person1, person2):
    # 得到两者同时评价过的电影的列表
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    # 若不存在同时评价过的电影则返回0
    if len(si) == 0:   return 0
    # 计算所有差值的平方和
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2)
                          for item in prefs[person1] if item in prefs[person2]])
    # sum()函数中的参数是一个list，sum([item for item in a if item in b])


 # return 1 / (1 + sqrt(sum_of_squares))
print
u"欧几里得距离（最后给出的数值，实际上是给出了相似度评价）："
print(sim_distance(critics, 'Lisa Rose', 'Gene Seymour'))


# 皮尔逊相关系数
def sim_pearson(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1
    n = len(si)
    if n == 0:
        return 1  # 如果两者不存在同时评论过的电影时，返回1

    # 对所有偏好求和
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    # 求平方和
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
    # 求乘积之和
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # 计算皮尔逊评价值
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))

    if den == 0: return 0
    r = num / den

    return r


print
u"皮尔逊相关系数："
print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))


# Jaccard相似度（狭义）——只能用于判断两者之间是否一致，而不能根据其评分来判定相似度
def sim_jaccard(prefs, per1, per2):
    si_union = {}  # 并集
    si_inter = {}  # 交集
    si_union = dict(prefs[per1], **prefs[per2])

    for item in prefs[per1]:
        if item in prefs[per2]:
            si_inter[item] = min(prefs[per1][item], prefs[per2][item])

    sum1 = len(si_inter)
    sum2 = len(si_union)

    if (sum2 == 0): return 0

    r = float(sum1) / sum2
    return r


print
u"Jaccard相似度（狭义）——只能用于判断两者之间是否一致，而不能根据其评分来判定相似度："
print
sim_jaccard(critics, 'Lisa Rose', 'Gene Seymour')


# 曼哈顿距离（城市街区距离  ）
def sim_manhattan(prefs, p1, p2):
    si = {}
    for item in p1:
        if item in p2: si[item] = 1
    if len(item) == 0: return 1

    sum_of_minus = sum([abs(prefs[p1][item] - prefs[p2][item])
                        for item in prefs[p1] if item in prefs[p2]])
    return 1 / (sum_of_minus + 1)


print
u"曼哈顿距离（最后得到的数值也是相似度）："
print
sim_manhattan(critics, 'Lisa Rose', 'Gene Seymour')