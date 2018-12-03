from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# 3.3 中文中的TF-IDF
ch_text1 = ' 非常失望，剧本完全敷衍了事，主线剧情没突破大家可以理解，可所有的人物都缺乏动机，' \
           '正邪之间、妇联内部都没什么火花。团结-分裂-团结的三段式虽然老套但其实也可以利用积' \
           '攒下来的形象魅力搞出意思，但剧本写得非常肤浅、平面。场面上调度混乱呆板，满屏的铁甲审美疲劳。' \
           '只有笑点算得上差强人意。'
ch_text2 = ' 2015年度最失望作品。以为面面俱到，实则画蛇添足；以为主题深刻，实则老调重弹；以为推陈出新，' \
           '实则俗不可耐；以为场面很high，实则high劲不足。气！上一集的趣味全无，这集的笑点明显刻意到心虚。' \
           '全片没有任何片段给我有紧张激动的时候，太弱了，跟奥创一样。'
ch_text3 = ' 《铁人2》中勾引钢铁侠，《妇联1》中勾引鹰眼，《美队2》中勾引美国队长，' \
           '在《妇联2》中终于……跟绿巨人表白了，黑寡妇用实际行动告诉了我们什么叫忠贞不二；' \
           '而且为了治疗不孕不育连作战武器都变成了两支验孕棒(坚决相信快银没有死，后面还得回来)'
ch_text4 = ' 虽然从头打到尾，但是真的很无聊啊。'
ch_text5 = ' 剧情不如第一集好玩了，全靠密集笑点在提神。僧多粥少的直接后果就是每部寡姐都要换着队友谈恋爱，' \
           '这特么比打斗还辛苦啊，真心求放过～～～（结尾彩蛋还以为是洛基呢，结果我呸！）'

ch_texts = [ch_text1, ch_text2, ch_text3, ch_text4, ch_text5]
# corpus = []
# for ch_text in ch_texts:
#     corpus.append(['/'.join(jieba.cut(ch_text, cut_all=False))])
# 或者
corpus = ['/'.join(jieba.cut(ch_text, cut_all=False)) for ch_text in ch_texts]

# print(corpus)

ch_vectorizer = TfidfVectorizer()
ch_feats = ch_vectorizer.fit_transform(corpus)
print(len((ch_vectorizer.get_feature_names())))
print(ch_feats.toarray()[0, :])