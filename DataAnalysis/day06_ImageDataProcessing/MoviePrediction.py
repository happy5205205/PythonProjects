"""
    时间：2018年4月20日
    作者：张鹏
    版本：1.0
    名称：根据海报预测电影分类
    项目描述：
        电影海报是获取电影内容和类型的途径之一。用户可以通过海报的一些信息（如：颜色，演员的表情等）
        推测出其所属的类型（恐怖片，喜剧，动画等）。研究表明图像的颜色是影响人类感觉（心情）的因素之一，
        在本次项目中，我们会通过海报的颜色信息构建模型，并对其所属的电影类型进行推测。

    项目任务：
        1、数据查看几及处理
        2、特征表示
        3、数据建模及验证
        4、数据预测
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# 解决matplotlib中中文显示乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据查看及处理
# 指定数据集路径
data_path = './data'
csv_datafile = os.path.join(data_path, 'MovieGenre.csv')
poster_path = os.path.join(data_path, 'SampleMoviePosters')

# 加载数据
movie_df = pd.read_csv(csv_datafile, encoding='ISO-8859-1',
                       usecols=['imdbId', 'Title', 'IMDB Score', 'Genre'])
print(movie_df.head(2))

# 处理genre列，是其中包含一中类型
movie_df['single_gener'] = movie_df['Genre'].str.split('|', expand=True)[0]
print('数据文件csv共有{}条。'.format(len(movie_df)))
print(movie_df.head(2))

# 将海报文件与csv文件进行合并

# 构造海报DataFrame
poster_df =pd.DataFrame(columns=['imdbId', 'img_path'])
# print(poster_df)
poster_df['img_path']= os.listdir(poster_path)
poster_df['imdbId'] = poster_df['img_path'].str[: -4].astype('int')
print(poster_df.head(2))

# 将csv文件与海报数据进行合并
data_df = movie_df.merge(poster_df, on='imdbId', how='inner')
print(data_df.head(2))
# print(len(data_df))
data_df.drop_duplicates(subset=['imdbId'], inplace=True)
# print(len(data_df))

# 查看各类电影的数量
movie_gener_num =data_df.groupby('single_gener').size().sort_values(ascending=False)
print('每个类型数量为：\n',movie_gener_num.head(2))

# 可视化个类别的
plt.figure()
sns.countplot(x='single_gener', data=data_df)
plt.title('电影类型数量统计')
plt.xticks(rotation='vertical')
plt.xlabel('电影类型')
plt.ylabel('数量')
plt.tight_layout()
# plt.show()

# 有些电影类型过于少，不利于预测。将上述问题抓换为3分类问题：Drama, Comedy, Other
cond = (data_df['single_gener'] != 'Drama') & (data_df['single_gener'] != 'Comedy')
# print(cond)
data_df.loc[cond, 'single_gener'] = 'Other'
movie_single_gener_num = data_df.groupby('single_gener').size().sort_values(ascending=False)
print('将电影类型分为三类，每类数量为：\n', movie_single_gener_num)


# 特征表示
from skimage import io, exposure
# 提取每个图片的直方图作为颜色特征
def extract_hist_feat(img_path, nbins=50, as_grey=True):
    '''
        提取每个图片的直方图作为颜色特征
        img_path：图片路径
        nbins：直方图bin的个数，即特征的维度
    '''
    image_data = io.imread(img_path, as_grey=as_grey)
    eq_image_data = exposure.equalize_hist(image_data)
    if as_grey:
        # 灰度图片
        # 提取直方图特征
        hist_feat, _ = exposure.histogram(eq_image_data, nbins=nbins)
    else:
        pass
    norm_hist_feat =hist_feat / sum(hist_feat)

    return norm_hist_feat

# 测试一张图片
# img_path = os.path.join(poster_path, data_df.iloc[1]['img_path'])
# print('img_path',img_path)
# hist_feat = extract_hist_feat(img_path)
# print(hist_feat)
# print('type(hist_feat)\n', type(hist_feat))
# print('hist_feat.shape\n', hist_feat.shape)
# print(data_df.index)

# 对数据集中的每张图片进行特征提取
n_feat_dim = 100
n_sample = len(data_df)

# # 初始化特征矩阵
X = np.zeros((n_sample, n_feat_dim))
# print('type(X)\n', type(X))
# print('11111111\n',X)
# print(X.shape)
for i, r_data in data_df.iterrows():
    if (i+1) % 100 ==0:
        print('正在提取，已完成{}个海报提取'.format(i+1))
    img_path =os.path.join(poster_path, r_data['img_path'])
    hist_feat = extract_hist_feat(img_path, n_feat_dim)
    X[i :] = hist_feat.copy()
# print(X)
# 获得标签名称
target_names = data_df['single_gener'].values
from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()
y = lab_enc.fit_transform(target_names)
# print(y)

print('电影类型：',lab_enc.classes_)
# print('y:', y)
for i in range(len(lab_enc.classes_)):
    print('{}对应标签为{}'.format(lab_enc.classes_[i], i))

# 分割数据集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/4, random_state=0)
print('训练样本：{}条。\n测试样本：{}条'.format(len(X_train), len(X_test)))

# 模型训练
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV



def model_train(X_train, X_test, y_train, y_test, config):
    model_name = config[0]
    paramaent = config[1]
    if paramaent is not None:
        clf = GridSearchCV(model_name, paramaent, cv= 3, scoring='accuracy')
        clf.fit(X_train, y_train)
        print('最优参数：{},验证机准确率最高为：{}'.format(clf.best_params_, clf.best_score_))
    else:
        model_name.fit(X_train, y_train)
        clf = model_name

    test_acc = clf.score(X_test, y_test)
    print('测试集准确率为：{}', test_acc)
    return test_acc, clf

model_train_dict = {'KNN':  (KNeighborsClassifier(),      {'n_neighbors': [5, 10, 15]}),
                    'LR':   (LogisticRegression(),        {'C': [0.1, 1, 10]}),
                    'SVM':  (SVC(),                       {'C': [100, 1000, 10000]}),
                    'DT':   (DecisionTreeClassifier(),    {'max_depth': [100, 150,200]}),
                    'GNB':  (GaussianNB(),None),
                    'RF':   (RandomForestClassifier(),    {'n_estimators': [10, 20, 30]}),
                    'GBDT': (GradientBoostingClassifier(),{'learning_rate': [0.01, 0.1, 0.5, 1]})
                    }
results_df = pd.DataFrame(columns=['Accuracy'], index=list(model_train_dict.keys()))
results_df.index.name = 'model'
models = []
model_num = 0
for model_name, config in model_train_dict.items():
    model_num += 1
    print('************************正在训练{}模型，第{}个/共{}个*******************************'
          .format(model_name, model_num, len(model_train_dict.keys())))
    acc, model = model_train(X_train, X_test, y_train, y_test, config)
    models.append(model)
    results_df.loc[model_name] = acc

# 保存结果
results_df.to_csv('./result.csv')
# 将结果可视化
results_df.plot(kind= 'bar')
plt.ylabel('accuracy')
plt.tight_layout()
plt.savefig('./results.jpg')
plt.show()

# 保存最优模型
import pickle
best_model_index =results_df.reset_index()['Accuracy'].argmax()
best_model = models.append(best_model_index)
save_model_path ='./predict_pjupyter.pkl'
with open(save_model_path, 'wb') as f:
    pickle.dump(best_model, f)

# 预测数据
# 加载保存的模型
with open(save_model_path, 'rb') as f:
    predictor = pickle.load(f)

# 进行预测

imdb_id = 2544
img_path = os.path.join(poster_path, str(imdb_id) + '.jpg')
poster_feat = extract_hist_feat(img_path, n_feat_dim)
pred_result = predictor.predict(poster_feat.reshape(1, -1))
pred_genre = lab_enc.inverse_transform(pred_result)
print('预测类型：{}', pred_genre)
print('真是类型：{}', data_df[data_df['imdbId'] == imdb_id]['single_gener'].values)

plt.figure()
plt.grid(False)
plt.imshow(io.imread(img_path))
plt.show()


