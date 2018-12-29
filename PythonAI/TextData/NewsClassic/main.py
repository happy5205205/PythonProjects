# _*_ coding: utf-8 _*_
"""
    时间：2018-12-25
    作者：张鹏
    文件名：main.py
    功能：主程序
"""
from PythonAI.TextData.NewsClassic import utils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    """
        主函数
    """
    # 准备数据
    all_data = utils.prepare_data()

    # 查看数据集
    top_categories = utils.get_top_categories(all_data=all_data)

    # 根据选取的类别对数据进行过滤
    used_data = all_data[all_data['category'].isin(top_categories)].copy()

    # 添加一列label用于模型输入
    label_enoder = LabelEncoder()
    used_data['label'] = label_enoder.fit_transform(used_data['category'].values)
    print('最终样本数量：', used_data.shape[0])
    train_data, test_data = train_test_split(used_data, test_size=1/4)

    # 特征工程处理
    X_train, X_test = utils.do_feature_engineering(train_data, test_data)
    print('共有{}维特征'.format(X_train.shape[1]))

    # 标签处理
    y_train = train_data['label'].values
    y_test = test_data['label'].values

    # 数据建模
    print('\n===================== 数据建模及验证 =====================')
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)

    # print('准确率为', accuracy_score(y_test, y_pred))

    cat_names = label_enoder.classes_
    cat_labels = label_enoder.transform(cat_names)
    print('标签：', cat_labels)
    print('标签名称：', cat_names)
    print('混淆矩阵')
    # print(confusion_matrix(y_test, y_pred, labels=cat_labels))


if __name__ == '__main__':
    main()