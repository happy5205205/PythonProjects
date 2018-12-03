# _*_coding: utf-8 _*_
"""
    时间：2018年11月23日 15：18
    作者：张鹏
    文件命： main.py
    功能：主程序
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PythonAI_Ⅱ.PhonePrediction import config
from PythonAI_Ⅱ.PhonePrediction import utils


def main():
    """
        主程序
    """
    # 加载数据
    dataset = pd.read_csv(os.path.join(config.dataset_path,'german_credit_data.csv'))

    # 数据清洗
    cln_data = utils.clearn_data(dataset)

    # 数据分割
    train_data, test_data = train_test_split(cln_data, test_size=1/4, random_state=10)

    # 数据预览及可视化
    utils.inspect_date(train_data=train_data, test_data=test_data)

    # 特征处理
    print('\n===================== 特征工程 =====================')
    X_trian, y_trian = utils.transform_data(train_data)
    X_test, y_test = utils.transform_data(test_data)

    # 构建训练测试数据
    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')
    model_name_param_dict = {'KNN': [5, 8, 10],
                             'LR': [0.01, 1, 10],
                             'DT': [5, 10, 15],
                             'SVM': [0.01, 1, 10]
                             }
    results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                             index=list(model_name_param_dict.keys()))
    for model_name, param_range  in model_name_param_dict.items():
        _, best_acc, mean_duration = utils.train_test_model(X_trian=X_trian,y_train=y_trian,X_test=X_test,
                                                            y_test=y_test,param_range=param_range,
                                                            model_name=model_name)
        results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
        results_df.loc[model_name, 'Time (s)'] = mean_duration

    results_df.to_csv(os.path.join(config.output_path,'result.csv'))

    # 对结果经行可视化
    print('\n===================== 模型及结果比较 =====================')
    plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    results_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[60, 100],ax=ax1, title='Accuracy(%)', legend=False)
    ax2 = plt.subplot(1, 2, 2)
    results_df.plot(y=['Time (s)'], kind='bar',ax=ax2, title='Time(s)', legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path,'compare_result.jpg'))
    plt.show()
if __name__ == '__main__':
    main()



