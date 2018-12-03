"""
    作者：张鹏
    时间：2018年11月21日
    版本：v1.0
    功能：主程序
    实战案例3-1：贷款违约预测 (1)
    任务：使用scikit-learn建立不同的机器学习模型进行贷款违约预测
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import PythonAI_Ⅱ.LoanDefaultForecast.config as config

def main():
    """
        主程序
    """
    # 加载数据
    raw_date = pd.read_csv(os.path(config.dataset_path, 'german_credit_data.csv'))
if __name__ == '__main__':
    main()