'''
    作者：张鹏
    时间：01/25/18
    版本1.0
    功能：计算AQI
'''
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    '''
        主函数
    '''
    aqi_data = pd.read_csv('china_city_aqi.csv')

    #数据清洗
    #只保留AQI大于0的数据
    clearn_aqi_data = aqi_data[aqi_data['AQI'] > 0]

    #基本统计
    print('AQI最大值：{}'.format(clearn_aqi_data['AQI'].max()))
    print('AQI最小值：{}'.format(clearn_aqi_data['AQI'].min()))
    print('AQI均值：{}'.format(clearn_aqi_data['AQI'].mean()))

    #top10

    top_50 = clearn_aqi_data.sort_values(by=['AQI']).head(50)
    top_50.plot(kind='bar', x='city', y='AQI', title='空气质量最好排名前50',
                figsize=(20, 10))
    plt.savefig('top50_aqi.jpeg')
    plt.show()
    # print('空气质量最好排名前十的城市：')
    # print(top_10)
    # bottom_10 = aqi_data.sort_values(by=['AQI'], ascending=False).head(10)
    bottom_10 = clearn_aqi_data.sort_values(by=['AQI']).tail(50)
    bottom_10.plot(kind='bar', x='city', y='AQI', title='空气质量最差排名前50',
                figsize=(20, 10))
    plt.savefig('bottom50_aqi.jpeg')
    plt.show()
    # print('空气质量最差排名前十的城市：')
    # print(bottom_10)


if __name__ == '__main__':
    main()
