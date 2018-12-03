'''
    作者：张鹏
    时间：01/25/18
    版本1.0
    功能：计算AQI
'''
import pandas as pd


def main():
    '''
        主函数
    '''
    aqi_data = pd.read_csv('china_city_aqi.csv')
    #读取前十条数据
    # print(aqi_data.head(10))
    #读取后十条数据
    # print(aqi_data.tail(10))
    #查看某一列的数据
    # print(aqi_data['AQI'])
    # print(aqi_data)

    # print('基本信息')
    # print(aqi_data.info())

    # print('数据预览：')

    #基本统计
    print('AQI最大值：{}'.format(aqi_data['AQI'].max()))
    print('AQI最小值：{}'.format(aqi_data['AQI'].min()))
    print('AQI均值：{}'.format(aqi_data['AQI'].mean()))

    #top10

    top_10 = aqi_data.sort_values(by=['AQI']).head(10)
    print('空气质量最好排名前十的城市：')
    print(top_10)
    # bottom_10 = aqi_data.sort_values(by=['AQI'], ascending=False).head(10)
    bottom_10 = aqi_data.sort_values(by=['AQI']).tail(10)
    print('空气质量最差排名前十的城市：')
    print(bottom_10)

    #保存csv

    top_10.to_csv('top10_aqi.csv', index=False)
    bottom_10.to_csv('bottom10_aqi.csv', index=False)


if __name__ == '__main__':
    main()


