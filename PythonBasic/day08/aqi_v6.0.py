'''
    作者：张鹏
    时间：01/25/18
    版本1.0
    功能：计算AQI
'''
import requests
from bs4 import BeautifulSoup


def get_city_aqi(city):
    '''
        返回url的文本
    '''
    url = 'http://pm25.in/' + city
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.text, 'lxml')
    div_list = soup.find_all('div', {'class' : 'span1'})
    city_aqi = []
    for i in range(8):
        div_content = div_list[i]
        caption = div_content.find('div', {'class' : 'caption'}).text.strip()
        value = div_content.find('div', {'class' : 'value'}).text.strip()
        city_aqi.append((caption,value))
    return city_aqi
def main():
    '''
        主函数
    '''
    city = input('请输入城市:')

    aqi_val = get_city_aqi(city)
    # print(url_text)

    print('{}空气质量为：{}\n'.format(city,aqi_val))
if __name__ == '__main__':
    main()


