'''
    作者：张鹏
    时间：01/25/18
    版本1.0
    功能：计算AQI
'''
import requests


def get_html_text(url):
    '''
        返回url的文本
    '''
    r = requests.get(url)
    # print(r.status_code)
    return r.text

def main():
    '''
        主函数
    '''
    city = input('请输入城市:')
    url = 'http://pm25.in/' + city
    url_text = get_html_text(url)
    # print(url_text)
    aqi_div = '''<div class="span12 data">
        <div class="span1">
          <div class="value">
            '''
    index = url_text.find(aqi_div)
    begin_index = index + len(aqi_div)
    end_index = begin_index + 2
    aqi_val = url_text[begin_index: end_index]
    print('{}空气质量为：{}'.format(city,aqi_val))
if __name__ == '__main__':
    main()


