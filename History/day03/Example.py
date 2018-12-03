from lxml import etree
from pprint import pprint
from urllib import request, error
import csv
url = 'http://www.6qt.net/index.asp?Field=Country&keyword=%B9%D8%BC%FC%D7%D6'
headers = {
    "User-Agent": 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.86 Safari/537.36',
}

def get_html(url):
    '''
    用urllib获取网页
    url 参数
    '''
    html = None
    req = request.Request(url, headers=headers)
    try:
        response = request.urlopen(req)
        html = response.read().decode('gbk')
    except error.URLError as e:
        print(e, url)
    except error.HTTPError as e:
        print(e, url)
    except Exception as e:
        print(e, url)
    return html


def crawl_data(html):
    '''
    从得到的网页中抓取数据
    '''
    # //div[@id="resultList"]/div[@class="el"]
    html = etree.ElementTree(etree.HTML(html))
    div_el = html.xpath('//div[@id="resultList"]/div[@class="el"]')
    #print(len(div_el))
    for i, row in enumerate(div_el):
        # div/p/span/a/@title
        row = etree.ElementTree(row)
        jobname = row.xpath('/div/p/span/a/@title')[0]
        # /div/span[1]/a
        company = row.xpath('/div/span[@class="t2"]/a/@title')[0]

        print(company)




url = 'http://www.6qt.net/index.asp?Field=Country&keyword=%B9%D8%BC%FC%D7%D6'
html = get_html(url)
crawl_data(html)




