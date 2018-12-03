d = {'小象学院': 'http://www.chinahadoop.cn/',
    '百度': 'https://www.baidu.com/',
    '阿里巴巴': 'https://www.alibaba.com/',
    '腾讯': 'https://www.tencent.com/'}
print(d)

for key in d.keys():
    print(key)

for value in d.values():
    print(value)

for key, value in d.items():
    print(key, ':', value)
