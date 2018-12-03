'''
    作者：张鹏
    时间：01/25/18
    版本1.0
    功能：计算AQI
'''
import json
import csv

def process_json_file(file_path):
    '''
        解析json文件
    '''
    f = open(file_path, mode='r', encoding='utf-8')
    city_list = json.load(f)
    # f.close()
    return city_list



def main():
    '''
        主函数
    '''
    file_path = input('请输入json文件名称:')
    #调用函数处理文件
    city_list = process_json_file(file_path)
    # print(city_list)
    city_list.sort(key=lambda city: city['aqi'])
    # top_list = city_list[:5]
    # print(city_list)

    #json写入文件
    # f = open('top_5.json', mode='w' ,encoding='utf-8')
    # json.dump(top_list, f, ensure_ascii=False)
    # f.close()

    lines = []
    #添加标题
    lines.append(city_list[0].keys())
    for city in city_list:
        lines.append(list(city.values()))
        # print(r)
    f = open('aqi.csv', 'w', encoding='utf-8',newline='')
    # f.write(codecs.BOM_UTF8)
    writer = csv.writer(f)
    for line in lines:
        writer.writerow(line)
    f.close()
if __name__ == '__main__':
    main()


