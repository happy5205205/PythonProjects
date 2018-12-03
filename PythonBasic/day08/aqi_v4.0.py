'''
    作者：张鹏
    时间：01/25/18
    版本1.0
    功能：计算AQI
'''
import json
import csv
import os

def process_json_file(file_path):
    '''
        解析json文件
    '''
    # f = open(file_path, mode='r', encoding='utf-8')
    # city_list = json.load(f)
    # # f.close()
    # return city_list
    with open(file_path, mode='r', encoding='utf-8') as f:
        city_list = json.load(f)
        print(city_list)


def process_csv_file(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        read = csv.reader(f)
        for i in read:
            #两种输出格式对比
            print(','.join(i))
            print('-----------------------------------------------')
            print(i)

def main():
    '''
        主函数
    '''
    file_path = input('请输入json文件名称:')
    file_name, file_ext = os.path.splitext(file_path)
    print(file_name,'----',file_ext)
    if file_ext == '.json':
        process_json_file(file_path)
    elif file_ext == '.csv':
        process_csv_file(file_path)
    else:
        print('不支持这种格式的文件1')
if __name__ == '__main__':
    main()


