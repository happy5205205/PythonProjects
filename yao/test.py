# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:48:41 2018

@author: dengyao
"""
import json
import random
from hashlib import md5
from http import client
from urllib.parse import urlencode

#APPID = "20171018000089127"
#SecretKey = "oxHQ8IodFp45h6ZDrnBH"
#APPID = "20180904000202626"
#SecretKey = "obzUYsfzk234D5whU_gV"
#
#APPID = "20180905000203154"
#SecretKey = "DIfSd1pgBF2zPIJ2Q0B1"
APPID ="20180911000205341"
SecretKey ="SstbiSxQBpKMiUeB8BPB"

#APP ID：20180905000203154
#密钥：DIfSd1pgBF2zPIJ2Q0B1

APIURL = "/api/trans/vip/translate?"
BaseURL = "api.fanyi.baidu.com"


class TranslateService(object):
    def __init__(self, query, from_lang="en", to_lang="zh"):
        self.params = {}
        salt = random.randint(32768, 65536)
        sign = APPID + query + str(salt) + SecretKey
        m = md5()
        m.update(sign.encode('utf-8'))
        sign = m.hexdigest()
        self.init_param(query, salt, sign, from_lang, to_lang)

    def init_param(self, query, salt, sign, from_lang="en", to_lang="zh"):
        self.params["appid"] = APPID
        self.params["q"] = query
        self.params["from"] = from_lang
        self.params["to"] = to_lang
        self.params["salt"] = salt
        self.params["sign"] = sign

    def get_result(self):
        try:
            param = urlencode(self.params)
            httpClient = client.HTTPConnection(BaseURL)
            myurl = APIURL + param
            httpClient.request('GET', myurl)

            response = httpClient.getresponse()
            ret = response.read()
            return TranslateService.process_data(ret)
        except Exception as e:
            return None

    @staticmethod
    def process_data(source):
        source = str(source, encoding="utf-8")
        target = json.loads(source)
        result = target.get("trans_result", None)
        if result:
            return result[0]["dst"]
        return None


#if __name__ == "__main__":
#    query = "hello world"
#    tr = TranslateService(query)
#    print(tr.get_result())


import pickle
import time

#  修改C:/Users/zhangpeng/Desktop/data.txt 为你的存放data的存放路径

ls=pickle.load(open("D:/PythonProjects/yao/data/inherited_disease.txt","rb"))

if __name__ == "__main__":
      
    for i in ls:
        n=len(i[8])
        if n>0:
            for j in range(n):
                query=i[8][j]
                tr=TranslateService(query)
                i[8][j]=tr.get_result()
                time.sleep(random.uniform(0.6,1.6))
        else:
            continue
            
#  修改C:/Users/dengyao.GENOMICS/Desktop/soft/pythoncode/pickle_temp/ 为你的桌面路径
       
pickle.dump(ls,open("D:/PythonProjects/yao/data/zp.txt","wb"))