"""
    测试python连接到odps
"""
from odps import ODPS
o = ODPS('LTAIFwE1F5V5Fucy','c1Daaf3vFHwu3PhLBK99iHvH2AqWC4','zhangp123',
         endpoint='http://service.cn.maxcompute.aliyun.com/api')
dual = o.get_table('20171210v7')

print(dual.name)
print(dual.schema)
print(dual.head(10))
# dual.drop()



