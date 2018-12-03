"""
    1、 比较两个版本字符串version1和version2两个字符串，用空格分割。
        每个字符串为一个version字符串，非空，只包含数字和字符.
"""



# ******************************开始写代码*****************************


def compareVersionNumber (version1, version2):
    str(version1)
    str(version2)
    v1 = version1.split(".")
    v2 = version2.split(".")
    len1 = len(v1)
    len2 = len(v2)
    lenMax = max(len1,len2)

    for x in range(lenMax):
        v1Token = 0
        if x < len1:
            v1Token = int(v1[x])
        v2Token = 0
        if x < len2:
            version2 = int(v2[x])
        if v1Token < v2Token:
            return -1
        if v1Token > version2:
            return 1
    return 0

# ******************************结束写代码******************************
if __name__ == '__main__':

    num1 = input()
    num2=num1.split(' ')
    print('aaaaaaaaa',num2)
    res = compareVersionNumber(num2[0],num2[1])
    print(str(res))

