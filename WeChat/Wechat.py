# _*_ coding: utf-8 _*_
"""
    时间；2018年09月10日
    版本：V1.0
    说明：分析微信好友数据

"""
import os
import itchat
import pandas as pd

def frindesinfo():
    itchat.login()
    frindes = itchat.get_friends(update=True)
    print('My nickname: %s' % frindes[0].NickName)

    fiendcount = len(frindes)
    print('I have %d friends' % fiendcount)

    gg = mm = unk = 0
    for f in frindes[1:]:
        sex = f["Sex"]
        if sex == 1:
            gg += 1
        elif sex == 2:
            mm += 1
        else:
            unk += 1
    print('gg={},mm={},unk={}'.format(gg, mm, unk))


def infoparams(keyparma,friends):
    ret = []
    for f in friends[1:]:
        ret.append(f[keyparma])
    return ret

def info2file():
    itchat.login()
    friends = itchat.get_friends(update = True)
    nicknames = infoparams("NickName", friends)
    sexs = infoparams("Sex", friends)
    provinces = infoparams("Province", friends)
    cities = infoparams("City",friends)
    signatures = infoparams("Signature",friends)

    info = {'NickName':nicknames, 'Sex':sexs, 'Province':provinces, 'City':cities, 'Signature':signatures}
    df = pd.DataFrame(info)
    df.to_csv('friends_info.csv', index=True, encoding='utf-8')


def main():
    # frindesinfo()
    info2file()
    # pass
if __name__ == '__main__':
    main()