# _*_ coding: utf-8 _*_
# @CREATE BY ZPENG
# @ time: 19:30
class Solution:
    def LCS(l1, l2):
        res = []
        l = [list(set(l2)) if len(set(l1)) > len(set(l2)) else list(set(l1))][0]
        l1_dict = {i: [k for k, v in enumerate(l1) if v == i] for i in l if i in l1 and i in l2}
        l2_dict = {i: [k for k, v in enumerate(l2) if v == i] for i in l if i in l1 and i in l2}

        for k, v in l1_dict.items():
            if len(v) == 1:
                res.append(k)
            elif len(v) <= len(l2_dict[k]):
                # l1_temp = {h: g for h, g in enumerate(v)}
                l1_temp = [v[i + 1] - v[i] for i in range(len(v) - 1)]
                l2_temp = [v[i + 1] - v[i] for i in range(len(l2_dict[k]) - 1)]

                if l1_temp.count(1) == 0 or l2_temp.count(1) == 0:
                    res.append(k)
                elif l1_temp.count(1) >= l2_temp.count(1):
                    res = res + [k] * (l2_temp.count(1) + 1)
                else:
                    res = res + [k] * (l1_temp.count(1) + 1)
            return res