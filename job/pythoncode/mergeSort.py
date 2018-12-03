# coding:utf-8
"""
    归并排序
"""


def MergeSort(nums):
    if len(nums) <= 1:
        return nums
    num = int(len(nums)/2)
    # 从中间，进行数据拆分，递归的返回数据进行迭排序
    left = MergeSort(nums[:num])
    rignt = MergeSort(nums[num:])
    print('left',left)
    print('rignt',rignt)
    print('*'*20)
    return Merge(left, rignt)


def Merge(left, right):
    l, r = 0, 0
    result = []
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l +=1
        else:
            result.append(right[r])
            r +=1
    result += left[l:]
    result += right[r:]
    return result

def main():
    nums = [2, 6, 8, 5, 1, 4, 9, 3, 7]
    res = MergeSort(nums)
    print('res',res)
if __name__ == '__main__':
    main()