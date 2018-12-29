# _*_ coding:utf-8 _*_
"""
    给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

    你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

    示例:

    给定 nums = [2, 7, 11, 15], target = 9

    因为 nums[0] + nums[1] = 2 + 7 = 9
    所以返回 [0, 1]
"""
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # 循环遍历下标
        for i in range(len(nums)):
            # target值减去当前下标在数组中的值，就是要在数组中查找另一个值
            num2 = target - nums[i]

            if num2 in nums:
                j = nums.index(num2)

                if i != j:
                    return [i, j] if i < j else [j, i]
solu = Solution()
print(solu.twoSum([4, 11, 2, 4, 15], 8))