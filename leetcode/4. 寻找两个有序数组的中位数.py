"""
    给定两个大小为 m 和 n 的有序数组 nums1 和 nums2。
    请你找出这两个有序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
    你可以假设 nums1 和 nums2 不会同时为空。
    示例 1:
    nums1 = [1, 3]
    nums2 = [2]
    则中位数是 2.0
    示例 2:
    nums1 = [1, 2]
    nums2 = [3, 4]
    则中位数是 (2 + 3)/2 = 2.5
"""
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = nums1 + nums2
        nums.sort()
        length = len(nums)
        if length == 2:
            return (nums[0] + nums[1]) / 2
        if length % 2 == 0:
            return (nums[length // 2 - 1] + nums[(length // 2)]) / 2
        return nums[length // 2]

sou = Solution()
mid = sou.findMedianSortedArrays([1,9],[2])
print(mid)