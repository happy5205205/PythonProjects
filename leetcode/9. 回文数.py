"""
    判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

    示例 1:
    输入: 121
    输出: true
    示例 2:
    输入: -121
    输出: false
    解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
    示例 3:
    输入: 10
    输出: false
    解释: 从右向左读, 为 01 。因此它不是一个回文数。
"""
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
    # 判断0到10之间的个位数
        if 0 < x < 10:
            return True
        str_x = str(x)

        if str_x[0] == '-':
            return False
        else:
            str_x = str_x[0:][::-1]
            x1 = int(str_x)
            if x1 == x:
                return True
            else:return False

s = Solution()
print(s.isPalindrome(-121))

