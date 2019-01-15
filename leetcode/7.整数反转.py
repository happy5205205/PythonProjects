"""
    给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。

    示例 1:
    输入: 123
    输出: 321
    示例 2:
    输入: -123
    输出: -321
    示例 3:
    输入: 120
    输出: 21
"""
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        # 判断是否事个位数，如果是个位数，直接返回
        if -10 < x < 10:
            return x
        # 将数字转换成字符串
        str_x = str(x)
        # 判断第一位是否为-1
        if str_x[0] == '-':
            # 从第二开始反转
            str_x = str_x[1:][::-1]
            # 字符串转换成数字
            x = int(str_x)
            # 加上符号
            x = -x
        else:
            str_x = str_x[::-1]
            x = int(str_x)
        return x if -2147483648 < x < 2147483647 else 0
s = Solution()
y = s.reverse(x = -123)
print(y)