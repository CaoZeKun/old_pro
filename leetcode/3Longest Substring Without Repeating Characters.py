# Given a string, find the length of the longest substring without repeating characters.

class Solution:
    def lengthOfLongestSubstring1(self, s):
        """
        :type s: str
        :rtype: int
        """
        max_sub_str = []
        compare_str = []

        flag = 1
        for i in s:
            if i in compare_str:
                if len(max_sub_str) >= len(compare_str):
                    compare_str = []

                else:
                    max_sub_str = compare_str
                    compare_str = []

            if i not in max_sub_str and flag:
                max_sub_str.append(i)
                # print(max_sub_str)
            else:
                flag = 0

            if not flag:
                compare_str.append(i)
                # print(compare_str)

        if len(max_sub_str) > len(compare_str):
            # max_sub_str = "".join(max_sub_str)
            # print(type(max_sub_str))
            return "".join(max_sub_str)
        else:
            # compare_str = "".join(compare_str)
            # print(type(compare_str))
            return "".join(compare_str)


def stringToString(input):
    import json

    return json.loads(input)


def main():
    import sys
    import io
    def readlines():
        for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8'):
            yield line.strip('\n')

    lines = readlines()
    while True:
        try:
            line = next(lines)
            s = stringToString(line);

            ret = Solution().lengthOfLongestSubstring1(s)

            out = str(ret);
            print(out)
        except StopIteration:
            break


if __name__ == '__main__':
    # main()




    s1 = Solution()
    str_s1 = 'abcabcbb'
    str_s2 = 'bbbbb'
    str_s3 = 'pwwkew'
    str_s4 = 'abcabcbb'
    str_s5 = 'dvdf'
    max_sub_string = s1.lengthOfLongestSubstring1(str_s5)
    print(len(max_sub_string))
    print(max_sub_string)








