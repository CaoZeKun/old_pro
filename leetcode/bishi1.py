# import sys
# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))
#
#


# import sys
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = 0
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = list(map(int, line.split()))
#         for v in values:
#             ans += v
#     print(ans)

# def search(a, n):
#     ans = 1e10
#     for i in range(n):
#         this_sum=0
#         j = i
#         while j < n-2:
#             this_sum+= a[j]
#             j+=1
#         if(abs((this_sum))<abs((ans))):
#             ans = abs(this_sum)
#
#     return ans
#
# import sys
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = sys.stdin.readline().strip()
#     values = list(map(int, n.split()))
#     # n = int(sys.stdin.readline().strip())
#     # ans = 0
#     # for i in range(n):
#     #     # 读取每一行
#     #     line = sys.stdin.readline().strip()
#     #     # 把每一行的数字分隔后转化成int列表
#     #     values = list(map(int, line.split()))
#     #     for v in values:
#     #         ans += v
#     # print(ans)
#     n = len(values)
#     print(n)
#     print(type(values))
#     a = search(values,n)
import sys
class Str(str):
    def __lt__(self, other):
        return str.__lt__(self + other, other + self)

while True:
    try:
        n = int(input())
        # a = list(map(Str, input().rstrip("\n").split()))
        line = sys.stdin.readline().strip()
        values = list(map(Str, line.split()))
        print(int("".join(sorted(values, reverse=True))))
    except:
        break