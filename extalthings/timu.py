import numpy as np
import sys

import numpy as np
n = int(sys.stdin.readline())
# 输入
new_list = []
for i in range(n):
    temp = sys.stdin.readline().split()
    l = int(temp[0])
    r = int(temp[1])
    temp2 = np.array(range(l,r))
    new_list.extend(temp2)
# 计算
temp_list = list(set(new_list))
max = 0
for x in temp_list:
    max_temp = new_list.count(x)
    if max_temp > max:
        max = max_temp
print(max)
# n = int(input("Please input the number of countries"))
# print("Please input Ai")
# temp_save = sys.stdin.readline()
# A_i = list(map(int,temp_save.split()))
# # # print(A_i)
# # n=5
# # A_i = [9,7,10,1,5]
# # A_i = np.array(A_i)
# def stay():
#     pass
#
# def buy(money,stone_price):
#     money -= stone_price
#     return money
# def sale(money,stone_price):
#     money += stone_price
#     return money
#
# max_money_value = 0
# max_get = []
# stone_get = 0
#
# max_value_index = np.where(A_i==np.max(A_i))
# print(max_value_index[0][0])
# list_temp = A_i
#
#
# flag = 1
# while flag:
#
#     left = list_temp[0:max_value_index[0][0]:-1]
#     right = list_temp[max_value_index[0][0]::-1]
#
#     for i in range(max_value_index[0][0]):
#
#
#
# #
# # for price in A_i:
# #     # a = A_i.index(max(A_i))
# #
# #     print(a)
# #     print(A_i[a])
#
#
# # for county in range(n):
# #
# #     stay()
#
#     # for price in A_i:
#
#
#
#
#
#
#
#
#
#
#









"""
bestV=0
curW=0
curV=0
bestx=None
def backtrack(i):
  global bestV,curW,curV,x,bestx
  if i>=n:
    if bestV<curV:
      bestV=curV
      bestx=x[:]
  else:
    if curW+w[i]<=c:
      x[i]=True
      curW+=w[i]
      curV+=v[i]
      backtrack(i+1)
      curW-=w[i]
      curV-=v[i]
    x[i]=False
    backtrack(i+1)
if __name__=='__main__':
  n=4
  c=12
  w=[3,4,4,5,]
  v=[6,10,3,9]
  x=[False for i in range(n)]
  backtrack(0)
  print(bestV)
  print(bestx)
  """















                # import sys
# """First """
# def lengthOfLongestSubstring(s):
#     # write your code here
#     res = 0
#     if s is None or len(s) == 0:
#         return res
#     d = {}
#     tmp = 0
#     start = 0
#     for i in range(len(s)):
#         if s[i] in d and d[s[i]] >= start:
#             start = d[s[i]] + 1
#         tmp = i - start + 1
#         d[s[i]] = i
#         res = max(res, tmp)
#     return res
# print("Please input the string")
# # s = input("Please input the string")
# s = sys.stdin.readline().strip()
# a = lengthOfLongestSubstring(s)
# print(a)



