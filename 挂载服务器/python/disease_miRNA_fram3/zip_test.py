a = [2,3,1]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)     # 打包为元组的列表
# print(list(zipped))
# 若之前就list后面会报错

# 元素个数与最短的列表一致
#print(zip(a,c) )
# 与 zip 相反，可理解为解压，返回二维矩阵式
#print(zip(*zipped))

#zipped = list(zipped)

d = sorted(zipped, key=lambda x: x[1])   # sort by age
# print(d)  # [(2, 4), (3, 5), (1, 6)]

c = zip(*d)
c1 = list(c)


# print(c1[0])  # (2, 3, 1)
# print(list(c1[1]))  # [4, 5, 6]

# for x in range(len(a)-1,-1,-1):
    # print (a[x])

from operator import itemgetter
import numpy as np
a = [[1,10,8],
    [5,2,11],
    [6,1,10]]
a = np.array(a).transpose()
#a = a.transpose()
print(a)

d = temp_test_data = sorted(a, key=itemgetter(0), reverse=True)

l = temp_test_data = sorted(a, key=lambda x:x[0], reverse=True)
print(d)
print(l)