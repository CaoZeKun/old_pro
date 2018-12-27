import numpy as np


A = np.loadtxt("../data_create/lnc_dis_association.txt")

num = np.sum(A,axis=0)

d = len(num)

print(d)

idx = np.where(np.sum(A,axis=0)>15)

c = len(idx[0])
print(c)

print(c/d)