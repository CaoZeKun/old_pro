import pandas as pd
from operator import itemgetter, attrgetter


lrb = pd.read_csv("lrb.csv",encoding='gbk',header=0)
lrb = lrb.T
debt = pd.read_csv("debt.csv",encoding='gbk',header=0)
debt = debt.T
money = pd.read_csv("money.csv",encoding='gbk',header=0)
money = money.T
# print(lrb)

lrb1 = lrb.iloc[:,0:10:2]
lrb2 = lrb.iloc[:,3:15:3]
# print(lrb1)
# print(lrb2)
lrb = pd.concat((lrb1,lrb2),axis=1)
# print(lrb)
# lrb = sorted(lrb, key= lambda x:x[0])
lrb.to_csv('lrb1.csv')
# lrb1 = pd.read_csv("lrb1.csv")
# print(lrb1)


