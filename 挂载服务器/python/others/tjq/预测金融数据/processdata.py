#coding: utf-8
import pandas as pd
import numpy as np
import csv

# with open("C:\Users\ljy\Desktop\quote00001\datastock.csv","r") as f:
#     reader=csv.reader(f)
#     rows =[row[1] for row in reader]
#   print rows 拿到了一列年份




d=pd.read_csv('datastock\data1.csv',usecols=['营业总收入(万元)','利息收入(万元)',\
                                                                     '手续费及佣金收入(万元)', \
                                                       '其他业务收入(万元)','营业总成本(万元)','利息支出(万元)', \
                                                       '手续费及佣金支出(万元)','营业税金及附加(万元)','销售费用(万元)', \
                                                       '资产减值损失(万元)','公允价值变动收益(万元)','投资收益(万元)', \
                                                       '汇兑收益(万元)','营业利润(万元)','营业外收入(万元)', \
                                                       '营业外支出(万元)','利润总额(万元)','所得税费用(万元)', \
                                                       '净利润(万元)','归属于母公司所有者的净利润(万元)', \
                                                   '基本每股收益', '稀释每股收益'])

#根据标签读出需要进行计算的那些列
#print(int(d.loc[3])

# d.loc[0]=int(d.loc[0])-int(d.loc[1])

#d.loc[0]=int(d.loc[0])-int(d.loc[1])

# r2=r0-r1
# print(r2)
# r0=list(map(float,d.loc[0]))
# r1=list(map(float,d.loc[1]))
# r0=np.array(r0)
# r1=np.array(r1)
# d.loc[0]=r0-r1

d.loc[0]=np.array(list(map(float,d.loc[0])))-np.array(list(map(float,d.loc[1])))   #单独计算2017年的数据


# #r3=list(map(int,r2))
# #print(d.loc[0])
numbers=[2,6,10,14,18,22,26,30,34,38]     #根据numbers可以依次取到每年12月的数据

for i in numbers:                         #根据逻辑循环计算
     d.loc[i] = np.array(list(map(float,d.loc[i]))) - np.array(list(map(float,d.loc[i + 1])))
     d.loc[i + 1] = np.array(list(map(float,d.loc[i + 1]))) - np.array(list(map(float,d.loc[i + 2])))
     d.loc[i + 2] = np.array(list(map(float,d.loc[i + 2]))) - np.array(list(map(float,d.loc[i + 3])))


# #print(d)

d.to_csv("datanew.csv")
