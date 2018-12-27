# coding: utf-8
import pandas as pd
import os
import csv
import numpy as np
from operator import itemgetter, attrgetter

path = r"data"    # data文件夹的路径
stocks = []       # 用来装股票代码
numbers = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38]     # 根据numbers可以依次取到每年12月的数据


for name in os.listdir(path):             # 获取文件夹名字

    namenew = name[5:]                    # 切片获取股票代码，添加股票代码的时候用
    # stocks.append(namenew)                 把所有的股票代码存储到stocks里
    pathLrb = "data\\"+name+"\\lrb.csv"     # 拼接成lrb的路径
    pathDebt = "data\\"+name+"\\debt.csv"   # 拼接成debt的路径
    pathMoney = "data\\"+name+"\\money.csv"  # 拼接成money的路径
    resultCombinePath = "datacombine\\"+name+".csv"
    stockPath = "datastock\\"+name+".csv"
    processPath = "dataprocess\\"+name+".csv"
    try:
        lrb = pd.read_csv(pathLrb, encoding='gbk', header=0)    # 读lrb表
    except Exception:                                           # 有可能是空表，所以要捕捉异常
        continue                                                # 是空表就跳过，读下一个文件夹的表
    else:
                                                                # lrb = lrb.T
        debt = pd.read_csv(pathDebt, encoding='gbk', header=0)  # 读debt表
# debt = debt.T
        money = pd.read_csv(pathMoney, encoding='gbk', header=0)  # 读money表
# money = money.T
# print(lrb)
        debt1 = debt.iloc[:, 1:]   # 读取除第一列的所有列
        debt2 = debt.iloc[:, 0]    # 读取第一列
        money1 = money.iloc[:, 1:]
        money2 = money.iloc[:, 0]
        lrb1 = lrb.iloc[:, 1:]
        lrb2 = lrb.iloc[:, 0]
        data_1 = pd.concat((debt1, money1, lrb1), axis=0)   # 拼接除第一列的数据 列对齐
        data_0 = pd.concat((debt2, money2, lrb2), axis=0)   # 拼接第一列 列对齐
        data_concat = pd.concat((data_1, data_0), axis=1)    # 拼接第一列和剩余部分 行对齐
        data_trans = data_concat.iloc[:, ::-1]               # 把列倒叙

        x = [1, 2]      # 需要删除的列号
        try:
            data = data_trans.drop(data_trans.columns[x], axis=1)  # inplace = True 是在原文件上保存，删除指定列
        except:
            continue
        else:
            data = data.T      # 将数据转置
            data.to_csv(resultCombinePath, index=True, header=0)   # index = true 去掉索引

            # 开始添加股票代码
            with open(resultCombinePath, "r", encoding='UTF-8') as f:  # 合成后的文件夹路径
                reader = csv.reader(f)
                rows = [row for row in reader]  # 按行读到rows里面，每一行都按一个列表存储

            with open(stockPath, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(rows)):
                    writer.writerow([namenew] + rows[i])  # 对每一行进行重新拼接，再写回新的文件中

            # 开始对添加股票代码后的文件做计算
            # d = pd.read_csv(stockPath, usecols=['营业总收入(万元)', '利息收入(万元)', '手续费及佣金收入(万元)', \
            #                                          '其他业务收入(万元)', '营业总成本(万元)', '利息支出(万元)', \
            #                                          '手续费及佣金支出(万元)', '营业税金及附加(万元)', '销售费用(万元)', \
            #                                          '资产减值损失(万元)', '公允价值变动收益(万元)', '投资收益(万元)', \
            #                                          '汇兑收益(万元)', '营业利润(万元)', '营业外收入(万元)', \
            #                                          '营业外支出(万元)', '利润总额(万元)', '所得税费用(万元)', \
            #                                          '净利润(万元)', '归属于母公司所有者的净利润(万元)', \
            #                                          '基本每股收益', '稀释每股收益'], encoding='gbk')
            #
            # d.loc[0] = np.array(list(map(float, d.loc[0])))-np.array(list(map(float, d.loc[1])))   # 单独计算2017年的数据
            #
            # for i in numbers:                         # 根据逻辑循环计算
            #     d.loc[i] = np.array(list(map(float, d.loc[i]))) - np.array(list(map(float, d.loc[i + 1])))
            #     d.loc[i + 1] = np.array(list(map(float, d.loc[i + 1]))) - np.array(list(map(float, d.loc[i + 2])))
            #     d.loc[i + 2] = np.array(list(map(float, d.loc[i + 2]))) - np.array(list(map(float, d.loc[i + 3])))
            #
            # # print(d)
            #
            # d.to_csv(processPath)








