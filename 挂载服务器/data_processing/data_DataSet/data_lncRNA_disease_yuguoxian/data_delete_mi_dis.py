import pandas as pd
import numpy as np

mi_dis = pd.read_table('./data/yuguoxian_mi_dis.txt',header=None,encoding='gbk')
dis_length = len(mi_dis.iloc[0,:])
# mi_dis = np.loadtxt('./data/yuguoxian_mi_dis.txt',encoding='gbk')
index_delete = pd.read_table('./data_create/disName_newPosi_oldPosi.txt',header=None,encoding='gbk')



index_delete = index_delete.iloc[:,2]
for j in range(405):
    print(index_delete.iloc[j]+1)
#
# for i in range(dis_length-1,-1,-1):
#     print(i)
