# lnc_sim 是有一行全为0的
# 这里建立的是把一行全为0去掉


"""删除时，lncRNA_sim 不仅要删除行，还要删除列
          lncRNA_dis_A 删除行 应该就可以了"""
import pandas as pd
import numpy as np

lncRNA_yuguoxian = pd.read_table('./data/yuguoxian_lncRNA_name.txt',header=None,encoding='gbk')


A_lnc_dis = np.loadtxt('./data_create/lnc_dis_association.txt', encoding='gbk')
lnc_sim = np.loadtxt('./data_create/lnc_sim.txt',encoding='gbk')  # 0 24839  24780( eye 1 )


row_delete = []
i = 0
j = len(lncRNA_yuguoxian)
count = 0
while i < j:
    if np.sum(lnc_sim[i]) == 1:
        row_delete.append(i)
        count += 1
    i += 1
print(count)
print(row_delete)

# i = 0
# j = len(lncRNA_yuguoxian)
# while i < j:
#     if np.sum(lnc_sim[i]) == 1:
#         lnc_sim = np.delete(lnc_sim, i, 0)  # 删除A的第二行
#
#         print(len(lncRNA_yuguoxian))
#         print(i)
#
#         A_lnc_dis = np.delete(A_lnc_dis,i,0)
#         j = j - 1
#         i = i - 1
#     i += 1
#
# lncRNA_yuguoxian = lncRNA_yuguoxian.drop(i, axis=0)  # row
#
#
# np.savetxt('./data_create_delete_lncsim0/lncsim.txt',lnc_sim)



# count2 = 0
# for i,l_name_yu in enumerate(lncRNA_yuguoxian):
#     for j ,l_name_Dinc in enumerate(lncRNA_Dinc):
#         if l_name_yu == l_name_Dinc:
#             count2 += 1
# print(count2)  # 78
















