import numpy as np
import pandas as pd


def save_to_file2(file_name, contents):
    with open(file_name, 'a') as f:
        f.write(contents + '\n')


"""read data"""
# yu guo xian
lncRNA_yuguoxian = pd.read_table('./data/yuguoxian_lncRNA_name.txt',header=None,encoding='gbk')
disease_DOID_yuguoxian = pd.read_table('./data/yuguoxian_diseases_doid.txt',header=None,encoding='gbk')
lncRNA_disease_association_yuguoxian = pd.read_table('./data/yuguoxian_lncRNA_diseases.txt',header=None,encoding='gbk')

# Dinc
diseases_DOID_Dinc = pd.read_table('./data/Dinc/diseases.txt',header=0,encoding='gbk')
lncRNA_Dinc = pd.read_table('./data/Dinc/lncRNAs.txt',header=0,encoding='gbk')

# ILNCSIM
lncRNA_ILNCSIM = pd.read_excel('./data/ILNCSIM.xlsx',header=None,encoding='gbk')

# SIMCLDA
lncRNA_SIMCLDA = pd.read_table('./data/lncRNA_Name_SIMCLDA.txt',header=0,encoding='gbk')


"""disease doid same number"""
diseases_DOID_Dinc = diseases_DOID_Dinc.iloc[:,1]
disease_DOID_yuguoxian = disease_DOID_yuguoxian.iloc[:,0]

count1 = 0
disease = []
for i,d_id_yu in enumerate(disease_DOID_yuguoxian):
    for j ,d_id_Dinc in enumerate(diseases_DOID_Dinc):
        if d_id_yu == d_id_Dinc:
            disease.append(d_id_yu)
            save_to_file2('./data_create/disease_name_yu_Dinc.txt', d_id_yu)
            count1 += 1
print(count1)  # 405
disease_sim_Dinc = pd.read_table('./data/Dinc/disease_sim_wang_Dinc.txt',header=0,encoding='gbk')
# disease_sim_Dinc_1 = disease_sim_Dinc.iloc[:,0]
# disease_sim_Dinc_2 = disease_sim_Dinc.iloc[:,1]
# disease_sim_Dinc_number = disease_sim_Dinc.iloc[:,2]

# form disease_sim
disease_sim = np.zeros((405,405))
for l in range(len(disease_sim_Dinc)):
    if l % 10000 ==0 :
        print(l)
    for i,dis1 in enumerate(disease):
        if dis1 == disease_sim_Dinc.iloc[l, 0]:
            for j, dis2 in enumerate(disease):
                if dis2 == disease_sim_Dinc.iloc[l,1]:
                    disease_sim[i,j] = disease_sim_Dinc.iloc[l,2]
                    break
            break

np.savetxt('./data_create/diseases_sim.txt',disease_sim)













"""lncRNA same number in yu and Dinc """
# lncRNA_Dinc = lncRNA_Dinc.iloc[:,0]
# lncRNA_yuguoxian = lncRNA_yuguoxian.iloc[:,0]
#
# count2 = 0
# for i,l_name_yu in enumerate(lncRNA_yuguoxian):
#     for j ,l_name_Dinc in enumerate(lncRNA_Dinc):
#         if l_name_yu == l_name_Dinc:
#             count2 += 1
# print(count2)  # 78

"""lncRNA same number in yu and ILNCSIM """
# lncRNA_ILNCSIM = lncRNA_ILNCSIM.iloc[:,0]
# lncRNA_yuguoxian = lncRNA_yuguoxian.iloc[:,0]
#
# count3 = 0
# for i,l_name_yu in enumerate(lncRNA_yuguoxian):
#     for j ,l_name_ILN in enumerate(lncRNA_ILNCSIM):
#         if l_name_yu == l_name_ILN:
#             count3 += 1
# print(count3)  # 54

"""lncRNA same number in yu and ILNCSIM """
# lncRNA_SIMCLDA = lncRNA_SIMCLDA.iloc[:,0]
# lncRNA_yuguoxian = lncRNA_yuguoxian.iloc[:,0]
#
# count4 = 0
# for i,l_name_yu in enumerate(lncRNA_yuguoxian):
#     for j ,l_name_SIM in enumerate(lncRNA_SIMCLDA):
#         if l_name_yu == l_name_SIM:
#             count4 += 1
# print(count4)  # 82


