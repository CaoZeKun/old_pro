import numpy as np
import pandas as pd

def save_to_file2(file_name, contents):
    with open(file_name, 'a') as f:
        f.write(contents + '\n')

# dis_sim = np.loadtxt('./data_create/diseases_sim')
dis_name = pd.read_table('./data_create/disease_name_yu_Dinc.txt',header=None,encoding='gbk')
lncRNA_name_yuguoxian = pd.read_table('./data/yuguoxian_lncRNA_name.txt',header=None,encoding='gbk')
disease_DOID_yuguoxian = pd.read_table('./data/yuguoxian_diseases_doid.txt',header=None,encoding='gbk')

lncRNA_disease_association_yuguoxian = np.loadtxt('./data/yuguoxian_lncRNA_diseases.txt',encoding='gbk')


lnc_dis_association = np.zeros((len(lncRNA_name_yuguoxian),len(dis_name)))

dis_name = dis_name.iloc[:,0]
disease_DOID_yuguoxian = disease_DOID_yuguoxian.iloc[:,0]
count = 0

for j, dis_new in enumerate(dis_name):
    for l, dis_yu in enumerate(disease_DOID_yuguoxian):
        if dis_yu == dis_new :
            lnc_dis_association[:,j] = lncRNA_disease_association_yuguoxian[:,l]
            dis_association = str(np.sum(lnc_dis_association[:,j]))
            dis = dis_new + '\t' + dis_association
            print(dis_association)
            save_to_file2('./data_create/disname_disassolen.txt', dis)
            disName_newPosi_oldPosi = dis_yu + '\t' + str(j) + '\t' + str(l)
            count += 1
            save_to_file2('./data_create/disName_newPosi_oldPosi.txt', disName_newPosi_oldPosi)
print(count)
print(np.sum(lnc_dis_association,axis=0))

print(np.sum(lnc_dis_association))
np.savetxt('./data_create/lnc_dis_association.txt',lnc_dis_association,fmt='%d')  # 2687









