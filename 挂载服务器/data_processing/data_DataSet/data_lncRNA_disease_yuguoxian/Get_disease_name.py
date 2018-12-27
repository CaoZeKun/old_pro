import numpy as np
import pandas as pd


def save_to_file2(file_name, contents):
    with open(file_name, 'a') as f:
        f.write(contents + '\n')

def save_to_file3(file_name,contents0, contents1,contents2):
    with open(file_name, 'a') as f:
        f.write(contents0 + '\t'+contents1 + '\t'+contents2 + '\n')

diseases_name = pd.read_table('./data/Dinc/diseases.txt',header=None,encoding='gbk')
disease_DOID = pd.read_table('./data/disName_newPosi_oldPosi.txt',header=None,encoding='gbk')

disease_DOID = disease_DOID.iloc[:,0]
diseases_name1 = diseases_name.iloc[:,1]
print(disease_DOID)

count1 = 0
for j, d_id in enumerate(disease_DOID):
    for i, d_name_id in enumerate(diseases_name1):
        if d_name_id == d_id:
            # disease.append(d_id_yu)
            save_to_file3('./data_create/disease_name.txt', str(count1),diseases_name.iloc[i,1],diseases_name.iloc[i,0])
            count1 += 1
            # print(count1)




