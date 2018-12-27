import numpy as np
import pandas as pd

lnc_name = pd.read_table('./case_study/yuguoxian_lncRNA_name.txt',header=None)
#
# lnc_69_index = pd.read_table('./case_study/results_predict_69.txt',header=None)
#
# d = lnc_69_index.loc[:,0]
# print(d)
# lnc_69= lnc_name.iloc[d]
# lnc_69.to_excel('lnc_69.xlsx')
#
# lnc_113_index = pd.read_table('./case_study/results_predict_113.txt',header=None)
#
# d = lnc_113_index.loc[:,0]
# lnc_113= lnc_name.iloc[d]
# lnc_113.to_excel('lnc_113.xlsx')
#
# lnc_140_index = pd.read_table('./case_study/results_predict_140.txt',header=None)
#
# d = lnc_140_index.loc[:,0]
# lnc_140= lnc_name.iloc[d]
# lnc_140.to_excel('lnc_140.xlsx')

# lnc_171_index = pd.read_table('./results_predict_171.txt',header=None)
#
# d = lnc_171_index.loc[:,0]
# lnc_171= lnc_name.iloc[d]
# lnc_171.to_excel('lnc_171.xlsx')

# lnc_276_index = pd.read_table('./results_predict_276.txt',header=None)
#
# d = lnc_276_index.loc[:,0]
# lnc_276= lnc_name.iloc[d]
# lnc_276.to_excel('lnc_276.xlsx')

# lnc_178_index = pd.read_table('./results_predict_178.txt',header=None)
#
# d = lnc_178_index.loc[:,0]
# lnc_178= lnc_name.iloc[d]
# lnc_178.to_excel('lnc_178.xlsx')

lnc_383_index = pd.read_table('./results_predict_383.txt',header=None)

d = lnc_383_index.loc[:,0]
lnc_383= lnc_name.iloc[d]
# lnc_383.to_excel('lnc_383.xlsx')
print(lnc_383)

# A = np.loadtxt("../data_create/lnc_dis_association.txt")

# print(A[161][140])
