import numpy as np
import pandas as pd



A = np.loadtxt("../data_create/lnc_dis_association.txt")
# # print(np.sum(A,axis=0))
idx = np.where(np.sum(A,axis=0)>26)
print(idx)
# a = [9,  10,  61,62,69, 113,126, 140, 156, 204, 211, 233, 256,297, 316, 334, 335, 338, 383]

a = [7,9, 10, 61,62,69,
     113,126, 140, 156,211, 233,
     316, 334, 335, 338]
print(a)
print(np.sum(A,axis=0)[a])
# # length = [114.  60.  34. 130. 130.  42.  50.  62.  40. 181.  34.  40.  40.  46. 34.]
# A = np.loadtxt('AUC_15_dis1.txt',delimiter=',')
# print(A)
# np.savetxt('AUC_15_dis1.txt',A,delimiter=',',newline='\n')

# Md = np.loadtxt("../data_create/mi_dis.txt")
# print(np.sum(Md))
CNNLDA = np.loadtxt('CNNLDA_15_AUC_PR.txt')
CNNLDA_ROC = CNNLDA[0]
CNNLDA_PR = CNNLDA[1]

a = [62, 69, 113, 140, 156, 178, 181, 187, 233, 297]

dis_15_ROC = np.loadtxt('AUC_15_dis2.txt')
# CNN = dis_15_ROC[0]
CNN = CNNLDA_ROC
SIMC = dis_15_ROC[1]
LDAP = dis_15_ROC[2]
ANML = dis_15_ROC[3]
MFLDA = dis_15_ROC[4]
ROC_AUC = []
ROC_AUC.append(a)
ROC_AUC.append(CNN)
ROC_AUC.append(SIMC)
ROC_AUC.append(ANML)
ROC_AUC.append(MFLDA)
ROC_AUC.append(LDAP)
ROC_AUC = np.array(ROC_AUC)
ROC_AUC = ROC_AUC.T

ROC_AUC = pd.DataFrame(ROC_AUC)

# ROC_AUC.to_excel('AUC_15_dis_last3.xls',float_format='%.3f')

print(ROC_AUC)

dis_15_PR = np.loadtxt('PR_15_dis_last2.txt')
# CNN = dis_15_PR[0]
CNN = CNNLDA_PR
SIMC = dis_15_PR[1]
LDAP = dis_15_PR[2]
ANML = dis_15_PR[3]
MFLDA = dis_15_PR[4]
PR_AUC = []
PR_AUC.append(a)
PR_AUC.append(CNN)
PR_AUC.append(SIMC)
PR_AUC.append(ANML)
PR_AUC.append(MFLDA)
PR_AUC.append(LDAP)
PR_AUC = np.array(PR_AUC)
print(np.shape(PR_AUC))
PR_AUC = PR_AUC.T

PR_AUC = pd.DataFrame(PR_AUC)


# PR_AUC.to_excel('PR_15_dis_last3.xls',float_format='%.3f')

print(PR_AUC)
# dis_15_ROC =pd.read_table('AUC_15_dis2.txt',header=None,delimiter=' ')
# dis_15_ROC = dis_15_ROC.transpose()
# dis_15_ROC.to_excel('AUC_15_dis_last2.xls',float_format='%.4f')
#
# dis_15_PR = pd.read_table('PR_15_dis_last2.txt',header=None,delimiter=' ')
# dis_15_PR = dis_15_PR.transpose()
# dis_15_PR.to_excel('PR_15_dis_last2.xls',float_format='%.4f')



A = np.loadtxt("../data_create/lnc_dis_association.txt")

print(np.sum(A,axis=1))

