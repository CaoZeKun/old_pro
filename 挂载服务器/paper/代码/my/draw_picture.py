import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt

ROC_FPR = np.loadtxt('ROC_FPR.txt')
ROC_TPR = np.loadtxt('ROC_TPR.txt')
R = np.loadtxt('R.txt')
P = np.loadtxt('P.txt')
X_Y_P_R = np.loadtxt('CNNLDA_PR.txt')



DL_R = X_Y_P_R[0]
DL_P = X_Y_P_R[1]

SIMC_R = R[1]
SIMC_P = P[1]

ANML_R = R[2]
ANML_P = P[2]

MFLDA_R = R[3]
MFLDA_P = P[3]

LDAP_R = R[4]
LDAP_P = P[4]

fig = plt.figure()
ax2 = plt.subplot(122)

p2, = plt.plot(SIMC_R, SIMC_P, 'lightgreen')
p1, = plt.plot(DL_R, DL_P, 'salmon')
p5, = plt.plot(LDAP_R, LDAP_P, 'lightskyblue')
p3, = plt.plot(ANML_R, ANML_P, 'gold')
p4, = plt.plot(MFLDA_R, MFLDA_P, 'orchid')

l1 = plt.legend([p1, p2, p3, p4, p5],
                ["CNNLDA", "SIMCLDA",
                 "Ping's method", "MFLDA",
                 "LDAP"],
                loc='upper right')

ax2.set_title('(b)PR curves',fontsize=10,) #color='b'

plt.xlabel("Recall")
plt.ylabel("Precision")

# plt.show()

DL_FPR = ROC_FPR[0]
DL_TPR = ROC_TPR[0]

SIMC_FPR = ROC_FPR[1]
SIMC_TPR = ROC_TPR[1]

ANML_FPR = ROC_FPR[2]
ANML_TPR = ROC_TPR[2]

MFLDA_FPR = ROC_FPR[3]
MFLDA_TPR = ROC_TPR[3]

LDAP_FPR = ROC_FPR[4]
LDAP_TPR = ROC_TPR[4]
ax1 = plt.subplot(121)
p2, = plt.plot(SIMC_FPR, SIMC_TPR, 'lightgreen')
p1, = plt.plot(DL_FPR, DL_TPR, 'salmon')
p5, = plt.plot(LDAP_FPR, LDAP_TPR, 'lightskyblue')
p3, = plt.plot(ANML_FPR, ANML_TPR, 'gold')
p4, = plt.plot(MFLDA_FPR, MFLDA_TPR, 'orchid')

# l1 = plt.legend([p1, p2, p3, p4, p5],
#                 ["CNNLDA(0.9519)", "SIMCLDA(0.7464)",
#                  "Ping's method(0.8714)", "MFLDA(0.6262)",
#                  "LDAP(0.8634)"],
#                 loc='lower right')
l1 = plt.legend([p1, p2, p3, p4, p5],
                ["CNNLDA", "SIMCLDA",
                 "Ping's method", "MFLDA",
                 "LDAP"],
                loc='lower right')
ax1.set_title('(a)ROC curves',fontsize=10,)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
plt.xlabel("FPR")
plt.ylabel("TPR")
# plt.title('Five fold Cross−Validation', fontsize='large', fontweight='bold')
plt.xlim(0, 1)
plt.ylim(0, 1)
# plt.savefig("ROC.png")
plt.show()


DL_Rcall = [0.89578036, 0.96193105, 0.98832547, 0.99176669, 0.99662401, 0.99796405, 0.99899497, 1.]
SIMC_Rcall =[0.49336316 ,0.62984015 ,0.74055623, 0.80332763, 0.88315825, 0.93033198, 0.93287759, 1.]
LDAP_Rcall =[0.68530222 ,0.8170659 , 0.88043092, 0.93338218 ,0.97059227 ,0.9708827 ,0.97458565 ,1.]
ANML_Rcall =[0.6892776 , 0.81294548 ,0.87507669, 0.92709645, 0.98577204 ,0.9943646, 0.99488292 ,1.]
MFLDA_Rcall =[0.42045601 ,0.53858183 ,0.60906449, 0.65534971 ,0.7060153, 0.76008743 ,0.8435835 , 1.]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
size = 8
x = np.arange(size)
total_width, n = 0.8, 5
width = total_width / n
x = x - (total_width - width) / 2
ax.set_xticks(x)
plt.bar(x - 0.3, DL_Rcall, width=width, label='CNNLDA', color='salmon')
plt.bar(x - 0.3 + width, SIMC_Rcall, width=width, label='SIMCLDA', color='lightgreen')
plt.bar(x - 0.3 + 2 * width, ANML_Rcall, width=width, label="Ping's method", color='lightskyblue')
plt.bar(x - 0.3 + 3 * width, MFLDA_Rcall, width=width, label='MFLDA', color='gold')
plt.bar(x - 0.3 + 4 * width, LDAP_Rcall, width=width, label='LDAP', color='orchid')
# ax.set_xticklabels(['one','two','three','four','five','two','three','four',],rotation=45,fontsize=12)
ax.set_xticklabels(['Top30', 'Top60', 'Top90', 'Top120', 'Top150', 'Top180', 'Top210', 'Top240', ])

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 1])
ax.legend(loc='upper left', bbox_to_anchor=(0, 1.12),ncol=5)
plt.ylabel("Recall")
# plt.legend(loc='NorthWestOutside')
# plt.savefig("filename.png")
plt.show()
# a =[]
# a.append(['name','disease'])
# a.append(['name','disease'])
# a.append(['name','disease'])
# a = np.array(a)
# df = pd.DataFrame(a)
# df.to_excel('a.xls')
#
#
# y = [0.875345857679525, 0.8871851786851936, 0.917041348080016, 0.9299993676420687, 0.9479663609607876, 0.9537739276879238, 0.9513553206153039]
# x = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
#
#
# plt.plot(x,y)
# plt.show()
#
# for i in x:
#     print(i)