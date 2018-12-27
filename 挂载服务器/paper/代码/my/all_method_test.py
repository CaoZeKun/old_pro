import aNewMethodLDAP_vector as ANML
import Attention_cnn as DL
import LDAP
import MFLDA_init_all_data as MFLDA
import SIMCLDA_2_innerproduct as SIMC
import numpy as np
import matplotlib.pyplot as plt
# import random




def read_data_flies():

    #  240 * 405  lncRNA * diseases
    A = np.loadtxt("../Data/lnc_dis_association.txt")  # 2687

    #  405 * 405 diseases * diseases
    Sd = np.loadtxt("../Data/dis_sim_matrix_process.txt")

    #  240 * 240  lncRNA * lncRNA
    # Sm = np.loadtxt("../data_create/lnc_Sim.txt")

    #  240 * 495 lncRNA * miRNA
    Lm = np.loadtxt("../Data/yuguoxian_lnc_mi.txt")  # 1002

    #  495 * 405 miRNA * diseases
    Md = np.loadtxt("../Data/mi_dis.txt")  # 13559
    #print("---1.read data files---")
    return A,Sd,Lm,Md






def test():
    k = 5
    A, Sd, Lm, Md = read_data_flies()

    all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, sava_association_A, \
    all_k_development, save_all_count_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed = DL.data_partitioning(
        A, k)


    SIMC_FPR, SIMC_TPR = SIMC.test2(A, Sd,save_all_count_A, sava_association_A, save_all_count_zero_every_time_changed, num, A_length)

    DL_FPR, DL_TPR = DL.test5(A, Sd, Lm, Md,all_k_Sd_associated, all_k_test_data,all_k_development, save_all_count_A)

    LDAP_FPR,LDAP_TPR = LDAP.test2(A,save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A)

    ANML_FPR, ANML_TPR = ANML.test3(A,save_all_count_A, sava_association_A, save_all_count_zero_every_time_changed, num, A_length)

    MFLDA_FPR, MFLDA_TPR = MFLDA.test2(A,save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length)

    p2, = plt.plot(SIMC_FPR, SIMC_TPR,'g')
    p1, = plt.plot(DL_FPR, DL_TPR,'r')
    p5, = plt.plot(LDAP_FPR, LDAP_TPR,'b')
    p3, = plt.plot(ANML_FPR, ANML_TPR,'y')
    p4, = plt.plot(MFLDA_FPR, MFLDA_TPR,'magenta')

    l1 = plt.legend([p1, p2,p3,p4,p5], ["DLACNNLDA(0.9415)","SIMCLDA(0.7464)", "ANMLDA(0.8723)",  "MFLDA(0.6545)", "LDAP(0.8826)"], loc='lower right')
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title('Five fold Cross−Validation', fontsize='large', fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()
    # for i in range(/k):






if __name__ == '__main__':
    test()
