import numpy as np
import svd
import copy
import datetime
import random
from operator import attrgetter
import matplotlib.pyplot as plt
random.seed(1)

# define a class
class value_index():
    def __init__(self,num,i,j):
        self.value = num  # save value
        self.value_x = i  # save index of row
        self.value_y = j  # save index of column


# cross_validation
def crossvalidation(matrix_A,k):
    k = k
    sava_association_A = []# save association  diseases and drugs
    save_all_count_A =[]# save k time changed A
    save_all_count_zero_not_changed = []  # all zero, but no  number which should changed ( 1 to 0 )
    save_all_count_zero_every_time_changed = []  # save k * number which should changed ( 1 to 0 )
    save_count_zero = []  # record current zero and its location

    #if 1 save idex
    for i in range(matrix_A.shape[0]):
        for j in range(matrix_A.shape[1]):
            if matrix_A[i][j]== 1:
                save_temp_one = value_index(matrix_A[i][j],i,j)
                sava_association_A.append(save_temp_one)
            else:
                save_temp_zero = value_index(matrix_A[i][j],i,j)
                save_count_zero.append(save_temp_zero)
    save_all_count_zero_not_changed.extend(save_count_zero)  # but just save one time

    random.shuffle(sava_association_A)  # shuffle data
    A_length = len(sava_association_A)
    num = int(A_length / k)

    # save K time input
    for count in range(k):
        temp_count_zero = []  # record current changed number and location
        temp_A = copy.deepcopy(matrix_A)  # sava changed matrix A time count
        if(count == k-1):
            for i in range(A_length-(num*count)):
                temp_A[sava_association_A[num*count+i].value_x][sava_association_A[num*count+i].value_y] = 0
                temp_count_zero.append(sava_association_A[num*count+i])
            save_all_count_zero_every_time_changed.append(temp_count_zero)
            save_all_count_A.append(temp_A)
            break
        if (count < k - 1):
            x = num *(count)
            for i in range(num):
                temp_A[sava_association_A[i+x].value_x][sava_association_A[i+x].value_y] = 0
                temp_count_zero.append(sava_association_A[num * count + i])
            save_all_count_zero_every_time_changed.append(temp_count_zero)
            save_all_count_A.append(temp_A)
    return save_all_count_A,sava_association_A,save_all_count_zero_not_changed,save_all_count_zero_every_time_changed,num,A_length


def read_data_flies():

    #  240 * 495 lncRNA * miRNA R12
    R12 = np.loadtxt("../data_create/yuguoxian_lnc_mi.txt")  # 1002


    # 240 * 15527 lncRNA * gene R13
    R13 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_lnc_gene.txt")

    # 240 * 6428 lncRNA * GO R14
    R14 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_lnc_go.txt")

    #  240 * 405  lncRNA * diseases R15
    R15 = np.loadtxt("../data_create/lnc_dis_association.txt")  # 2687

    # 495 * 15527 miRNA * gene R23
    R23 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_miRNA_gene.txt")

    #  495 * 405 miRNA * diseases R25
    R25 = np.loadtxt("../data_create/mi_dis.txt")  # 13559

    # 15527 * 15527 gene * gene (ppi) R33
    R33 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_gene_inter1.txt")

    # 15527 * 6428 gene * go (ppi) R34
    R34 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_gene_go.txt")

    # 15527 * 405 gene * disease R35
    R35 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_gene_disease.txt")

    # 8283 * 15527 drug * gene R63
    R63 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_drug_gene.txt")

    # 8283 * 8283 drug * gene R66
    R66 = np.loadtxt("../data_create/data_add_for_yuguoxian/yuguoxian_drug_inter.txt")

    return R12, R13, R14, R15, R23, R25, R33, R34, R35, R63, R66


def getOptimalWeights(Hs,alpha):
    k = len(Hs)
    index = np.argsort(Hs)
    newHs = Hs[index]
    p = k
    bfind = 1
    gamma = 0
    while p>0 and bfind:
        gamma = (np.sum(newHs[:p]) + 2*alpha)/p
        if(gamma-newHs[p-1])>0:
            bfind = 0
        else:
            p = p - 1
    newWs = np.zeros(k)
    for i in range(p):
        newWs[i] = (gamma - newHs[i])/(2*alpha)
    Ws = np.zeros(k)
    Ws[index] = newWs
    return Ws


def MFLDA_init(R15, R12, R13, R14, R23, R25, R33, R34, R35, R63, R66, max_iter):
    k_G1 = 50
    k_G2 = 110
    k_G3 = 50
    k_G4 = 70
    k_G5 = 170
    k_G6 = 50


    max_iter = max_iter
    alpha = 100000
    threshold = 0.00001

    R1 = np.concatenate((R12, R13, R14), axis=1)  # 240 * (495 + 15527)
    R5 = np.concatenate((R25.T, R35.T), axis=1)  # 405 * (495 + 15527)

    # G1 = svd.initial_G(R1, k_G1)  # 240 * 50
    # G2 = svd.initial_G(R25, k_G2)  # 495 * 110
    # G3 = svd.initial_G(R35, k_G3)  # 15527 * 50
    # G4 = svd.initial_G(R14.T, k_G4)  # 6428 * 70
    # G5 = svd.initial_G(R5, k_G5)  # 405 * 170
    # G6 = svd.initial_G(R63, k_G6)  # 8823 * 50
    G1 = np.loadtxt("../data_create/intial_G/G1.txt")  # 240 * 50
    G2 = np.loadtxt("../data_create/intial_G/G2.txt")  # 495 * 110
    G3 = np.loadtxt("../data_create/intial_G/G3.txt")  # 15527 * 50
    G4 = np.loadtxt("../data_create/intial_G/G4.txt")  # 6428 * 70
    G5 = np.loadtxt("../data_create/intial_G/G5.txt")  # 405 * 170
    G6 = np.loadtxt("../data_create/intial_G/G6.txt")  # 8823 * 50

    # print(G2)
    theta_3 = R33 * -1
    theta_6 = R66 * -1

    nTypes = 6
    G = np.array([G1, G2, G3, G4, G5, G6])
    R = np.array([R12, R13, R14, R15, R23, R25, R34, R35, R63])
    # [0,1] [0,2] [0,3] [1,3] [2,3]
    #   1     2     3     4     8
    instanseIdx = [2, 3, 4, 5, 9, 11, 16, 17, 33]

    theta = np.array([theta_3, theta_6])

    # MFLDA_Demo(nTypes, G, R, theta)
    # MFLDA
    LD = R[3]
    NL = np.shape(LD)[0]
    ND = np.shape(LD)[1]

    g1 = np.zeros_like(G1)
    g2 = np.zeros_like(G2)
    g3 = np.zeros_like(G3)
    g4 = np.zeros_like(G4)
    g5 = np.zeros_like(G5)
    g6 = np.zeros_like(G6)

    G_enum = np.array([g1, g2, g3, g4, g5, g6])
    G_denom = np.array([g1, g2, g3, g4, g5, g6])

    s1 = np.zeros((k_G1, k_G2))
    s2 = np.zeros((k_G1, k_G3))
    s3 = np.zeros((k_G1, k_G4))
    s4 = np.zeros((k_G1, k_G5))
    s5 = np.zeros((k_G2, k_G3))
    s6 = np.zeros((k_G2, k_G5))
    s7 = np.zeros((k_G3, k_G4))
    s8 = np.zeros((k_G3, k_G5))
    s9 = np.zeros((k_G6, k_G3))

    Scell = np.array([s1, s2, s3, s4, s5, s6, s7, s8, s9])
    theta_p = np.array([np.zeros_like(R33), np.zeros_like(R66)])
    theta_n = np.array([np.zeros_like(R33), np.zeros_like(R66)])

    for i in range(len(theta)):
        theta_temp = theta[i]
        t = np.abs(theta_temp)
        theta_p[i] = (t + theta_temp) / 2
        theta_n[i] = (t - theta_temp) / 2

    for iter_i in range(max_iter):
        mus = np.zeros(len(R))

        # initialize s
        for rr in range(len(instanseIdx)):
            i = int(instanseIdx[rr] / nTypes)
            j = (instanseIdx[rr] % nTypes) - 1
            # if (j == 0):
            #     j = 3

            Gmatii = G[i]
            Gmatjj = G[j]
            # print(j)


            Rmat = R[rr]

            Smat_left = np.linalg.inv(np.matmul(Gmatii.T, Gmatii))
            Smat_mid = np.matmul(np.matmul(Gmatii.T, Rmat), Gmatjj)
            Smat_right = np.linalg.inv(np.matmul(Gmatjj.T, Gmatjj))
            Smat = np.matmul(np.matmul(Smat_left, Smat_mid), Smat_right)
            Smat[np.isnan(Smat)] = 0
            Scell[rr] = Smat

            result = np.sum(np.square((R[rr] - np.matmul(np.matmul(Gmatii, Scell[rr]), Gmatjj.T))))
            mus[rr] = result

        # optimal weights
        Ws = getOptimalWeights(mus, alpha)

        # update G
        for rr in range(len(instanseIdx)):
            i = int(instanseIdx[rr] / nTypes)
            j = (instanseIdx[rr] % nTypes)-1
            # if (j == 0):
            #     j = 3
            temp1 = np.matmul(np.matmul(R[rr], G[j]), Scell[rr].T)
            temp1[np.isnan(temp1)] = 0
            t = np.abs(temp1)

            temp1p = (t + temp1) / 2
            temp1n = (t - temp1) / 2

            temp2 = np.matmul(np.matmul(np.matmul(Scell[rr], G[j].T), G[j]), Scell[rr].T)
            temp2[np.isnan(temp2)] = 0
            t = np.abs(temp2)

            temp2p = (t + temp2) / 2
            temp2n = (t - temp2) / 2

            temp3 = np.matmul(np.matmul(R[rr].T, G[i]), Scell[rr])
            temp3[np.isnan(temp3)] = 0
            t = np.abs(temp3)
            t[t < 0] = 0
            temp3p = (t + temp3) / 2
            temp3n = (t - temp3) / 2

            temp4 = np.matmul(np.matmul(np.matmul(Scell[rr].T, G[i].T), G[i]), Scell[rr])
            temp4[np.isnan(temp4)] = 0
            t = np.abs(temp4)
            t[t < 0] = 0
            temp4p = (t + temp4) / 2
            temp4n = (t - temp4) / 2

            G_enum[i] = G_enum[i] + Ws[rr] * temp1p + Ws[rr] * np.matmul(G[i], temp2n)
            G_denom[i] = G_denom[i] + Ws[rr] * temp1n + Ws[rr] * np.matmul(G[i], temp2p)

            G_enum[j] = G_enum[j] + Ws[rr] * temp3p + Ws[rr] * np.matmul(G[j], temp4n)
            G_denom[j] = G_denom[j] + Ws[rr] * temp3n + Ws[rr] * np.matmul(G[j], temp4p)


        G_enum[2] = G_enum[2] + np.matmul(theta_n[0], G[2])
        G_denom[2] = G_denom[2] + np.matmul(theta_p[0], G[2])

        G_enum[5] = G_enum[5] + np.matmul(theta_n[1], G[5])
        G_denom[5] = G_denom[5] + np.matmul(theta_p[1], G[5])

        for i in range(len(G)):
            G_denom[i] = G_denom[i] + np.spacing(1)
            factor = np.sqrt((G_enum[i] / G_denom[i]))
            G[i] = G[i] * factor
            G[i][np.isnan(G[i])] = 0
            G[i][np.isinf(G[i])] = 0

        #  compare the target approximation (||R15-G1S15G5'||^2) with threshold
        result = R[3] - np.matmul(np.matmul(G[0], Scell[3]), G[4].T)
        R_threshold = np.sum(np.square(result))
        if R_threshold < threshold:
            break

    newF = np.matmul(np.matmul(G[0], Scell[3]), G[4].T)
    # print(newF)
    return newF


# draw ROC
def draw_roc(RD,matrix_A,association_A,count_k,num,A_length,k,count_zero_every_time_changed):
    temp_matrix_A = matrix_A.copy()
    length_row = RD.shape[0]
    length_column = RD.shape[1]
    # record all data value and location
    disease_drug_A=[]
    for i in range(length_row):
        disease_drug_line=[]
        for j in range(length_column):
            disease_drug=value_index(RD[i][j],i,j)
            disease_drug_line.append(disease_drug)# or extend
        disease_drug_A.append(disease_drug_line)
    # set the one to zero
    #当前 有关系数据不属于最后一个测试集---最后一个测试集的长度可能与之前不一致
    if (count_k < k - 1):
        for x in range(k):
            if(x!=count_k and x< k-1):
                for i in range(num):
                    disease_drug_A[association_A[num*x+i].value_x][association_A[num*x+i].value_y].value = -1
            if (x != count_k and x == k-1):
                for i in range(A_length -num*(k-1)):
                    disease_drug_A[association_A[num * x + i].value_x][association_A[num * x + i].value_y].value = -1
    #当前是最后一个测试集
    if(count_k == k - 1):
        for i in range(A_length - num):
            disease_drug_A[association_A[i].value_x][association_A[i].value_y].value = -1
    # save data not -1
    temp_disease_drug_A = []
    #for disease_drug_line in disease_drug_A:
    #   for disease_drug in disease_drug_line:
    #      if(disease_drug.value != -1):
    #         temp_disease_drug_A.append(disease_drug)
    # every column become a list and there is no number is 1
    for j in range(length_column):
        temp_disease_drug_line = []
        for i in range(length_row):
            if(disease_drug_A[i][j].value != -1):
                temp_disease_drug_line.append(disease_drug_A[i][j])
        temp_disease_drug_A.append(temp_disease_drug_line)
    # count number is 1 of every column
    count_positive_every_column = np.zeros(length_column)
    for j in range(length_column):
        count = 0
        for i in range(len(count_zero_every_time_changed)):
            if(count_zero_every_time_changed[i].value_y == j):
                count += 1
        count_positive_every_column[j] = count
    # sort for every column
    y_all_TPR_temp = []# all column TPR
    x_all_FPR_temp = []
    y_P_all_columns = []
    x_R_all_columns = []
    Rcall_all = []
    AUC_column = []
    PR_column = []
    areaROC = 0.0
    areaPR = 0.0
    # temp_RD = []
    # dis_15 = [7, 9, 10, 61, 62, 69, 113, 126, 140, 156, 211, 233, 316, 334, 335, 338]
    dis_15 = [62, 69, 113, 140, 156, 178, 181, 187, 233, 297]

    test_15_dis_AUC = []
    test_15_dis_PR = []
    for j in range(length_column):
        temp_left_RD = sorted(temp_disease_drug_A[j], key=attrgetter('value'), reverse=True)

        # ROC of every column
        all_number = len(temp_left_RD)

        count_positive_all = count_positive_every_column[j]
        if(count_positive_all ==0):
            continue  # if this column only have 0 continue

        count_negative_all = all_number - count_positive_all
        # compute tp,fn,fp,tn --------first using not sorted data
        R = 0.0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        TPR = 0.0
        FPR = 0.0
        P_temp = []
        R_temp = []
        y_TPR_temp = []
        x_FPR_temp = []
        Rcall_temp = []

        for x in range(all_number):
            if (x == 30 or x == 60 or x == 90 or x == 120 or x == 150 or x == 180 or x == 210 or x ==240 ):
                Rcall_temp.append(R)
            # print(matrix_A[temp_left_RD[x].value_x][temp_left_RD[x].value_y])
            if (matrix_A[temp_left_RD[x].value_x][temp_left_RD[x].value_y] == 1):
                TP += 1
                FN = count_positive_all - TP
                TPR = TP / count_positive_all
                FPR = FP / count_negative_all
                P = TP / (TP + FP)
                R = TP / (count_positive_all)
                P_temp.append(P)
                R_temp.append(R)
                y_TPR_temp.append(TPR)
                x_FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative_all - FP
                TPR = TP / count_positive_all
                FPR = FP / count_negative_all
                P = TP / (TP + FP)
                R = TP / (count_positive_all)
                P_temp.append(P)
                R_temp.append(R)
                y_TPR_temp.append(TPR)
                x_FPR_temp.append(FPR)
        while len(Rcall_temp)<8:
            Rcall_temp.append(1.0)
        Rcall_all.append(Rcall_temp)
        y_P_all_columns.append(P_temp)
        x_R_all_columns.append(R_temp)
        y_all_TPR_temp.append(y_TPR_temp)
        x_all_FPR_temp.append(x_FPR_temp)
        area_column_ROC = area_ROC(y_TPR_temp, x_FPR_temp)
        area_column_PR = area_PR(P_temp, R_temp)
        AUC_column.extend([area_column_ROC])
        PR_column.extend([area_column_PR])
        if j in dis_15:
            test_15_dis_AUC.extend([area_column_ROC])
            test_15_dis_PR.extend([area_column_PR])
        areaROC += area_column_ROC
        areaPR += area_column_PR

    areaROC = areaROC / len(y_all_TPR_temp)
    areaPR = areaPR / len(y_P_all_columns)
    return y_all_TPR_temp, x_all_FPR_temp,y_P_all_columns,x_R_all_columns,areaROC,areaPR,Rcall_all,AUC_column,PR_column,test_15_dis_AUC,test_15_dis_PR


def draw_roc_test(RD,matrix_A,association_A,count_k,num,A_length,k,count_zero_every_time_changed):
    temp_matrix_A = matrix_A.copy()
    length_row = RD.shape[0]
    length_column = RD.shape[1]
    # record all data value and location
    disease_drug_A=[]
    for i in range(length_row):
        disease_drug_line=[]
        for j in range(length_column):
            disease_drug=value_index(RD[i][j],i,j)
            disease_drug_line.append(disease_drug)# or extend
        disease_drug_A.append(disease_drug_line)
    # set the one to zero
    #当前 有关系数据不属于最后一个测试集---最后一个测试集的长度可能与之前不一致
    if (count_k < k - 1):
        for x in range(k):
            if(x!=count_k and x< k-1):
                for i in range(num):
                    disease_drug_A[association_A[num*x+i].value_x][association_A[num*x+i].value_y].value = -1
            if (x != count_k and x == k-1):
                for i in range(A_length -num*(k-1)):
                    disease_drug_A[association_A[num * x + i].value_x][association_A[num * x + i].value_y].value = -1
    #当前是最后一个测试集
    if(count_k == k - 1):
        for i in range(A_length - num):
            disease_drug_A[association_A[i].value_x][association_A[i].value_y].value = -1
    # save data not -1
    temp_disease_drug_A = []
    #for disease_drug_line in disease_drug_A:
    #   for disease_drug in disease_drug_line:
    #      if(disease_drug.value != -1):
    #         temp_disease_drug_A.append(disease_drug)
    # every column become a list and there is no number is 1
    for j in range(length_column):
        temp_disease_drug_line = []
        for i in range(length_row):
            if(disease_drug_A[i][j].value != -1):
                temp_disease_drug_line.append(disease_drug_A[i][j])
        temp_disease_drug_A.append(temp_disease_drug_line)
    # count number is 1 of every column
    count_positive_every_column = np.zeros(length_column)
    for j in range(length_column):
        count = 0
        for i in range(len(count_zero_every_time_changed)):
            if(count_zero_every_time_changed[i].value_y == j):
                count += 1
        count_positive_every_column[j] = count
    # sort for every column

    areaROC = 0.0
    areaPR = 0.0
    # temp_RD = []
    Rcall_all = []

    AUC_column = []
    PR_column = []

    for j in range(length_column):
        temp_left_RD = sorted(temp_disease_drug_A[j], key=attrgetter('value'), reverse=True)

        # ROC of every column
        all_number = len(temp_left_RD)

        count_positive_all = count_positive_every_column[j]
        if(count_positive_all ==0):
            continue  # if this column only have 0 continue

        count_negative_all = all_number - count_positive_all
        # compute tp,fn,fp,tn --------first using not sorted data
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        TPR = 0.0
        FPR = 0.0
        R = 0.0
        P_temp = []
        R_temp = []
        y_TPR_temp = []
        x_FPR_temp = []
        Rcall_temp = []

        for x in range(all_number):
            # print(matrix_A[temp_left_RD[x].value_x][temp_left_RD[x].value_y])
            if (matrix_A[temp_left_RD[x].value_x][temp_left_RD[x].value_y] == 1):
                TP += 1
                FN = count_positive_all - TP
                TPR = TP / count_positive_all
                FPR = FP / count_negative_all
                P = TP / (TP+FP)
                R = TP / (count_positive_all)
                P_temp.append(P)
                R_temp.append(R)
                y_TPR_temp.append(TPR)
                x_FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative_all - FP
                TPR = TP / count_positive_all
                FPR = FP / count_negative_all
                P = TP / (TP + FP)
                R = TP / (count_positive_all)
                P_temp.append(P)
                R_temp.append(R)
                y_TPR_temp.append(TPR)
                x_FPR_temp.append(FPR)


        area_column_ROC = area_ROC(y_TPR_temp, x_FPR_temp)
        area_column_PR = area_PR(P_temp, R_temp)
        AUC_column.extend([area_column_ROC])
        PR_column.extend([area_column_PR])



    return AUC_column,PR_column

# get same point of TPR,FPR in columns,not contain all 0 column
def gsp(y_all_TPR_temp, x_all_FPR_temp,y_all_P_temp,x_all_R_temp):
    # find the min number in the column
    numbers = []
    min_number = len(y_all_TPR_temp[0])
    for i in range(len(y_all_TPR_temp)):
        number = len(y_all_TPR_temp[i])
        numbers.extend([number])
        if min_number > number:
            min_number = number
    y_all_TPR = []
    x_all_FPR = []
    y_all_P = []
    x_all_R = []
    for i in range(len(y_all_TPR_temp)):
        y_TPR = []
        x_FPR = []
        y_P = []
        x_R = []
        current_length = numbers[i]
        for j in range(min_number-1):
            division_result = current_length / (min_number-1)
            y_TPR.append(y_all_TPR_temp[i][round(division_result*(j))])
            x_FPR.append(x_all_FPR_temp[i][round(division_result*(j))])
            y_P.append(y_all_P_temp[i][round(division_result*(j))])
            x_R.append(x_all_R_temp[i][round(division_result*(j))])

        y_TPR.append(y_all_TPR_temp[i][current_length-1])
        x_FPR.append(x_all_FPR_temp[i][current_length-1])
        y_P.append(y_all_P_temp[i][current_length-1])
        x_R.append(x_all_R_temp[i][current_length-1])
        y_all_TPR.append(y_TPR)
        x_all_FPR.append(x_FPR)
        # print(y_all_TPR[i][0])
        # print(x_all_FPR[i][0])
        y_all_P.append(y_P)
        x_all_R.append(x_R)
    #print("---4.get TPR FPR---")
    return x_all_FPR,y_all_TPR,x_all_R,y_all_P


def get_all_k_Curve(X_all_k_FPR,Y_all_k_TPR,X_all_k_R,Y_all_k_P,k):
    numbers = []
    min_number = len(Y_all_k_TPR[0])
    for i in range(len(Y_all_k_TPR)):
        number = len(Y_all_k_TPR[i])
        numbers.extend([number])
        if min_number > number:
            min_number = number
    y_all_TPR = []
    x_all_FPR = []
    y_all_P = []
    x_all_R = []
    for i in range(len(Y_all_k_TPR)):
        y_TPR = []
        x_FPR = []
        y_P = []
        x_R = []
        current_length = numbers[i]
        for j in range(min_number - 1):
            division_result = current_length / (min_number - 1)
            y_TPR.append(Y_all_k_TPR[i][round(division_result * (j))])
            x_FPR.append(X_all_k_FPR[i][round(division_result * (j))])
            y_P.append(Y_all_k_P[i][round(division_result * (j))])
            x_R.append(X_all_k_R[i][round(division_result * (j))])

        y_TPR.append(Y_all_k_TPR[i][current_length - 1])
        x_FPR.append(X_all_k_FPR[i][current_length - 1])
        y_P.append(Y_all_k_P[i][current_length - 1])
        x_R.append(X_all_k_R[i][current_length - 1])
        y_all_TPR.append(y_TPR)
        x_all_FPR.append(x_FPR)
        # print(y_all_TPR[i][0])
        # print(x_all_FPR[i][0])
        y_all_P.append(y_P)
        x_all_R.append(x_R)

    FPR = np.sum(x_all_FPR, axis=0) / k
    TPR = np.sum(y_all_TPR, axis=0) / k
    R = np.sum(x_all_R, axis=0) / k
    P = np.sum(y_all_P, axis=0) / k
    # plt.plot(FPR, TPR)
    # plt.plot(R, P)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.show()
    return FPR, TPR, R, P


def get_current_AUC_ROC(model,A,sava_association_A,i,num, A_length, k,count_zero_changed):


    # according to different column
    # Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns, areaROC, areaPR = draw_roc_column(clf1, clf2, clf3, clf4, clf5,columns, all_column_test_data,lnc_sim,dis_sim)
    y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns, areaROC, areaPR,Rcall_all,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR = draw_roc(model, A,sava_association_A, i,num, A_length, k,count_zero_changed)
    x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns)
    # x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns)
    # print(len(x_all_FPR_temp))
    # print(len(y_all_TPR_temp))
    # print(len(x_R_all_columns))
    # print(len(y_P_all_columns))

    Rcall = np.sum(Rcall_all,axis=0) /len(Rcall_all)

    x_FPR = np.sum(x_all_FPR, axis=0) / len(x_all_FPR_temp)
    y_TPR = np.sum(y_all_TPR, axis=0) / len(y_all_TPR_temp)
    x_R = np.sum(x_all_R, axis=0) / len(x_R_all_columns)
    y_P = np.sum(y_all_P, axis=0) / len(y_P_all_columns)
    #print('Time:{}, AUC:{}'.format(time, average_column_area))
    #plt.plot(x_FPR, y_TPR)
    #plt.show()
    return x_FPR,y_TPR,x_R,y_P,areaROC, areaPR,Rcall,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR

def area_ROC(TPR,FPR):
    point_number = len(TPR)  # Y_FPR is the same with X_TPR
    area = 0.0
    for i in range(point_number-1):
        area += (FPR[i+1] - FPR[i]) * TPR[i+1]
    return area


def area_PR(P,R):
    point_number = len(P)  # Y_FPR is the same with X_TPR
    area = P[0] * R[0]
    for i in range(point_number-1):
        area += (R[i+1] - R[i]) * P[i]
    return area


def test(k):
    max_iter = 5
    R12, R13, R14, R15, R23, R25, R33, R34, R35, R63, R66 = read_data_flies()
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
        R15, k)
    all_area = []
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    for i in range(k):
        newA = MFLDA_init(save_all_count_A[i], R12, R13, R14, R23, R25, R33, R34, R35, R63, R66, max_iter)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(newA,R15, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # x_all_FPR, y_all_TPR = gsp(y_every_column_TPR, x_every_column_FPR)

        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(newA, R15, sava_association_A, i, num,
                                                                      A_length, k,
                                                                      save_all_count_zero_every_time_changed[i])
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR

        print("该次面积", areaROC, areaPR)

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print("平均roc面积", a_a_1, a_a_2)
    get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)


def test2(R15,save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length):
    k = 5
    max_iter = 5
    R12, R13, R14, _, R23, R25, R33, R34, R35, R63, R66 = read_data_flies()
    # save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
    #     R15, k)
    all_area = []
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    Rcall_all = []

    for i in range(k):
        newA = MFLDA_init(save_all_count_A[i], R12, R13, R14, R23, R25, R33, R34, R35, R63, R66, max_iter)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(newA,R15, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # x_all_FPR, y_all_TPR = gsp(y_every_column_TPR, x_every_column_FPR)

        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR, Rcall = get_current_AUC_ROC(newA, R15, sava_association_A, i, num,
                                                                      A_length, k,
                                                                      save_all_count_zero_every_time_changed[i])
        Rcall_all.append(Rcall)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR

        # print("该次面积", areaROC, areaPR)
    Rcall_last = np.sum(Rcall_all,axis=0) /k

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print("平均roc面积", a_a_1, a_a_2)
    FPR, TPR, R, P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    return FPR, TPR, R, P,a_a_1,a_a_2,Rcall_last


def test2_test(R15,save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length):
    k = 5
    max_iter = 5
    R12, R13, R14, _, R23, R25, R33, R34, R35, R63, R66 = read_data_flies()
    # save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
    #     R15, k)
    all_area = []
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    Rcall_all = []
    AUC_all_column = []
    PR_all_column = []
    dis_15_AUC = []
    dis_15_PR = []

    for i in range(k):
        newA = MFLDA_init(save_all_count_A[i], R12, R13, R14, R23, R25, R33, R34, R35, R63, R66, max_iter)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(newA,R15, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # x_all_FPR, y_all_TPR = gsp(y_every_column_TPR, x_every_column_FPR)

        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR, Rcall,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR = get_current_AUC_ROC(newA, R15, sava_association_A, i, num,
                                                                      A_length, k,
                                                                      save_all_count_zero_every_time_changed[i])
        Rcall_all.append(Rcall)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR

        # print("该次面积", areaROC, areaPR)
        AUC_all_column.extend(AUC_column)
        PR_all_column.extend(PR_column)
        dis_15_AUC.append(test_15_dis_AUC)
        dis_15_PR.append(test_15_dis_PR)
    Rcall_last = np.sum(Rcall_all,axis=0) /k

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print("平均roc面积", a_a_1, a_a_2)
    FPR, TPR, R, P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    return FPR, TPR, R, P,a_a_1,a_a_2,Rcall_last,AUC_all_column, PR_all_column,dis_15_AUC,dis_15_PR


def test(k):
    max_iter = 5
    R12, R13, R14, R15, R23, R25, R33, R34, R35, R63, R66 = read_data_flies()
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
        R15, k)
    all_area = []
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    for i in range(k):
        newA = MFLDA_init(save_all_count_A[i], R12, R13, R14, R23, R25, R33, R34, R35, R63, R66, max_iter)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(newA,R15, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # x_all_FPR, y_all_TPR = gsp(y_every_column_TPR, x_every_column_FPR)

        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(newA, R15, sava_association_A, i, num,
                                                                      A_length, k,
                                                                      save_all_count_zero_every_time_changed[i])
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR

        print("该次面积", areaROC, areaPR)

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print("平均roc面积", a_a_1, a_a_2)
    get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)


def test5_test(R15,save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length):
    k = 5
    max_iter = 5
    R12, R13, R14, _, R23, R25, R33, R34, R35, R63, R66 = read_data_flies()
    # save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
    #     R15, k)
    all_area = []
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    Rcall_all = []
    AUC_all_column = []
    PR_all_column = []
    for i in range(k):
        newA = MFLDA_init(save_all_count_A[i], R12, R13, R14, R23, R25, R33, R34, R35, R63, R66, max_iter)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(newA,R15, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # x_all_FPR, y_all_TPR = gsp(y_every_column_TPR, x_every_column_FPR)


        AUC_column, PR_column = draw_roc_test(newA, R15, sava_association_A, i, num, A_length, k,
                                              save_all_count_zero_every_time_changed[i])

        AUC_all_column.extend(AUC_column)
        PR_all_column.extend(PR_column)

    return AUC_all_column, PR_all_column


if __name__ == '__main__':
    k = 5
    test(k)





























































