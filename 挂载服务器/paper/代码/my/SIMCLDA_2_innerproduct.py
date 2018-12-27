import random
import numpy as np
import copy
from operator import attrgetter
import matplotlib.pyplot as plt
random.seed(1)


def read_data_flies():

    #  240 * 405  lncRNA * diseases
    A = np.loadtxt("../data_create/lnc_dis_association.txt")  # 2687

    #  405 * 405 diseases * diseases
    Sd = np.loadtxt("../data_create/dis_sim_matrix_process.txt")

    #  240 * 240  lncRNA * lncRNA
    # Sm = np.loadtxt("../data_create/lnc_Sim.txt")

    #  240 * 495 lncRNA * miRNA
    Lm = np.loadtxt("../data_create/yuguoxian_lnc_mi.txt")  # 1002

    #  495 * 405 miRNA * diseases
    Md = np.loadtxt("../data_create/mi_dis.txt")  # 13559
    #print("---1.read data files---")
    return A,Sd,Lm,Md

def lncRNA_sim_gKernel(A):
    length_lnc = len(A)
    length_dis = len(A[0])


    sum0 = 0
    for i in range(length_lnc):
        sum0 += np.sum(np.square((A[i])))
    gamal = sum0/length_lnc
    # print(gamal)
    lnc_sim = np.zeros((length_lnc,length_lnc))
    for i in range(length_lnc):
        for j in range(length_lnc):
            lnc_sim[i][j] = np.exp((np.sum(np.square(A[i] - A[j])))/(-gamal))
    return lnc_sim


def computer_inter_for_new(A,lnc_sim):
    length_lnc = len(A)
    length_dis = len(A[0])

    for i in range(length_lnc):
        if((A[i]==0).all()):
            rowVec = lnc_sim[i]
            rowVec[i] = 0

            simNeighbors = np.where(rowVec>np.mean(lnc_sim))
            if (np.shape(simNeighbors)[1]):
                new_row = np.zeros((1,length_dis))
                for j in range(len(simNeighbors)):
                    new_row += A[simNeighbors[0][j],:]
                new_row /= len(simNeighbors)
                A[i] = new_row
    return A


def pca_energy(simMat,para):
    u, s, vh = np.linalg.svd(simMat)
    # print(s)
    rank_length = np.linalg.matrix_rank(simMat)
    pca_rank = 0
    # print(rank_length)
    for i in range(rank_length):
        if(np.sum(s[0:i])>(para*sum(s))):
            pca_rank = i
            break
    # print(pca_rank)
    u = u[:,:pca_rank]
    return u


def Z_upadte(lamda,s,omega_liner,L,Y,D,A):
    z_right_mid = np.matmul(np.matmul(L,Y),D.T) - A

    z_right_mid = z_right_mid * A

    z_right = Y - np.matmul(np.matmul(L.T,z_right_mid),D)/s
    # print(z_right)
    k = np.linalg.matrix_rank(z_right)
    # print(k)
    L_svd, S_svd, T_svd = np.linalg.svd(z_right)
    # print(np.shape(L_svd))
    S_svd = np.diag(np.maximum(0,S_svd-lamda/s))
    L_svd = L_svd[:,:k]
    S_svd = S_svd[:k, :k]
    T_svd = T_svd[:k,:]
    # print(np.shape(L_svd))
    # print(np.shape(S_svd))
    # print(np.shape(T_svd))
    retVal = np.matmul(np.matmul(L_svd,S_svd),T_svd)

    return retVal


def deta_matrix(L,matrix,D,A,Omega_linear):
    f = np.matmul(np.matmul(L, matrix), D.T) - A
    f = f * A
    return f


def sideobjectCalc(Z,lamda,diffl):
    _, s, _ = np.linalg.svd(Z)
    retValue = lamda * np.sum(s)
    retValue = retValue + diffl
    return retValue


def SIMC(A,Omega_linear,L,D):
    lambda_simc = 1
    max_iter = 1000
    r_a = np.shape(L)[1]
    r_b = np.shape(D)[1]
    s = 1
    gamma = 2
    Z0 = np.zeros((r_a,r_b))
    Z1 = np.zeros((r_a,r_b))
    # Z = [Z0,Z1]

    # Y0 = np.zeros((r_a, r_b))
    # Y1 = np.zeros((r_a, r_b))
    # Y = [Y0, Y1]

    theta0 = 1
    theta1 = 1
    i = 0
    convergence = np.zeros(max_iter)

    while i < max_iter:

        Y1 = Z1 + theta1 * (1 / theta0 - 1) * (Z1 - Z0)
        Z0 = Z1
        Z1 = Z_upadte(lambda_simc,s,Omega_linear,L,Y1,D,A)

        deta_f_Z = deta_matrix(L, Z1, D, A, Omega_linear)
        f_Z = np.square(np.linalg.norm(deta_f_Z,'fro'))/2

        deta_f_Y = deta_matrix(L, Y1, D, A, Omega_linear)
        deta_f_Y_1 = np.matmul(np.matmul(L.T, deta_f_Y), D)
        f_Z_Y_left_mid = Y1 -  deta_f_Y_1/ s
        f_Z_Y_left = np.square(np.linalg.norm((Z1 - f_Z_Y_left_mid) ,'fro'))*s/2
        f_Z_Y_mid = np.square(np.linalg.norm(deta_f_Y,'fro'))/2
        f_Z_Y_rigth = np.square(np.linalg.norm(deta_f_Y_1,'fro'))/2/s

        f_Z_Y = f_Z_Y_left + f_Z_Y_mid + f_Z_Y_rigth

        while f_Z >= f_Z_Y:
            s = s * gamma
            Z1 = Z_upadte(lambda_simc, s, Omega_linear, L, Y1, D, A)

            deta_f_Z = deta_matrix(L, Z1, D, A, Omega_linear)
            f_Z = np.square(np.linalg.norm(deta_f_Z, 'fro')) / 2
            deta_f_Y = deta_matrix(L, Y1, D, A, Omega_linear)
            deta_f_Y_1 = np.matmul(np.matmul(L.T, deta_f_Y), D)
            f_Z_Y_left_mid = Y1 - deta_f_Y_1 / s
            f_Z_Y_left = np.square(np.linalg.norm((Z1 - f_Z_Y_left_mid), 'fro')) * s / 2
            f_Z_Y_mid = np.square(np.linalg.norm(deta_f_Y, 'fro')) / 2
            f_Z_Y_rigth = np.square(np.linalg.norm(deta_f_Y_1, 'fro')) / 2 / s
            f_Z_Y = f_Z_Y_left + f_Z_Y_mid + f_Z_Y_rigth

        theta0 = theta1
        theta1 = (np.sqrt(np.power(theta1,4)+4*np.power(theta1,2))-np.square(theta1))/2
        convergence[i] = sideobjectCalc(Z1,lambda_simc,f_Z)
        if i>1:
            if abs(convergence[i] - convergence[i - 1]) < (1e-5) * convergence[i]:
                break
        i = i + 1

    M_recover = np.matmul(np.matmul(L, Z1), D.T)
    return M_recover


def area_PR(P,R):
    point_number = len(P)  # Y_FPR is the same with X_TPR
    area = P[0] * R[0]
    for i in range(point_number-1):
        area += (R[i+1] - R[i]) * P[i]
    return area


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
    AUC_column = []
    PR_column = []
    areaROC = 0.0
    areaPR = 0.0
    # temp_RD = []
    Rcall_all = []
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
            if (x == 30 or x == 60 or x == 90 or x == 120 or x == 150 or x == 180 or x == 210 or x ==240 ):
                Rcall_temp.append(R)
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
    return y_all_TPR_temp, x_all_FPR_temp,y_P_all_columns,x_R_all_columns,areaROC,areaPR,Rcall_all,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR


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
    return FPR,TPR,R,P


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

    Rcall = np.sum(Rcall_all, axis=0) / len(Rcall_all)
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


def test():


    # print(lnc_sim)

    k = 5
    # ld = np.loadtxt("./SIMCLDA_data/Dataset1/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset1/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset1/lncSim.txt")
    # print(np.shape(lnc_sim_1))
    # ld = np.loadtxt("./SIMCLDA_data/Dataset2/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset2/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset2/lncSim.txt")
    ld, dis_sim, _, _ = read_data_flies()
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(ld,k)
    all_area = []
    for i in range(k):


        lnc_sim = lncRNA_sim_gKernel(save_all_count_A[i])
        # print(np.shape(lnc_sim))
        ld_new = computer_inter_for_new(save_all_count_A[i], lnc_sim)

        lnc_feature = pca_energy(lnc_sim, 0.8)
        dis_feature = pca_energy(dis_sim, 0.6)

        omega = np.where(ld_new == 1)
        # print(omega)
        M_recover = SIMC(ld_new, omega, lnc_feature, dis_feature)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns, areaROC, areaPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i])


        # x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(M_recover,ld, sava_association_A, i, num, A_length, k,
                                                          save_all_count_zero_every_time_changed[i])
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR

        print("该次面积", areaROC,areaPR)

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print("平均roc面积", a_a_1,a_a_2)
    get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)




def test2(ld, dis_sim,save_all_count_A, sava_association_A, save_all_count_zero_every_time_changed, num, A_length):


    # print(lnc_sim)

    k = 5
    # ld = np.loadtxt("./SIMCLDA_data/Dataset1/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset1/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset1/lncSim.txt")
    # print(np.shape(lnc_sim_1))
    # ld = np.loadtxt("./SIMCLDA_data/Dataset2/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset2/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset2/lncSim.txt")
    # ld, dis_sim, _, _ = read_data_flies()
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    Rcall_all = []
    # save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(ld,k)
    all_area = []
    for i in range(k):


        lnc_sim = lncRNA_sim_gKernel(save_all_count_A[i])
        # print(np.shape(lnc_sim))
        ld_new = computer_inter_for_new(save_all_count_A[i], lnc_sim)

        lnc_feature = pca_energy(lnc_sim, 0.8)
        dis_feature = pca_energy(dis_sim, 0.6)

        omega = np.where(ld_new == 1)
        # print(omega)
        M_recover = SIMC(ld_new, omega, lnc_feature, dis_feature)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns, areaROC, areaPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i])


        # x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR, Rcall = get_current_AUC_ROC(M_recover,ld, sava_association_A, i, num, A_length, k,
                                                          save_all_count_zero_every_time_changed[i])
        Rcall_all.append(Rcall)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR

        # print("该次面积", areaROC,areaPR)
    Rcall_last = np.sum(Rcall_all, axis=0) / k
    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print("平均roc面积", a_a_1,a_a_2)
    FPR,TPR,R,P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    return FPR,TPR,R,P,a_a_1,a_a_2,Rcall_last



def test_top():


    # print(lnc_sim)

    k = 5
    # ld = np.loadtxt("./SIMCLDA_data/Dataset1/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset1/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset1/lncSim.txt")
    # print(np.shape(lnc_sim_1))
    # ld = np.loadtxt("./SIMCLDA_data/Dataset2/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset2/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset2/lncSim.txt")
    ld, dis_sim, _, _ = read_data_flies()
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    R_all = []
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(ld,k)
    all_area = []
    for i in range(k):


        lnc_sim = lncRNA_sim_gKernel(save_all_count_A[i])
        # print(np.shape(lnc_sim))
        ld_new = computer_inter_for_new(save_all_count_A[i], lnc_sim)

        lnc_feature = pca_energy(lnc_sim, 0.8)
        dis_feature = pca_energy(dis_sim, 0.6)

        omega = np.where(ld_new == 1)
        # print(omega)
        M_recover = SIMC(ld_new, omega, lnc_feature, dis_feature)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns, areaROC, areaPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i])


        # x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns)
        # x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i])
        R_call = preict_top(M_recover,ld,sava_association_A,i,num,A_length,k)
        print(R_call)
        R_all.append(R_call)


    R = np.sum(R_all, axis=0) / 5
    print(R)


def test2_test(ld, dis_sim,save_all_count_A, sava_association_A, save_all_count_zero_every_time_changed, num, A_length):


    # print(lnc_sim)

    k = 5
    # ld = np.loadtxt("./SIMCLDA_data/Dataset1/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset1/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset1/lncSim.txt")
    # print(np.shape(lnc_sim_1))
    # ld = np.loadtxt("./SIMCLDA_data/Dataset2/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/Dataset2/dissim_Jaccard.txt")
    # lnc_sim_1 = np.loadtxt("./SIMCLDA_data/Dataset2/lncSim.txt")
    # ld, dis_sim, _, _ = read_data_flies()
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
    # save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(ld,k)
    all_area = []
    for i in range(k):


        lnc_sim = lncRNA_sim_gKernel(save_all_count_A[i])
        # print(np.shape(lnc_sim))
        ld_new = computer_inter_for_new(save_all_count_A[i], lnc_sim)

        lnc_feature = pca_energy(lnc_sim, 0.8)
        dis_feature = pca_energy(dis_sim, 0.6)

        omega = np.where(ld_new == 1)
        # print(omega)
        M_recover = SIMC(ld_new, omega, lnc_feature, dis_feature)

        pre_area = 0.0
        # y_every_column_TPR, x_every_column_FPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i],)
        # y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns, areaROC, areaPR = draw_roc(M_recover,ld, sava_association_A, i, num, A_length, k,
        #                                                   save_all_count_zero_every_time_changed[i])


        # x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(y_all_TPR_temp, x_all_FPR_temp, y_P_all_columns, x_R_all_columns)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR, Rcall,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR = get_current_AUC_ROC(M_recover,ld, sava_association_A, i, num, A_length, k,
                                                          save_all_count_zero_every_time_changed[i])
        Rcall_all.append(Rcall)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR
        AUC_all_column.extend(AUC_column)
        PR_all_column.extend(PR_column)
        dis_15_AUC.append(test_15_dis_AUC)
        dis_15_PR.append(test_15_dis_PR)

        # print("该次面积", areaROC,areaPR)
    Rcall_last = np.sum(Rcall_all, axis=0) / k
    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print("平均roc面积", a_a_1,a_a_2)
    FPR,TPR,R,P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    return FPR,TPR,R,P,a_a_1,a_a_2,Rcall_last,AUC_all_column, PR_all_column,dis_15_AUC,dis_15_PR




if __name__ == '__main__':
    # ld = np.loadtxt("./SIMCLDA_data/interMatrix.txt")
    # dis_sim = np.loadtxt("./SIMCLDA_data/dissim_Jaccard.txt")
    # lnc_sim = lncRNA_sim_gKernel(ld)
    # # print(lnc_sim)
    #
    # ld = computer_inter_for_new(ld, lnc_sim)
    #
    # lnc_feature = pca_energy(lnc_sim,0.8)
    # dis_feature = pca_energy(dis_sim,0.6)
    #
    # omega = np.where(ld==1)
    # # print(omega)
    # M_recover = SIMC(ld, omega, lnc_feature, dis_feature)
    # test()
    test_top()












































