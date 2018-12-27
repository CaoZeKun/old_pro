# BI-RandomWalk
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from operator import attrgetter
import copy
random.seed(9)


# read data file
def read_data_flies():
    # get diseases
    DSim = np.loadtxt("./SD_5cross_tab.txt")
    simD = np.array(DSim)  # -----it is MD   326 * 326  diseases * diseases
    DsimD = np.eye(simD.shape[0])
    # print(DsimD)
    MD = np.zeros((simD.shape[0], simD.shape[1]))
    for i in range(simD.shape[0]):
        DsimD[i, i] = np.sum(simD[i])
    for i in range(simD.shape[0]):
        for j in range(simD.shape[1]):
            MD[i][j] = simD[i][j] / math.sqrt(DsimD[i][i] * DsimD[j][j])
    # get miRNAs
    RSim = np.loadtxt("./SM_all_5cross_tab.txt")
    simR = np.array(RSim)  # -----it is MR 490 * 490  miRNA * miRNA
    DsimR = np.eye(simR.shape[0])
    MR = np.zeros((simR.shape[0], simR.shape[1]))
    for i in range(simR.shape[0]):
        DsimR[i, i] = np.sum(simR[i])
    for i in range(simR.shape[0]):
        for j in range(simR.shape[1]):
            MR[i][j] = simR[i][j] / math.sqrt(DsimR[i][i] * DsimR[j][j])
    # get associations
    RDsim = np.loadtxt("./A_5cross_All_space.txt")
    A = np.array(RDsim)  # -----it is A   490 * 326  miRNA * diseases

    return A,MR,MD




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

def data_fusion(A,Sm,Sd):
    pre_matrix_1 = np.concatenate((Sm,A),axis=1)  # (490, 816)
    pre_matrix_2 = np.concatenate((A.T, Sd), axis=1)  # (326, 816)
    pre_matrix = np.concatenate((pre_matrix_1,pre_matrix_2),axis=0)  # (816, 816)
    #print("---2.data fusion---")
    return pre_matrix



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
    # temp_RD = []
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
        x_TPR_temp = []
        y_FPR_temp = []
        for x in range(all_number):
            # print(matrix_A[temp_left_RD[x].value_x][temp_left_RD[x].value_y])
            if (matrix_A[temp_left_RD[x].value_x][temp_left_RD[x].value_y] == 1):
                TP += 1
                FN = count_positive_all - TP
                TPR = TP / count_positive_all
                FPR = FP / count_negative_all
                x_TPR_temp.append(TPR)
                y_FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative_all - FP
                TPR = TP / count_positive_all
                FPR = FP / count_negative_all
                x_TPR_temp.append(TPR)
                y_FPR_temp.append(FPR)
        y_all_TPR_temp.append(x_TPR_temp)
        x_all_FPR_temp.append(y_FPR_temp)
    return y_all_TPR_temp, x_all_FPR_temp


# get same point of TPR,FPR in columns,not contain all 0 column
def gsp1(y_all_TPR_temp, x_all_FPR_temp):
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
    for i in range(len(y_all_TPR_temp)):
        y_TPR = []
        x_FPR = []
        for j in range(min_number-1):
            current_length = numbers[i]
            division_result = current_length / (min_number-1)
            y_TPR.append(y_all_TPR_temp[i][round(division_result*(j))])
            x_FPR.append(x_all_FPR_temp[i][round(division_result*(j))])
        y_TPR.append(y_all_TPR_temp[i][current_length-1])
        x_FPR.append(x_all_FPR_temp[i][current_length-1])
        y_all_TPR.append(y_TPR)
        x_all_FPR.append(x_FPR)
    #print("---4.get TPR FPR---")
    return x_all_FPR,y_all_TPR


def gsp(y_all_TPR_temp, x_all_FPR_temp):
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
    for i in range(len(y_all_TPR_temp)):
        y_TPR = []
        x_FPR = []
        for j in range(min_number):
            current_length = numbers[i]
            division_result = current_length / (min_number)
            y_TPR.append(y_all_TPR_temp[i][round(division_result*(j+1))-1])
            x_FPR.append(x_all_FPR_temp[i][round(division_result*(j+1))-1])
        #y_TPR.append(y_all_TPR_temp[i][current_length-1])
        #x_FPR.append(x_all_FPR_temp[i][current_length-1])
        y_all_TPR.append(y_TPR)
        x_all_FPR.append(x_FPR)
    #print("---4.get TPR FPR---")
    return x_all_FPR,y_all_TPR

def area_ROC(TPR,FPR):
    point_number = len(TPR)  # Y_FPR is the same with X_TPR
    area = 0.0
    for i in range(point_number-1):
        area += (FPR[i+1] - FPR[i]) * TPR[i+1]
    return area


# get k time pt
def get_k_pt(alpha,k,l,r):
    k_pt = []
    A, Sm, Sd = read_data_flies()
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
        A, k)
    all_area = []
    save_all_x_TPR = []
    save_all_y_FPR = []
    for i in range(len(save_all_count_A)):
        RD_Zero = save_all_count_A[i] / np.sum(save_all_count_A[i])
        RD = BiRandom_Walk(Sm, Sd, RD_Zero, RD_Zero, l, r, alpha)
        pt = data_fusion(RD,Sm,Sd)
        # pre_matrix = data_fusion(save_all_count_A[i], Sm, Sd)
        # normal_matrix = raw_norm(pre_matrix)
        # p0 = np.eye(len(normal_matrix[0]))
        # pt = BiRandom_Walk(normal_matrix, alpha, p0, max_count)
        # print(np.sum(pt,axis=0))
        k_pt.append(pt)
    return A, Sm, Sd,save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length,k_pt

# test
# def test1(alpha,k,max_count):
#     A, Sm, Sd = read_data_flies()
#     save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
#         A, k)
#     all_area = []
#     save_all_x_TPR = []
#     save_all_y_FPR = []
#     for i in range(len(save_all_count_A)):
#         pre_matrix = data_fusion(save_all_count_A[i], Sm, Sd)
#         normal_matrix = raw_norm(pre_matrix)
#         p0 = np.eye(len(normal_matrix[0]))
#         pt = BiRandom_Walk(normal_matrix, alpha, p0, max_count)
#         A_in_pt = pt[:490,490:]
#         #print(np.shape(A_in_pt))
#         save_every_column_TPR = []
#         save_every_column_FPR = []
#
#         pre_area = 0.0
#         y_every_column_TPR, x_every_column_FPR = draw_roc(A_in_pt, A, sava_association_A, i, num, A_length, k,
#                                                           save_all_count_zero_every_time_changed[i])
#         y_length = len(y_every_column_TPR)
#
#         for j in range(y_length):
#             pre_area += area_ROC(y_every_column_TPR[j], x_every_column_FPR[j])
#         #all_area += pre_area
#         save_every_column_TPR.append(np.sum(y_every_column_TPR, axis=0))
#         save_every_column_FPR.append(np.sum(x_every_column_FPR, axis=0))
#         #print("该组最终roc面积", pre_area / y_length)
#         all_area.append([pre_area / y_length])
#         # print(np.shape(save_every_column_TPR))
#         save_all_x_TPR.extend(save_every_column_TPR)
#         save_all_y_FPR.extend(save_every_column_FPR)
#         # print(np.shape(save_all_x_TPR[2]))
#     print("平均roc面积", np.mean(all_area))
#     #print(np.shape(save_all_y_FPR))
#     y_FPR_average = (np.sum(save_all_y_FPR, axis=0))
#     x_TPR_average = (np.sum(save_all_x_TPR, axis=0))
#
#     # plt.plot(X_TPR_average, Y_FPR_average)
#     # plt.show()
#     #print("---5.program over---")


# test
def test2(alpha, k, l,r):
    A, Sm, Sd = read_data_flies()
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
        A, k)
    all_area = []
    save_all_y_TPR = []
    save_all_x_FPR = []
    for i in range(k):
        RD_Zero = save_all_count_A[i] / np.sum(save_all_count_A[i])
        RD = BiRandom_Walk(Sm, Sd, RD_Zero, RD_Zero,l, r, alpha)
        #A_in_pt = pt[:490, 490:]

        save_every_column_TPR = []
        save_every_column_FPR = []

        pre_area = 0.0
        y_every_column_TPR, x_every_column_FPR = draw_roc(RD, A, sava_association_A, i, num, A_length, k,
                                                          save_all_count_zero_every_time_changed[i])

        x_all_FPR, y_all_TPR = gsp(y_every_column_TPR, x_every_column_FPR)
        y_length = len(y_all_TPR)

        for j in range(y_length):
            pre_area += area_ROC(y_all_TPR[j], x_all_FPR[j])
        #all_area += pre_area
        # save_every_column_TPR.append(np.sum(y_all_TPR, axis=0)/y_length)
        # save_every_column_FPR.append(np.sum(x_all_FPR, axis=0)/y_length)
        #print("该组最终roc面积", pre_area / y_length)
        all_area.append([pre_area / y_length])
        #save_all_y_TPR.extend(save_every_column_TPR)
        #save_all_x_FPR.extend(save_every_column_FPR)
        # plt.plot(np.sum(x_all_FPR, axis=0)/y_length, np.sum(y_all_TPR, axis=0)/y_length)
        # plt.show()
        #print(np.shape(save_all_x_TPR))
    print("平均roc面积", np.mean(all_area))
    #print(np.shape(save_all_y_FPR))
    # Y_TPR_average = (np.sum(save_all_x_FPR, axis=0))
    # X_FPR_average = (np.sum(save_all_y_TPR, axis=0))
    # plt.plot(X_FPR_average, Y_TPR_average)
    # plt.show()




def BiRandom_Walk( MR, MD, RD, A,l, r, alpha):
    count = 0
    alpha = alpha
    l = l
    r = r

    while count < max(r, l):
        z = RD
        rflag = dflag = 1
        if count < l:  # 原文是从 1开始循环 ，这里从0
            Rr = alpha * np.dot(MR, RD) + (1 - alpha) * A
            rflag = 1
            # df = pd.DataFrame(Rr)
            # df.to_excel('matrix_Rr.xlsx')
        if count < r:
            Rd = alpha * np.dot(RD, MD) + (1 - alpha) * A
            dflag = 1
            # df = pd.DataFrame(Rd)
            # df.to_excel('matrix_Rd.xlsx')
        RD = (rflag * Rr + dflag * Rd) / (rflag + dflag)

        # distance = np.sum(np.fabs(z- RD))
        # print(distance)
        # if distance < 1e-6:f
        #   break
        count += 1
    return RD


if __name__ == '__main__':

    alpha = 0.1
    k = 10
    l = 20
    r = 20
    #test1(alpha=alpha,k=k,max_count=max_count)
    test2(alpha,k,l,r)
    #get_k_pt(alpha, k, max_count)
