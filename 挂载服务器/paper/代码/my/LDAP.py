import random
import numpy as np
import copy
from operator import attrgetter
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

random.seed(1)




class value_index():
    def __init__(self,num,i,j):
        self.value = num  # save value
        self.value_x = i  # save index of row
        self.value_y = j  # save index of column
    def add_probability_predict(self,probability,predict_value):
        self.probability = probability
        self.predict_value = predict_value

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


# data partitioning train dev test
def data_partitioning(matrix_A, k):
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
        matrix_A, k)
    sava_k_Sd_associated = []
    save_count_positive = []
    all_k_test_data = []  # save all k test data
    for x in range(k):
        save_count_zero = copy.deepcopy(save_all_count_zero_not_changed)  # temp save zero

        column_length = len(save_all_count_A[x][0])  # the length of column    diseases 326
        row_length = len(save_all_count_A[x])  # the length of row        miRNA 429

        # just positive sample
        k_Sd_associated = []  # save 1 * 326  disease * diseases
        count_positive = 0
        for j in range(column_length):
            for i in range(row_length):
                if (save_all_count_A[x][i][j] == 1):  # use new associated_A get positive
                    count_positive += 1
                    # because save_all_count_A come from matrix A, so location i and j not change ,so could use i ,j
                    data = value_index(1, i, j)
                    k_Sd_associated.append(data)  # save a certain disease  with certain miRNA  label is 1
                    # print('positive number')
                    # print(count_positive)
                    # shuffle before choose negative samples
        random.shuffle(save_count_zero)
        # positive sets
        random.shuffle(k_Sd_associated)
        sava_k_Sd_associated.append(k_Sd_associated)
        save_count_positive.append(count_positive)

        # get test data
        save_all_count_zero = []  # 0 + previous 1 to 0
        save_all_count_zero.extend(save_count_zero)
        save_all_count_zero.extend(save_all_count_zero_every_time_changed[x])
        length = len(save_all_count_zero)
        # print('test data number')
        # print(length)
        random.shuffle(save_all_count_zero)  # shuffle all data [ 0 and  (1 to 0) ]
        temp_save = []
        for i in range(length):
            temp_save.append(
                value_index(save_all_count_zero[i].value, save_all_count_zero[i].value_x,
                           save_all_count_zero[i].value_y))
        all_k_test_data.append(temp_save)

    return save_all_count_zero_not_changed, sava_k_Sd_associated,save_count_positive,  all_k_test_data, num, A_length, sava_association_A, save_all_count_A


def get_data_feature(data,lnc_sim,dis_sim):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        x_A = data[j].value_x
        y_A = data[j].value_y
        # print(np.shape(lnc_sim[x_A]))  # (240,)
        # print(np.shape(dis_sim[y_A]))  # (405,)
        dis_lnc_feature = np.concatenate((lnc_sim[x_A], dis_sim[y_A]), axis=0)
        # dis_lnc_feature =  dis_sim[y_A]
        # print(np.shape(dis_lnc_feature))  # (645,)
        x.append(dis_lnc_feature)
        # print(np.shape(x))  # (1, 645)
        y.append(data[j].value)
    return x, y

def get_data_feature1(data,lnc_sim):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        x_A = data[j].value_x
        y_A = data[j].value_y
        # print(np.shape(lnc_sim[x_A]))  # (240,)
        # print(np.shape(dis_sim[y_A]))  # (405,)
        dis_lnc_feature = lnc_sim[x_A]
        # dis_lnc_feature =  dis_sim[y_A]
        # print(np.shape(dis_lnc_feature))  # (645,)
        x.append(dis_lnc_feature)
        # print(np.shape(x))  # (1, 645)
        y.append(data[j].value)
    return x, y
def get_data_feature2(data,dis_sim):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        x_A = data[j].value_x
        y_A = data[j].value_y
        # print(np.shape(lnc_sim[x_A]))  # (240,)
        # print(np.shape(dis_sim[y_A]))  # (405,)
        dis_lnc_feature = dis_sim[y_A]
        # dis_lnc_feature =  dis_sim[y_A]
        # print(np.shape(dis_lnc_feature))  # (645,)
        x.append(dis_lnc_feature)
        # print(np.shape(x))  # (1, 645)
        y.append(data[j].value)
    return x, y





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


def draw_roc_column1(clf1, clf2, clf3, clf4, clf5,rows,test_data,lnc_sim,dis_sim):
    #Disease_name = np.loadtxt("H:\\研究任务\\miRNA-diseaseAssociation_FirstTask\\资料准备\\Disname_5cross.txt", dtype=bytes,
                              #delimiter='/n').astype(str)
    #miRNA_name = np.loadtxt("H:\\研究任务\\miRNA-diseaseAssociation_FirstTask\\资料准备\\miname_all_5cross.txt", dtype=bytes,
                              #delimiter='/n').astype(str)
    areaROC = 0.0
    areaPR = 0
    count_have_positive = rows
    # according to column
    for i in range(rows):
        """
        # ---------------------test
        if(i!=200 and i!= 44 and i!=265):
            continue
        # ---------------------test
        """
        test_x, test_y = get_data_feature(test_data[i],lnc_sim,dis_sim)

        y_pred1 = clf1.predict_proba(test_x)
        y_pred2 = clf2.predict_proba(test_x)
        y_pred3 = clf3.predict_proba(test_x)
        y_pred4 = clf4.predict_proba(test_x)
        y_pred5 = clf5.predict_proba(test_x)

        pred_probability = ((y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5) / 5)
        # print(pred_probability)
        pred_one_probability = pred_probability[:,1]
        #
        # # y_pred_index =np.where(y_pred_pro ==np.max(y_pred_pro))
        predict_value = np.argmax(pred_probability, axis=1)
        # print(predict_value)
        # print(predict_value



        # according column to their initial column, and compute every column samples and its positive samples
        length_test = len(test_data[i])
        row = 1
        count_positive = np.array([0,0])  # [[count_column_number,count_positive_number] [] ...]
        for j in range(length_test):
            test_data[i][j].add_probability_predict(pred_one_probability[j], predict_value[j])
            count_positive[0] += 1
            if(test_data[i][j].value==1):
                count_positive[1] += 1
        #if the current column of test data no positive,so continue (get rid of train 0,1 )
        if(count_positive[1]==0):
            count_have_positive -= 1
            continue
        # else
        # sort
        temp_test_data = sorted(test_data[i], key=attrgetter('probability'), reverse=True)
        """
        # ---------test
        print("this lung neoplasms")
        for i in range(length_test):
            a = miRNA_name[temp_test_data[i].index_x]
            b = temp_test_data[i].predict_value
            c = temp_test_data[i].value
            print("miRNA_name,predict_value,real_value",a,b,c)
        # ---------test
        """
        # ROC
        all_number = length_test
        count_assume_true = 0
        count_negative = all_number - count_positive[1]
        # compute tp,fn,fp,tn --------first using not sorted data
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        TPR = 0.0
        FPR = 0.0
        TPR_temp = []
        FPR_temp = []
        P_temp = []
        R_temp = []
        for x in range(all_number):
            if (temp_test_data[x].value == 1 ):
                TP += 1
                FN = count_positive[1] - TP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                TPR_temp.append(TPR)
                #print(x_TPR_temp)
                FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative - FP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                TPR_temp.append(TPR)
                FPR_temp.append(FPR)
        area_column_ROC = area_ROC(TPR_temp, FPR_temp)
        area_column_PR = area_PR(P_temp, R_temp)
        #print("该次最终roc面积", area_column)
        #plt.plot(FPR_temp, TPR_temp)
        #plt.show()
        areaROC += area_column_ROC
        areaPR += area_column_PR
    areaROC = areaROC / count_have_positive
    areaPR = areaPR / count_have_positive
    return areaROC,areaPR


def draw_roc_column(clf1, clf2, clf3, clf4, clf5,columns,column_test_data,lnc_sim,dis_sim):
    areaROC = 0.0
    areaPR = 0.0
    count_have_positive = columns
    Y_TPR_all_columns = []
    X_FPR_all_columns = []
    Y_P_all_columns = []
    X_R_all_columns = []
    area_all_column = []
    # according to column
    for i in range(columns):

        test_x, test_y = get_data_feature(column_test_data[i], lnc_sim, dis_sim)

        y_pred1 = clf1.predict_proba(test_x)
        y_pred2 = clf2.predict_proba(test_x)
        y_pred3 = clf3.predict_proba(test_x)
        y_pred4 = clf4.predict_proba(test_x)
        y_pred5 = clf5.predict_proba(test_x)

        pred_probability = ((y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5) / 5)
        # print(pred_probability)
        pred_one_probability = pred_probability[:, 1]
        #
        # # y_pred_index =np.where(y_pred_pro ==np.max(y_pred_pro))
        predict_value = np.argmax(pred_probability, axis=1)





        # according column to their initial column, and compute every column samples and its positive samples
        length_test = len(column_test_data[i])
        row = 1
        count_positive = np.array([0,0])  # [[count_column_number,count_positive_number] [] ...]
        for j in range(length_test):
            column_test_data[i][j].add_probability_predict(pred_one_probability[j], predict_value[j])
            count_positive[0] += 1
            if(column_test_data[i][j].value==1):
                count_positive[1] += 1
        #if the current column of test data no positive,so continue (get rid of train 0,1 )
        if(count_positive[1]==0):
            count_have_positive -= 1
            continue
        # else
        # sort
        temp_test_data = sorted(column_test_data[i], key=attrgetter('probability'), reverse=True)
        """
        # ---------test
        print("this lung neoplasms")
        for i in range(length_test):
            a = miRNA_name[temp_test_data[i].index_x]
            b = temp_test_data[i].predict_value
            c = temp_test_data[i].value
            print("miRNA_name,predict_value,real_value",a,b,c)
        # ---------test
        """
        # ROC
        all_number = length_test
        count_assume_true = 0
        count_negative = all_number - count_positive[1]
        # compute tp,fn,fp,tn --------first using not sorted data
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        TPR = 0.0
        FPR = 0.0
        TPR_temp = []
        FPR_temp = []
        P_temp = []
        R_temp = []
        for x in range(all_number):
            if (temp_test_data[x].value == 1 ):
                TP += 1
                FN = count_positive[1] - TP
                TPR = TP / count_positive[1]
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                #print(x_TPR_temp)
                FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative - FP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                TPR_temp.append(TPR)
                FPR_temp.append(FPR)
            # print(R,P)
        # plt.plot(FPR_temp, TPR_temp)
        # plt.plot(R_temp,P_temp)
        # plt.show()
        area_column_ROC = area_ROC(TPR_temp, FPR_temp)
        area_column_PR = area_PR(P_temp, R_temp)
        Y_TPR_all_columns.append(TPR_temp)
        X_FPR_all_columns.append(FPR_temp)
        Y_P_all_columns.append(P_temp)
        X_R_all_columns.append(R_temp)
        area_all_column.extend([area_column_ROC])
        #print("该次最终roc面积", area_column)
        #plt.plot(FPR_temp, TPR_temp)
        #plt.show()
        areaROC += area_column_ROC
        areaPR += area_column_PR
    #print("----")
    #print(len(X_FPR_all_columns))
    #print(count_have_positive)
    areaROC = areaROC / count_have_positive
    areaPR = areaPR / count_have_positive

    return Y_TPR_all_columns,X_FPR_all_columns,Y_P_all_columns,X_R_all_columns,areaROC,areaPR


def lncRNA_sim_gip(A):
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

def dis_sim_gip(A):
    length_lnc = len(A)
    length_dis = len(A[0])

    sum0 = 0
    for i in range(length_dis):
        sum0 += np.sum(np.square((A[:,i])))
    gamal = sum0/length_dis
    # print(gamal)
    dis_sim = np.zeros((length_dis,length_dis))
    for i in range(length_dis):
        for j in range(length_dis):
            dis_sim[i][j] = np.exp((np.sum(np.square(A[:,i] - A[:,j])))/(-gamal))
    return dis_sim


def train_set(save_count_zero,k_Sd_associated,count_positive):
    k_Sd_samples = []
    k_Sd_negative = []
    count_negative = count_positive
    # get frontal count_positive number because shuffle before taking
    for i in range(count_negative):
        k_Sd_negative.append(value_index(0, save_count_zero[i].value_x, save_count_zero[i].value_y))
    k_Sd_samples.extend(k_Sd_associated)
    k_Sd_samples.extend(k_Sd_negative)  # save in a raw
    random.shuffle(k_Sd_samples)
    return k_Sd_samples


def LDAP(i,save_count_zero,k_Sd_associated,count_positive,lnc_sim,dis_sim):
    random.seed(1)
    random.shuffle(save_count_zero)
    x_train_data1,y_train_data1 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    random.seed(2)
    random.shuffle(save_count_zero)
    x_train_data2, y_train_data2 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    random.seed(3)
    random.shuffle(save_count_zero)
    x_train_data3, y_train_data3 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    random.seed(4)
    random.shuffle(save_count_zero)
    x_train_data4, y_train_data4 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    random.seed(5)
    random.shuffle(save_count_zero)
    x_train_data5, y_train_data5 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    # x_test,y_test = get_data_feature(all_k_test_data, lnc_sim, dis_sim)


    clf1 = SVC(probability=True)
    clf2 = SVC(probability=True)
    clf3 = SVC(probability=True)
    clf4 = SVC(probability=True)
    clf5 = SVC(probability=True)
    # voting_clf = VotingClassifier(estimators=[("svm1", clf1), ("svm2", clf2), ("svm3", clf3), ("svm4", clf4), ("svm5", clf5)], voting="soft")
    # voting_clf.fit(X_train, y_train)
    clf1.fit(x_train_data1,y_train_data1)
    clf2.fit(x_train_data2, y_train_data2)
    clf3.fit(x_train_data3, y_train_data3 )
    clf4.fit(x_train_data4, y_train_data4)
    clf5.fit(x_train_data5, y_train_data5)

    return clf1,clf2,clf3,clf4,clf5

def get_current_AUC_ROC(clf1, clf2, clf3, clf4, clf5,test_data,columns,lnc_sim,dis_sim):

    #cnn.eval()
    length_test = len(test_data)
    columns = columns
    all_column_test_data = [[] for row in range(columns)]  # create a  list [[],[]...]  rows
    for i in range(length_test):
        all_column_test_data[test_data[i].value_y].append(test_data[i])  # ------pay attention append or extend
    # according to different column
    Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns, areaROC, areaPR = draw_roc_column(clf1, clf2, clf3, clf4, clf5,columns, all_column_test_data,lnc_sim,dis_sim)
    x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns)
    x_FPR = np.sum(x_all_FPR, axis=0) / len(X_FPR_all_columns)
    y_TPR = np.sum(y_all_TPR, axis=0) / len(Y_TPR_all_columns)
    x_R = np.sum(x_all_R, axis=0) / len(X_R_all_columns)
    y_P = np.sum(y_all_P, axis=0) / len(Y_P_all_columns)
    #print('Time:{}, AUC:{}'.format(time, average_column_area))
    #plt.plot(x_FPR, y_TPR)
    #plt.show()
    return x_FPR,y_TPR,x_R,y_P,areaROC, areaPR


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

def LDAP1(i,save_count_zero,k_Sd_associated,count_positive,lnc_sim,dis_sim):
    random.seed(1)
    random.shuffle(save_count_zero)
    x_train_data1,y_train_data1 = get_data_feature1(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim)
    x_train_data2, y_train_data2 = get_data_feature2(train_set(save_count_zero, k_Sd_associated, count_positive), dis_sim)
    # random.seed(2)
    # random.shuffle(save_count_zero)
    # x_train_data2, y_train_data2 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    # random.seed(3)
    # random.shuffle(save_count_zero)
    # x_train_data3, y_train_data3 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    # random.seed(4)
    # random.shuffle(save_count_zero)
    # x_train_data4, y_train_data4 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    # random.seed(5)
    # random.shuffle(save_count_zero)
    # x_train_data5, y_train_data5 = get_data_feature(train_set(save_count_zero, k_Sd_associated, count_positive), lnc_sim, dis_sim)
    # x_test,y_test = get_data_feature(all_k_test_data, lnc_sim, dis_sim)


    clf1 = SVC(probability=True)
    clf2 = SVC(probability=True)
    # clf2 = SVC(probability=True)
    # clf3 = SVC(probability=True)
    # clf4 = SVC(probability=True)
    # clf5 = SVC(probability=True)
    # voting_clf = VotingClassifier(estimators=[("svm1", clf1), ("svm2", clf2), ("svm3", clf3), ("svm4", clf4), ("svm5", clf5)], voting="soft")
    # voting_clf.fit(X_train, y_train)
    clf1.fit(x_train_data1,y_train_data1)
    clf2.fit(x_train_data2, y_train_data2 )
    # clf2.fit(x_train_data2, y_train_data2)
    # clf3.fit(x_train_data3, y_train_data3 )
    # clf4.fit(x_train_data4, y_train_data4)
    # clf5.fit(x_train_data5, y_train_data5)

    return clf1,clf2


def draw_roc_column5(clf1,clf2,columns,column_test_data,lnc_sim,dis_sim):
    areaROC = 0.0
    areaPR = 0.0
    count_have_positive = columns
    Y_TPR_all_columns = []
    X_FPR_all_columns = []
    Y_P_all_columns = []
    X_R_all_columns = []
    area_all_column = []
    Rcall_all = []
    AUC_column = []
    PR_column = []
    # dis_15 = [7, 9, 10, 61, 62, 69, 113, 126, 140, 156, 211, 233, 316, 334, 335, 338]
    dis_15 = [62, 69, 113, 140, 156, 178, 181, 187, 233, 297]

    test_15_dis_AUC = []
    test_15_dis_PR = []
    # according to column
    for i in range(columns):

        # test_x, test_y = get_data_feature(column_test_data[i], lnc_sim, dis_sim)
        test_x1, test_y = get_data_feature1(column_test_data[i], lnc_sim)
        test_x2, test_y = get_data_feature2(column_test_data[i], dis_sim)
        y_pred1 = clf1.predict_proba(test_x1)
        y_pred2 = clf2.predict_proba(test_x2)
        # y_pred3 = clf3.predict_proba(test_x)
        # y_pred4 = clf4.predict_proba(test_x)


        pred_probability = ((y_pred1+y_pred2 ) / 2)
        # print(pred_probability)
        pred_one_probability = pred_probability[:, 1]
        #
        # # y_pred_index =np.where(y_pred_pro ==np.max(y_pred_pro))
        predict_value = np.argmax(pred_probability, axis=1)





        # according column to their initial column, and compute every column samples and its positive samples
        length_test = len(column_test_data[i])
        row = 1
        count_positive = np.array([0,0])  # [[count_column_number,count_positive_number] [] ...]
        for j in range(length_test):
            column_test_data[i][j].add_probability_predict(pred_one_probability[j], predict_value[j])
            count_positive[0] += 1
            if(column_test_data[i][j].value==1):
                count_positive[1] += 1
        #if the current column of test data no positive,so continue (get rid of train 0,1 )
        if(count_positive[1]==0):
            count_have_positive -= 1
            continue
        # else
        # sort
        temp_test_data = sorted(column_test_data[i], key=attrgetter('probability'), reverse=True)
        """
        # ---------test
        print("this lung neoplasms")
        for i in range(length_test):
            a = miRNA_name[temp_test_data[i].index_x]
            b = temp_test_data[i].predict_value
            c = temp_test_data[i].value
            print("miRNA_name,predict_value,real_value",a,b,c)
        # ---------test
        """
        # ROC
        all_number = length_test
        count_assume_true = 0
        count_negative = all_number - count_positive[1]
        # compute tp,fn,fp,tn --------first using not sorted data
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        R = 0.0
        TPR = 0.0
        FPR = 0.0
        TPR_temp = []
        FPR_temp = []
        P_temp = []
        R_temp = []
        Rcall_temp = []
        for x in range(all_number):
            if (x == 30 or x == 60 or x == 90 or x == 120 or x == 150 or x == 180 or x == 210 or x ==240 ):
                Rcall_temp.append(R)
            if (temp_test_data[x].value == 1 ):
                TP += 1
                FN = count_positive[1] - TP
                TPR = TP / count_positive[1]
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                #print(x_TPR_temp)
                FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative - FP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                TPR_temp.append(TPR)
                FPR_temp.append(FPR)
        while len(Rcall_temp) < 8:
            Rcall_temp.append(1.0)
            # print(R,P)
        # plt.plot(FPR_temp, TPR_temp)
        # plt.plot(R_temp,P_temp)
        # plt.show()
        Rcall_all.append(Rcall_temp)
        area_column_ROC = area_ROC(TPR_temp, FPR_temp)
        area_column_PR = area_PR(P_temp, R_temp)
        AUC_column.extend([area_column_ROC])
        PR_column.extend([area_column_PR])
        if i in dis_15:
            test_15_dis_AUC.extend([area_column_ROC])
            test_15_dis_PR.extend([area_column_PR])
        Y_TPR_all_columns.append(TPR_temp)
        X_FPR_all_columns.append(FPR_temp)
        Y_P_all_columns.append(P_temp)
        X_R_all_columns.append(R_temp)
        area_all_column.extend([area_column_ROC])
        #print("该次最终roc面积", area_column)
        #plt.plot(FPR_temp, TPR_temp)
        #plt.show()
        areaROC += area_column_ROC
        areaPR += area_column_PR
    #print("----")
    #print(len(X_FPR_all_columns))
    #print(count_have_positive)
    areaROC = areaROC / count_have_positive
    areaPR = areaPR / count_have_positive

    return Y_TPR_all_columns,X_FPR_all_columns,Y_P_all_columns,X_R_all_columns,areaROC,areaPR,Rcall_all,AUC_column,PR_column,test_15_dis_AUC,test_15_dis_PR


def draw_roc_column5_test(clf1,clf2,columns,column_test_data,lnc_sim,dis_sim):

    count_have_positive = columns


    AUC_column = []
    PR_column = []

    # according to column
    for i in range(columns):

        # test_x, test_y = get_data_feature(column_test_data[i], lnc_sim, dis_sim)
        test_x1, test_y = get_data_feature1(column_test_data[i], lnc_sim)
        test_x2, test_y = get_data_feature2(column_test_data[i], dis_sim)
        y_pred1 = clf1.predict_proba(test_x1)
        y_pred2 = clf2.predict_proba(test_x2)
        # y_pred3 = clf3.predict_proba(test_x)
        # y_pred4 = clf4.predict_proba(test_x)


        pred_probability = ((y_pred1+y_pred2 ) / 2)
        # print(pred_probability)
        pred_one_probability = pred_probability[:, 1]
        #
        # # y_pred_index =np.where(y_pred_pro ==np.max(y_pred_pro))
        predict_value = np.argmax(pred_probability, axis=1)





        # according column to their initial column, and compute every column samples and its positive samples
        length_test = len(column_test_data[i])
        row = 1
        count_positive = np.array([0,0])  # [[count_column_number,count_positive_number] [] ...]
        for j in range(length_test):
            column_test_data[i][j].add_probability_predict(pred_one_probability[j], predict_value[j])
            count_positive[0] += 1
            if(column_test_data[i][j].value==1):
                count_positive[1] += 1
        #if the current column of test data no positive,so continue (get rid of train 0,1 )
        if(count_positive[1]==0):
            count_have_positive -= 1
            continue
        # else
        # sort
        temp_test_data = sorted(column_test_data[i], key=attrgetter('probability'), reverse=True)
        """
        # ---------test
        print("this lung neoplasms")
        for i in range(length_test):
            a = miRNA_name[temp_test_data[i].index_x]
            b = temp_test_data[i].predict_value
            c = temp_test_data[i].value
            print("miRNA_name,predict_value,real_value",a,b,c)
        # ---------test
        """
        # ROC
        all_number = length_test
        count_assume_true = 0
        count_negative = all_number - count_positive[1]
        # compute tp,fn,fp,tn --------first using not sorted data
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        R = 0.0
        TPR = 0.0
        FPR = 0.0
        TPR_temp = []
        FPR_temp = []
        P_temp = []
        R_temp = []
        Rcall_temp = []
        for x in range(all_number):

            if (temp_test_data[x].value == 1 ):
                TP += 1
                FN = count_positive[1] - TP
                TPR = TP / count_positive[1]
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                #print(x_TPR_temp)
                FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative - FP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                P = TP / (TP + FP)
                R = TP / (count_positive[1])
                P_temp.append(P)
                R_temp.append(R)
                TPR_temp.append(TPR)
                FPR_temp.append(FPR)

            # print(R,P)
        # plt.plot(FPR_temp, TPR_temp)
        # plt.plot(R_temp,P_temp)
        # plt.show()

        area_column_ROC = area_ROC(TPR_temp, FPR_temp)
        area_column_PR = area_PR(P_temp, R_temp)
        AUC_column.extend([area_column_ROC])
        PR_column.extend([area_column_PR])

    return AUC_column,PR_column


def get_current_AUC_ROC1(clf1,clf2,test_data,columns,lnc_sim,dis_sim):

    #cnn.eval()
    length_test = len(test_data)
    columns = columns
    all_column_test_data = [[] for row in range(columns)]  # create a  list [[],[]...]  rows
    for i in range(length_test):
        all_column_test_data[test_data[i].value_y].append(test_data[i])  # ------pay attention append or extend
    # according to different column
    Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns, areaROC, areaPR, Rcall_all,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR  = draw_roc_column5(clf1,clf2, columns, all_column_test_data,lnc_sim,dis_sim)
    x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns)
    x_FPR = np.sum(x_all_FPR, axis=0) / len(X_FPR_all_columns)
    y_TPR = np.sum(y_all_TPR, axis=0) / len(Y_TPR_all_columns)
    x_R = np.sum(x_all_R, axis=0) / len(X_R_all_columns)
    y_P = np.sum(y_all_P, axis=0) / len(Y_P_all_columns)
    Rcall = np.sum(Rcall_all,axis=0) /len(Rcall_all)

    #print('Time:{}, AUC:{}'.format(time, average_column_area))
    #plt.plot(x_FPR, y_TPR)
    #plt.show()
    return x_FPR,y_TPR,x_R,y_P,areaROC, areaPR,Rcall,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR


def test3():


    # print(lnc_sim)

    k = 5
    alpha = 0.6
    # A = np.loadtxt('interMatrix.txt')
    A = np.loadtxt("../data_create/lnc_dis_association.txt")
    # A = np.loadtxt("./Dataset1/interMatrix.txt")

    save_count_zero, sava_k_Sd_associated,save_count_positive,all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)

    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    for i in range(k):
        dis_sim = dis_sim_gip(save_all_count_A[i])
        lnc_sim = lncRNA_sim_gip(save_all_count_A[i])

        clf1, clf2, clf3, clf4, clf5 = LDAP(i,save_count_zero,sava_k_Sd_associated[i],save_count_positive[i],lnc_sim,dis_sim)


        test_data = all_k_test_data[i]
        length_test = len(test_data)
        rows = len(A[0])
        # print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].value_y].append(test_data[j])  # ------pay attention append or extend

        # # according to different column
        # areaROC, areaPR = draw_roc_column1(clf1, clf2, clf3, clf4, clf5, rows, all_column_test_data, lnc_sim,dis_sim)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(clf1, clf2, clf3, clf4, clf5,  test_data, len(A[0]),lnc_sim,dis_sim)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR
        # print(areaROC,areaPR)
        del clf1, clf2, clf3, clf4, clf5

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    # print(a_a_1,a_a_2)
    # d = 'area:{},LR:{},epoch1:{},epoch2:{}'.format(a_a,_LR,epoch1,epoch2)
    get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    d = 'area_ROC:{},area_PR:{}'.format(a_a_1,a_a_2)

    print(d)
def predict_top(clf1,clf2,columns,column_test_data,lnc_sim,dis_sim):
    all_predict_test = []
    count_positive = np.array([0,0])
    R_all = []
    # according to column
    for i in range(columns):
        test_x1, test_y = get_data_feature1(column_test_data[i], lnc_sim)
        test_x2, test_y = get_data_feature2(column_test_data[i], dis_sim)
        y_pred1 = clf1.predict_proba(test_x1)
        y_pred2 = clf2.predict_proba(test_x2)

        pred_probability = ((y_pred1 + y_pred2) / 2)
        # print(pred_probability)
        pred_one_probability = pred_probability[:, 1]
        #
        # # y_pred_index =np.where(y_pred_pro ==np.max(y_pred_pro))
        predict_value = np.argmax(pred_probability, axis=1)
        # according column to their initial column, and compute every column samples and its positive samples
        length_test = len(column_test_data[i])
        row = 1
        # count_positive =   # [[count_column_number,count_positive_number] [] ...]
        for j in range(length_test):
            column_test_data[i][j].add_probability_predict(pred_one_probability[j], predict_value[j])
            count_positive[0] += 1
            if(column_test_data[i][j].value==1):
                count_positive[1] += 1
        #if the current column of test data no positive,so continue (get rid of train 0,1 )
        # if(count_positive[1]==0):
        #     count_have_positive -= 1
        #     continue
        # else
        # sort
        all_predict_test.extend(column_test_data[i])
    print(count_positive[1])
    temp_test_data = sorted(all_predict_test, key=attrgetter('probability'), reverse=True)
    length_test = len(temp_test_data)
    # print(length_test)
    # print(count_positive[0])
    # ROC
    all_number = length_test
    count_assume_true = 0
    count_negative = all_number - count_positive[1]


    # compute tp,fn,fp,tn --------first using not sorted data
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    TPR = 0.0
    FPR = 0.0
    R = 0.0

    # TPR_temp = []
    # FPR_temp = []
    # P_temp = []
    # R_temp = []
    for x in range(all_number):
        if(x==30 or x==60 or x==90 or x==120 or x==150 or x==180 or x==210 or x==240 ):
            R_all.append(R)
        if (temp_test_data[x].value == 1):
            TP += 1
            FN = count_positive[1] - TP
            TPR = TP / count_positive[1]
            P = TP / (TP + FP)
            R = TP / (count_positive[1])
            # P_temp.append(P)
            # R_temp.append(R)
            # FPR = FP / count_negative
            # TPR_temp.append(TPR)
            # print(x_TPR_temp)
            # FPR_temp.append(FPR)
        else:
            FP += 1
            TN = count_negative - FP
            TPR = TP / count_positive[1]
            FPR = FP / count_negative
            P = TP / (TP + FP)
            R = TP / (count_positive[1])
            # P_temp.append(P)
            # R_temp.append(R)
            # TPR_temp.append(TPR)
            # FPR_temp.append(FPR)


    return R_all

def test():


    # print(lnc_sim)

    k = 5
    alpha = 0.6
    # A = np.loadtxt('interMatrix.txt')
    A = np.loadtxt("../data_create/lnc_dis_association.txt")
    # A = np.loadtxt("./Dataset1/interMatrix.txt")
    # dis_sim = np.loadtxt("../data_create/dis_sim_matrix_process.txt")

    save_count_zero, sava_k_Sd_associated,save_count_positive,all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)

    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    for i in range(k):
        dis_sim = dis_sim_gip(save_all_count_A[i])
        lnc_sim = lncRNA_sim_gip(save_all_count_A[i])

        # clf1, clf2, clf3, clf4, clf5 = LDAP(i,save_count_zero,sava_k_Sd_associated[i],save_count_positive[i],lnc_sim,dis_sim)
        clf1,clf2= LDAP1(i, save_count_zero, sava_k_Sd_associated[i], save_count_positive[i],
                                            lnc_sim, dis_sim)


        test_data = all_k_test_data[i]
        length_test = len(test_data)
        rows = len(A[0])
        # print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].value_y].append(test_data[j])  # ------pay attention append or extend

        # # according to different column
        # areaROC, areaPR = draw_roc_column1(clf1, clf2, clf3, clf4, clf5, rows, all_column_test_data, lnc_sim,dis_sim)
        # x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(clf1, clf2, clf3, clf4, clf5,  test_data, len(A[0]),lnc_sim,dis_sim)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC1(clf1,clf2 ,test_data,
                                                                      len(A[0]), lnc_sim, dis_sim)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR
        print(areaROC,areaPR)
        del clf1

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print(a_a_1,a_a_2)
    # d = 'area:{},LR:{},epoch1:{},epoch2:{}'.format(a_a,_LR,epoch1,epoch2)
    get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    d = 'area_ROC:{},area_PR:{}'.format(a_a_1,a_a_2)

    print(d)

def data_partitioning1( k,save_all_count_A,save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A,):
    # save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = crossvalidation(
    #     matrix_A, k)
    sava_k_Sd_associated = []
    save_count_positive = []
    all_k_test_data = []  # save all k test data
    for x in range(k):
        save_count_zero = copy.deepcopy(save_all_count_zero_not_changed)  # temp save zero

        column_length = len(save_all_count_A[x][0])  # the length of column    diseases 326
        row_length = len(save_all_count_A[x])  # the length of row        miRNA 429

        # just positive sample
        k_Sd_associated = []  # save 1 * 326  disease * diseases
        count_positive = 0
        for j in range(column_length):
            for i in range(row_length):
                if (save_all_count_A[x][i][j] == 1):  # use new associated_A get positive
                    count_positive += 1
                    # because save_all_count_A come from matrix A, so location i and j not change ,so could use i ,j
                    data = value_index(1, i, j)
                    k_Sd_associated.append(data)  # save a certain disease  with certain miRNA  label is 1
                    # print('positive number')
                    # print(count_positive)
                    # shuffle before choose negative samples
        random.shuffle(save_count_zero)
        # positive sets
        random.shuffle(k_Sd_associated)
        sava_k_Sd_associated.append(k_Sd_associated)
        save_count_positive.append(count_positive)

        # get test data
        save_all_count_zero = []  # 0 + previous 1 to 0
        save_all_count_zero.extend(save_count_zero)
        save_all_count_zero.extend(save_all_count_zero_every_time_changed[x])
        length = len(save_all_count_zero)
        # print('test data number')
        # print(length)
        random.shuffle(save_all_count_zero)  # shuffle all data [ 0 and  (1 to 0) ]
        temp_save = []
        for i in range(length):
            temp_save.append(
                value_index(save_all_count_zero[i].value, save_all_count_zero[i].value_x,
                           save_all_count_zero[i].value_y))
        all_k_test_data.append(temp_save)

    return save_all_count_zero_not_changed, sava_k_Sd_associated,save_count_positive,  all_k_test_data, num, A_length, sava_association_A, save_all_count_A
def test3(A,save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A):


    # print(lnc_sim)

    k = 5
    alpha = 0.6
    # A = np.loadtxt('interMatrix.txt')
    # A = np.loadtxt("../data_create/lnc_dis_association.txt")
    # A = np.loadtxt("./Dataset1/interMatrix.txt")
    save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = \
        data_partitioning1(k, save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A)
    # save_count_zero, sava_k_Sd_associated,save_count_positive,all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)
    # save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    for i in range(k):
        dis_sim = dis_sim_gip(save_all_count_A[i])
        lnc_sim = lncRNA_sim_gip(save_all_count_A[i])

        clf1, clf2, clf3, clf4, clf5 = LDAP(i,save_all_count_zero_not_changed,sava_k_Sd_associated[i],save_count_positive[i],lnc_sim,dis_sim)


        test_data = all_k_test_data[i]
        length_test = len(test_data)
        rows = len(A[0])
        # print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].value_y].append(test_data[j])  # ------pay attention append or extend

        # # according to different column
        # areaROC, areaPR = draw_roc_column1(clf1, clf2, clf3, clf4, clf5, rows, all_column_test_data, lnc_sim,dis_sim)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(clf1, clf2, clf3, clf4, clf5,  test_data, len(A[0]),lnc_sim,dis_sim)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR
        # print(areaROC,areaPR)
        del clf1, clf2, clf3, clf4, clf5

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print(a_a_1,a_a_2)
    # d = 'area:{},LR:{},epoch1:{},epoch2:{}'.format(a_a,_LR,epoch1,epoch2)
    FPR, TPR, R, P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    # d = 'area_ROC:{},area_PR:{}'.format(a_a_1,a_a_2)
    return FPR, TPR, R, P,a_a_1,a_a_2
    # print(d)

def test2(A,save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A):


    # print(lnc_sim)

    k = 5
    alpha = 0.6
    # A = np.loadtxt('interMatrix.txt')
    # A = np.loadtxt("../data_create/lnc_dis_association.txt")
    # A = np.loadtxt("./Dataset1/interMatrix.txt")
    save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = \
        data_partitioning1(k, save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A)
    # save_count_zero, sava_k_Sd_associated,save_count_positive,all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)
    # save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)
    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    Rcall_all = []

    for i in range(k):
        dis_sim = dis_sim_gip(save_all_count_A[i])
        lnc_sim = lncRNA_sim_gip(save_all_count_A[i])

        # clf1, clf2, clf3, clf, clf5 = LDAP(i,save_all_count_zero_not_changed,sava_k_Sd_associated[i],save_count_positive[i],lnc_sim,dis_sim)
        clf1, clf2 = LDAP1(i, save_all_count_zero_not_changed, sava_k_Sd_associated[i], save_count_positive[i],
                           lnc_sim, dis_sim)

        test_data = all_k_test_data[i]
        length_test = len(test_data)
        rows = len(A[0])
        # print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].value_y].append(test_data[j])  # ------pay attention append or extend

        # # according to different column
        # areaROC, areaPR = draw_roc_column1(clf1, clf2, clf3, clf4, clf5, rows, all_column_test_data, lnc_sim,dis_sim)
        # x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(clf1, clf2, clf3, clf4, clf5,  test_data, len(A[0]),lnc_sim,dis_sim)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR,Rcall = get_current_AUC_ROC1(clf1, clf2, test_data,
                                                                       len(A[0]), lnc_sim, dis_sim)
        Rcall_all.append(Rcall)

        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC += areaROC
        area_all_PR += areaPR
        # print(areaROC,areaPR)
        del clf1, clf2
    Rcall_last = np.sum(Rcall_all,axis=0) /k

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print(a_a_1,a_a_2)
    # d = 'area:{},LR:{},epoch1:{},epoch2:{}'.format(a_a,_LR,epoch1,epoch2)
    FPR, TPR, R, P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    # d = 'area_ROC:{},area_PR:{}'.format(a_a_1,a_a_2)
    return FPR, TPR, R, P,a_a_1,a_a_2,Rcall_last
    # print(d)


def test_top():


    # print(lnc_sim)

    k = 5
    alpha = 0.6
    # A = np.loadtxt('interMatrix.txt')
    A = np.loadtxt("../data_create/lnc_dis_association.txt")
    # A = np.loadtxt("./Dataset1/interMatrix.txt")
    # dis_sim = np.loadtxt("../data_create/dis_sim_matrix_process.txt")

    save_count_zero, sava_k_Sd_associated,save_count_positive,all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)

    area_all_ROC = 0
    area_all_PR = 0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    R_all = []
    for i in range(k):
        dis_sim = dis_sim_gip(save_all_count_A[i])
        lnc_sim = lncRNA_sim_gip(save_all_count_A[i])

        # clf1, clf2, clf3, clf4, clf5 = LDAP(i,save_count_zero,sava_k_Sd_associated[i],save_count_positive[i],lnc_sim,dis_sim)
        clf1,clf2= LDAP1(i, save_count_zero, sava_k_Sd_associated[i], save_count_positive[i],
                                            lnc_sim, dis_sim)


        test_data = all_k_test_data[i]
        length_test = len(test_data)
        rows = len(A[0])
        # #print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].value_y].append(test_data[j])  # ------pay attention append or extend
        # # # according to different column
        # areaROC0, areaPR0 = draw_roc_column1(model, rows,all_column_test_data,save_all_count_A[i],Sm,Sd,Lm,Md)
        # x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(model, test_data, len(A[0]),save_all_count_A[i],Sm,Sd,Lm,Md,i)
        R_call = predict_top(clf1,clf2 ,len(A[0]),all_column_test_data, lnc_sim, dis_sim)
        print(R_call)
        R_all.append(R_call)
        del clf1

    R = np.sum(R_all, axis=0) / 5
    print(R)


def test2_test(A,save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A):


    # print(lnc_sim)

    k = 5
    alpha = 0.6
    # A = np.loadtxt('interMatrix.txt')
    # A = np.loadtxt("../data_create/lnc_dis_association.txt")
    # A = np.loadtxt("./Dataset1/interMatrix.txt")
    save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = \
        data_partitioning1(k, save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A)
    # save_count_zero, sava_k_Sd_associated,save_count_positive,all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)
    # save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)
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
        dis_sim = dis_sim_gip(save_all_count_A[i])
        lnc_sim = lncRNA_sim_gip(save_all_count_A[i])

        # clf1, clf2, clf3, clf, clf5 = LDAP(i,save_all_count_zero_not_changed,sava_k_Sd_associated[i],save_count_positive[i],lnc_sim,dis_sim)
        clf1, clf2 = LDAP1(i, save_all_count_zero_not_changed, sava_k_Sd_associated[i], save_count_positive[i],
                           lnc_sim, dis_sim)

        test_data = all_k_test_data[i]
        length_test = len(test_data)
        rows = len(A[0])
        # print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].value_y].append(test_data[j])  # ------pay attention append or extend

        # # according to different column
        # areaROC, areaPR = draw_roc_column1(clf1, clf2, clf3, clf4, clf5, rows, all_column_test_data, lnc_sim,dis_sim)
        # x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(clf1, clf2, clf3, clf4, clf5,  test_data, len(A[0]),lnc_sim,dis_sim)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR,Rcall,AUC_column, PR_column,test_15_dis_AUC,test_15_dis_PR  = get_current_AUC_ROC1(clf1, clf2, test_data,
                                                                       len(A[0]), lnc_sim, dis_sim)
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
        # print(areaROC,areaPR)
        del clf1, clf2
    Rcall_last = np.sum(Rcall_all,axis=0) /k

    a_a_1 = area_all_ROC / k
    a_a_2 = area_all_PR / k
    print(a_a_1,a_a_2)
    # d = 'area:{},LR:{},epoch1:{},epoch2:{}'.format(a_a,_LR,epoch1,epoch2)
    FPR, TPR, R, P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    # d = 'area_ROC:{},area_PR:{}'.format(a_a_1,a_a_2)
    return FPR, TPR, R, P,a_a_1,a_a_2,Rcall_last,AUC_all_column, PR_all_column,dis_15_AUC,dis_15_PR
    # print(d)


def test3_test(A,save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A):


    # print(lnc_sim)

    k = 5
    alpha = 0.6
    # A = np.loadtxt('interMatrix.txt')
    # A = np.loadtxt("../data_create/lnc_dis_association.txt")
    # A = np.loadtxt("./Dataset1/interMatrix.txt")
    save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = \
        data_partitioning1(k, save_all_count_A, save_all_count_zero_not_changed,save_all_count_zero_every_time_changed, num, A_length, sava_association_A)
    # save_count_zero, sava_k_Sd_associated,save_count_positive,all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)
    # save_all_count_zero_not_changed, sava_k_Sd_associated, save_count_positive, all_k_test_data, num, A_length, sava_association_A, save_all_count_A = data_partitioning(A,k)

    AUC_all_column = []
    PR_all_column = []
    for i in range(k):
        dis_sim = dis_sim_gip(save_all_count_A[i])
        lnc_sim = lncRNA_sim_gip(save_all_count_A[i])

        # clf1, clf2, clf3, clf, clf5 = LDAP(i,save_all_count_zero_not_changed,sava_k_Sd_associated[i],save_count_positive[i],lnc_sim,dis_sim)
        clf1, clf2 = LDAP1(i, save_all_count_zero_not_changed, sava_k_Sd_associated[i], save_count_positive[i],
                           lnc_sim, dis_sim)

        test_data = all_k_test_data[i]
        length_test = len(test_data)
        columns = len(A[0])
        # print(rows)
        all_column_test_data = [[] for column in range(columns)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].value_y].append(test_data[j])  # ------pay attention append or extend


        # according to different column
        AUC_column, PR_column = draw_roc_column5_test(clf1, clf2, columns, all_column_test_data, lnc_sim, dis_sim)

        AUC_all_column.extend(AUC_column)
        PR_all_column.extend(PR_column)

    return AUC_all_column, PR_all_column




if __name__ == '__main__':
    # A = np.loadtxt("./Dataset1/interMatrix.txt")
    # dis_sim = dis_sim_gip(A)
    # lnc_sim = lncRNA_sim_gip(A)
    # test()
    test_top()
































# print(np.exp(-(5/3)))