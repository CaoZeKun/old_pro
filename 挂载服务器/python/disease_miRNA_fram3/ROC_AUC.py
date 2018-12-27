import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn import utils as nn_utils
import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from operator import itemgetter
import data_load as dl


class node():
    #def __init__(self):
    def add_predict_probability_value(self,predict_probability,predict_value):
        self.predict_probability = predict_probability
        self.predict_value = predict_value

def draw_roc_column1(cnn,rows,test_data,Sd_associated,Sm,Sd):
    area = 0.0
    count_have_positive = rows
    # according to column
    for i in range(rows):
        """
        # ---------------------test
        if(i!=200 and i!= 44 and i!=265):
            continue
        # ---------------------test
        """
        test_x, test_y = get_test_data(test_data[i],Sd_associated,Sm,Sd)
        test_out = cnn(test_x)
        #print(test_out)
        #print("--------------")
        pred_one_probability = F.softmax(test_out,dim=1)[:,1].data.squeeze().cpu().numpy()
        #print(F.softmax(test_out))
        #print("--------------")
        #print(F.softmax(test_out)[:,1])
        #pred_one_probability = test_out[:][1].data.squeeze().numpy()  # the probability of predicting 1
        predict_value = torch.max(F.softmax(test_out,dim=1), 1)[1].data.squeeze().int().cpu().numpy()  # the index of the high probability, predicted value

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
        for x in range(all_number):
            if (temp_test_data[x].value == 1 ):
                TP += 1
                FN = count_positive[1] - TP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                #print(x_TPR_temp)
                FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative - FP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                FPR_temp.append(FPR)
        area_column = area_ROC(TPR_temp, FPR_temp)
        #print("该次最终roc面积", area_column)
        #plt.plot(FPR_temp, TPR_temp)
        #plt.show()
        area += area_column
    return area/count_have_positive


def draw_roc_column2(the_model,rows,test_data,Sd_associated,Sm,Sd,A):
    area = 0.0
    count_have_positive = rows
    Y_TPR_all_columns = []
    X_FPR_all_columns = []
    # according to column
    for i in range(rows):
        """
        # ---------------------test
        if(i!=200 and i!= 44 and i!=265):
            continue
        # ---------------------test
        """
        rvalue_probability_pvalue = []
        pred_column_probability = []
        predict_column_value = []
        real_value_column = []
        nodes_length = len(test_data[i])
        for sample in range(nodes_length):
            the_model.train()
            # print(nodes_set[sample])
            goals, paths_feature, paths_length = dl.add_feature5(test_data[i][sample], A, Sd_associated, Sm, Sd)
            # print(np.shape(paths_feature))

            x = Variable(torch.FloatTensor(paths_feature)).cuda()  # paths * nodes * feature
            # y = Variable(torch.LongTensor(np.array([goals[2]]).astype(np.int64))).cuda()
            y = np.array([goals[2]])
            real_value_column.extend(y)
            z_st, temp_feature1, temp_feature2 = dl.add_feature_goal_node(int(goals[0]), int(goals[1]),
                                                                          Sd_associated, Sm, Sd)

            z_st = Variable(torch.FloatTensor([z_st])).cuda()  # [m_m,m_d,d_m,d_d] 816 * 2
            # print(z_st.size())
            z_element = Variable(torch.FloatTensor(np.array([temp_feature1]) * np.array([temp_feature2]))).cuda()

            # print(np.shape(z_element))
            # print(z_element.size())
            # z_st [torch.FloatTensor of size 1x1632]
            x_packed = nn_utils.rnn.pack_padded_sequence(x, paths_length, batch_first=True)

            # out = the_model(x_packed, z_st)
            test_out = the_model(x_packed, z_st, z_element)

            pred_one_probability = F.softmax(test_out, dim=1)[:, 1].data.squeeze().cpu().numpy()
            predict_value = torch.max(F.softmax(test_out, dim=1), 1)[
                1].data.squeeze().int().cpu().numpy()  # the index of the high probability, predicted value
            pred_column_probability.extend([pred_one_probability])
            predict_column_value.extend([predict_value])

        rvalue_probability_pvalue.append(real_value_column)
        rvalue_probability_pvalue.append(pred_column_probability)
        rvalue_probability_pvalue.append(predict_column_value)

        # according column to their initial column, and compute every column samples and its positive samples
        length_test = nodes_length

        count_positive = np.array([0,0])  # [[count_column_number,count_positive_number] [] ...]
        count_positive[0] = length_test
        for j in range(length_test):
            #print(real_value_column[j])
            if(real_value_column[j]==1):
                count_positive[1] += 1
        #print(count_have_positive)
        #if the current column of test data no positive,so continue (get rid of train 0,1 )
        if(count_positive[1]==0):
            count_have_positive -= 1
            continue

        rvalue_probability_pvalue = np.array(rvalue_probability_pvalue).transpose()
        # else
        # sort
        temp_test_data = sorted(rvalue_probability_pvalue, key=itemgetter(1), reverse=True)
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
        for x in range(all_number):
            if (temp_test_data[x][0] == 1):
                TP += 1
                FN = count_positive[1] - TP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                #print(x_TPR_temp)
                FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative - FP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                FPR_temp.append(FPR)
        Y_TPR_all_columns.append(TPR_temp)
        X_FPR_all_columns.append(FPR_temp)
        area_column = area_ROC(TPR_temp, FPR_temp)

        #print("该次最终roc面积", area_column)
        #plt.plot(FPR_temp, TPR_temp)
        #plt.show()
        area += area_column
    average_column_area = area / count_have_positive
    return X_FPR_all_columns,Y_TPR_all_columns,average_column_area

def draw_roc_column(cnn,columns,column_test_data,pt_feature):

    area = 0.0
    count_have_positive = columns
    Y_TPR_all_columns = []
    X_FPR_all_columns = []
    area_all_column = []
    # according to column
    for i in range(columns):
        test_x, test_y = get_test_data(column_test_data[i],pt_feature)
        test_out = cnn(test_x)

        pred_one_probability = F.softmax(test_out, dim=1)[:, 1].data.squeeze().cpu().numpy()

        predict_value = torch.max(F.softmax(test_out, dim=1), 1)[1].data.squeeze().int().cpu().numpy()
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
        for x in range(all_number):
            if (temp_test_data[x].value == 1 ):
                TP += 1
                FN = count_positive[1] - TP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                #print(x_TPR_temp)
                FPR_temp.append(FPR)
            else:
                FP += 1
                TN = count_negative - FP
                TPR = TP / count_positive[1]
                FPR = FP / count_negative
                TPR_temp.append(TPR)
                FPR_temp.append(FPR)
        area_column = area_ROC(TPR_temp, FPR_temp)
        Y_TPR_all_columns.append(TPR_temp)
        X_FPR_all_columns.append(FPR_temp)
        area_all_column.extend([area_column])
        #print("该次最终roc面积", area_column)
        #plt.plot(FPR_temp, TPR_temp)
        #plt.show()
        area += area_column
    #print("----")
    #print(len(X_FPR_all_columns))
    #print(count_have_positive)
    average_column_area = area/count_have_positive

    return Y_TPR_all_columns,X_FPR_all_columns,area_all_column,average_column_area

#get same point of TPR,FPR in columns,not contain all 0 column
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


def area_ROC(TPR,FPR):
    point_number = len(TPR)  # Y_FPR is the same with X_TPR
    area = 0.0
    for i in range(point_number-1):
        area += (FPR[i+1] - FPR[i]) * TPR[i+1]
    return area


def get_current_AUC_ROC(cnn,test_data,columns,pt_feature,time):


    #cnn.eval()
    length_test = len(test_data)
    columns = columns
    all_column_test_data = [[] for row in range(columns)]  # create a  list [[],[]...]  rows
    for i in range(length_test):
        all_column_test_data[test_data[i].index_y].append(test_data[i])  # ------pay attention append or extend
    # according to different column
    Y_TPR_all_columns, X_FPR_all_columns, area_all_column,average_column_area = draw_roc_column(cnn,columns, all_column_test_data,pt_feature)
    x_all_FPR, y_all_TPR = gsp(Y_TPR_all_columns,X_FPR_all_columns)
    x_FPR = np.sum(x_all_FPR, axis=0) / len(X_FPR_all_columns)
    y_TPR = np.sum(y_all_TPR, axis=0) / len(Y_TPR_all_columns)
    #print('Time:{}, AUC:{}'.format(time, average_column_area))
    #plt.plot(x_FPR, y_TPR)
    #plt.show()
    return x_FPR,y_TPR,average_column_area


def get_all_k_ROC(X_all_k_FPR,Y_all_k_TPR,x_all_FPR,y_all_TPR,k):

    min_number = len(X_all_k_FPR[0])

    y_TPR = []
    x_FPR = []
    for j in range(min_number):
        current_length = len(x_all_FPR)
        division_result = current_length / (min_number)
        y_TPR.append(y_all_TPR[round(division_result * (j+1))-1])
        x_FPR.append(x_all_FPR[round(division_result * (j+1))-1])
    X_all_k_FPR.append(y_TPR)
    Y_all_k_TPR.append(x_FPR)
    # FPR = np.sum(X_all_k_FPR,axis=0)/(min_number*k)
    # TPR = np.sum(Y_all_k_TPR,axis=0)/(min_number*k)
    print(np.shape(x_all_FPR))
    FPR = np.sum(X_all_k_FPR, axis=0)/k
    TPR = np.sum(Y_all_k_TPR,axis=0)/k
    plt.plot(FPR, TPR)
    plt.show()


# validation
def validation_test(alpha, k, max_count,batchsize,EPOCH,LR,dev_batchsize):
    all_k_Sd_associated, all_k_count_positive, all_k_test_data, A, num,\
    A_length, sava_association_A, all_k_development, k_pt_features,Sm, Sd = data_partitioning(alpha, k, max_count)
    X_all_k_FPR = []
    Y_all_k_TPR = []
    area_all_k = []
    for i in range(k):
        train_loader = L8.load_data1(all_k_Sd_associated[i],k_pt_features[i], BATCH_SIZE=batchsize)
        # test_x, test_y = get_test_data(all_k_validation[0][:])
        dev_loader = L8.load_data1(all_k_development[i],k_pt_features[i], BATCH_SIZE=dev_batchsize)
        L8.train2(EPOCH, train_loader, dev_loader, LR)


        cnn.eval()
        test_data = all_k_test_data[i]
        x_all_FPR, y_all_TPR,current_k_area = get_current_AUC_ROC(test_data,len(A[0]),k_pt_features[i],i)  # all column
        if(i!= k-1):
            X_all_k_FPR.append(x_all_FPR)
            Y_all_k_TPR.append(y_all_TPR)
        area_all_k.extend([current_k_area])
    get_all_k_ROC(X_all_k_FPR,Y_all_k_TPR,x_all_FPR,y_all_TPR,k)
    print("K Avreage AUC", np.mean(area_all_k))
    pass