# 0 import
import pandas as pd
import numpy as np
import random
import copy

random.seed(123)

#1 read data file
def read_data_flies():
    #  490 * 326  miRNA * diseases
    A = np.loadtxt("./A_5cross_All_space.txt")  # type umpy.ndarray

    #  490 * 490  miRNA * miRNA
    Sm = np.loadtxt("./SM_all_5cross_tab.txt")

    #  326 * 326  diseases * diseases
    Sd = np.loadtxt("./SD_5cross_tab.txt")
    print("---1.read data files---")
    return A,Sm,Sd

#2 class node
class node():
    def __init__(self, value, x, y):
        self.value = value  # value 0 or 1
        self.index_x = x  # the row in A_association   A(x,y)
        self.index_y = y  # the column in A_association A(x,y)
        #self.miRNA = 0
        #self.disease = 0
        #self.predict_probability = 0.0                  #  row-mi  coloumn(mj+dj)  or row-di  coloumn(mj+dj)
        #self.predict_value = -1
    def judge_RNA_disease(self,flag):
        if(flag == 1):
            self.miRNA = 1
        else:
            self.disease =1
    def add_association(self,m_d_association):
        self.m_d_association = m_d_association  # 1 * 490 + 1 * 326 = 1 * 816     1 * (miRNA + disease)
                                                #  row-mi  coloumn(mj+dj)  or row-di  coloumn(mj+dj)
    def add_predict_probability_value(self,predict_probability,predict_value):
        self.predict_probability = predict_probability
        self.predict_value = predict_value

#3 class matrix_A for its value and index
class value_index():
    def __init__(self,num,i,j):
        self.value = num  # save value
        self.value_x = i  # save index of row
        self.value_y = j  # save index of column

#4 cross validation
def k_crossvalidation(matrix_A,k):
    # define save
    sava_association_A = []# save association  diseases and drugs
    save_all_count_A = []# save k time changed A
    save_all_count_zero_not_changed = []  # all zero, but not contain the number which should changed ( 1 to 0 )
    save_all_count_zero_every_time_changed = []  # save k * number which should changed ( 1 to 0 )
    save_count_zero = []  # record current zero and its location

    # if 1 save in sava_association_A, else in save_count_zero
    for i in range(matrix_A.shape[0]): # row
        for j in range(matrix_A.shape[1]): # column
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
        temp_count_zero = []  # record current the changed number and location
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
    print("---2.k validation parts---")
    return save_all_count_A,sava_association_A,save_all_count_zero_not_changed,save_all_count_zero_every_time_changed,num,A_length
    """
    #save_all_count_A   save all k input
    #sava_association_A save shuffle data (value_index(1,i,j))
    #save_all_count_zero_not_changed all zero, but no  number which should changed ( 1 to 0 )
    #save_all_count_zero_every_time_changed save k * number which should changed ( 1 to 0 )
    #num    int(A_length / k)
    #A_length   len(sava_association_A)
    """

#5 devide data into train_part:development_part:test_part   e.g. 7:2:1
def get_pre_train_dev_test_set(k_fold):
#  k-fold
    k = k_fold
#  for k = 10, previous code is 9 part-1 as train set, 1 part-1 as test set
#  so, now we devide the previous train set(9 part-1) into 7 part-1 train set and 2 part-1 development set
    train_part = 7
    development_part = 2
#   read files
    A, Sm, Sd = read_data_flies()
#  get all k-fold train data
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = k_crossvalidation(A,k)
#   devide data into train_part:development_part:test_part
    all_k_train_set = []  # save all k train sets
    all_k_count_positive = []  # save all  k positive length  (every time number of positive samples)
    all_k_development_set = []  # save all k development sets
    all_k_test_set = [] # save all k test sets
    for x in range(k):
        save_count_zero = copy.deepcopy(save_all_count_zero_not_changed)  # temp save zero
        k_Sd_samples = []  # save  samples which conclude positive and negative ---save train data
        k_Sd_samples_development = []  # save development data
        column_length = len(save_all_count_A[x][0])  # the length of column    diseases 326
        row_length = len(save_all_count_A[x])  # the length of row        miRNA 429

    # just positive sample
        k_Sd_associated = []  # save 1 * 326  disease * diseases
        count_positive = 0
        for j in range(column_length):
            for i in range(row_length):
                if(save_all_count_A[x][i][j] == 1):  # use new associated_A get positive
                    count_positive += 1
                    # because save_all_count_A come from matrix A, so location i and j not change ,so could use i ,j
                    data = node(1,i, j)
                    k_Sd_associated.append(data)  # save a certain disease  with certain miRNA  label is 1
    # shuffle before choose negative samples
        random.shuffle(save_count_zero)
    # choose negative sample
        k_Sd_negative = []
        count_negative = count_positive
        # get frontal count_positive number because shuffle before taking
        for i in range(count_negative):
            k_Sd_negative.append(node(0,save_count_zero[i].value_x,save_count_zero[i].value_y))
        """
        #for test data to delete negative 0
        # del frontal count_positive number
        count = 0
        while(count < count_negative):
            del save_count_zero[0]
            count += 1
        """
    # combine positive and negative， but divide into train sets and development sets
        # first shuffle and then divide
        random.shuffle(k_Sd_associated)
        random.shuffle(k_Sd_negative)
        # length_dividing_point of train and validation
        length_dividing_point = int(len(k_Sd_associated) / (train_part+development_part) * train_part)

        # train sets
        k_Sd_samples.extend(k_Sd_associated[:length_dividing_point])
        k_Sd_samples.extend(k_Sd_negative[:length_dividing_point])  # save in a raw
        random.shuffle(k_Sd_samples)
        all_k_train_set.append(k_Sd_samples)
        all_k_count_positive.append(count_positive)  #  k time number of positive

        # development sets
        k_Sd_samples_development.extend(k_Sd_associated[length_dividing_point:])
        k_Sd_samples_development.extend(k_Sd_negative[length_dividing_point:])
        random.shuffle(k_Sd_samples_development)
        all_k_development_set.append(k_Sd_samples_development)

        # get test data-----there contain all 0
        save_all_count_zero = []  # 0 + previous 1 to 0
        save_all_count_zero.extend(save_count_zero)
        save_all_count_zero.extend(save_all_count_zero_every_time_changed[x])
        random.shuffle(save_all_count_zero)  # shuffle all data [ 0 and  (1 to 0) ]
        length = len(save_all_count_zero)
        temp_save = []
        for i in range(length):
            temp_save.append(
                node(save_all_count_zero[i].value, save_all_count_zero[i].value_x, save_all_count_zero[i].value_y))
        all_k_test_set.append(temp_save)

        """
    # get test data-----there no negative 0
        save_all_count_zero = []  # 0 + previous 1 to 0
        save_all_count_zero.extend(save_count_zero)
        save_all_count_zero.extend(save_all_count_zero_every_time_changed[x])
        length = len(save_all_count_zero)
        #print('test data number')
        #print(length)
        random.shuffle(save_all_count_zero)  # shuffle all data [ 0 and  (1 to 0) ]
        temp_save = []
        for i in range(length):
            temp_save.append(data_input(save_all_count_zero[i].value,save_all_count_zero[i].value_x,save_all_count_zero[i].value_y,Sd[save_all_count_zero[i].value_y],Sm[save_all_count_zero[i].value_x],save_all_count_A[x][:,save_all_count_zero[i].value_y],np.concatenate((save_all_count_A[x][save_all_count_zero[i].value_x],a))))
        all_k_test_set.append(temp_save)
        """
    print("---3.devide data into train,development,test---")
    return all_k_train_set,all_k_development_set,all_k_test_set,all_k_count_positive,A,num, A_length,sava_association_A, Sm, Sd,save_all_count_A
    """
    # k * ( count_positive + count_negative) =k * ( count_positive + count_positive)
      in every count_positive, is a class                    
                            class data_input(value # value 0 or 1
                            x  # the row in A_association   A(x,y)
                            y  # the column in A_association A(x,y)
                            d_d_association  # save 1 * 326  disease * diseases
                            m_m_association  # save 1 * 190  miRNA * miRNAs  )
     save_all_count_A   k time temp_A
    """


if __name__ == '__main__':

    # define parameters
    k = 10
    """
    # execute
    print("---1.read data files---")
    A, Sm, Sd = read_data_flies()
    print("---2.k validation parts---")
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length = k_crossvalidation(A,k)
    """
    #A, Sm, Sd = read_data_flies()
    all_k_train_set, all_k_development_set, all_k_test_set, all_k_count_positive, A, num, A_length, sava_association_A, Sm, Sd,save_all_count_A = get_pre_train_dev_test_set(k)
    np.savez('./get_data/data_prepare/devided_data',all_k_train_set=all_k_train_set,all_k_development_set=all_k_development_set,
             all_k_test_set=all_k_test_set,save_all_count_A=save_all_count_A)
    #print(len((all_k_train_set[0])))#7124
    #print(len(all_k_development_set[0]))#2036
    #print(len(all_k_test_set[0]))#155160

