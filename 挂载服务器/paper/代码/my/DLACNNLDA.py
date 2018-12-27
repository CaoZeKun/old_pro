"""
### 之前的版本 之前版本最后设置了超参，其实可以手动或者删去--这里先删去
### 交叉验证式，lncRNA 相似性应该再算
   """


import aNewMethodLDAP_vector as ANML
# import Attention_cnn as DL
import LDAP
import MFLDA_init_all_data as MFLDA
import SIMCLDA_2_innerproduct as SIMC
import random
import numpy as np
import copy
from operator import attrgetter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
random.seed(1)
torch.backends.cudnn.enabled = False


# --- lncRNA similarity
def get_index(lncrna):
    index = []
    for i in range(len(lncrna)):
        if lncrna[i] == 1 :
            index.append(i)

    return index


def get_sim(l1_disease,l2_diseases,disease_sim):
    dis_sims = []
    for i in range(len(l2_diseases)):
        dis_sims.append([disease_sim[l1_disease,l2_diseases[i]]])
    return np.max(dis_sims)

def get_sim_all(l1_diseases,l2_diseases,disease_sim):
    diss_sims1 = []
    for i in range(len(l1_diseases)):
        dis_sim_temp = get_sim(l1_diseases[i],l2_diseases,disease_sim)
        diss_sims1.append(dis_sim_temp)
    l1_dis = np.sum(diss_sims1)

    diss_sims2 = []
    for i in range(len(l2_diseases)):
        dis_sim_temp = get_sim(l2_diseases[i], l1_diseases,disease_sim)
        diss_sims2.append(dis_sim_temp)
    l2_dis = np.sum(diss_sims2)

    l_dis_sim = np.sum((l1_dis,l2_dis))
    return l_dis_sim


def get_sim_value(lncrna1,lncrna2,disease_sim):  # ln1 = A[0]

    lnc1_dis_index = get_index(lncrna1)
    lnc2_dis_index = get_index(lncrna2)


    lnc1_dis_len = len(lnc1_dis_index)
    lnc2_dis_len = len(lnc2_dis_index)
    length_l1_l2 = lnc1_dis_len + lnc2_dis_len
    if lnc1_dis_len==0 or lnc2_dis_len==0:
        sim = 0
    else:
        l_dis_sim = get_sim_all(lnc1_dis_index,lnc2_dis_index,disease_sim)
        sim = l_dis_sim / (length_l1_l2)
    return sim


def get_all_sim1(lnc_dis,dis_sim):
    lnc_length = len(lnc_dis)
    lnc_sim = np.zeros((lnc_length,lnc_length))
    print(np.shape(lnc_sim))

    for i in range(lnc_length):
        for j in range(lnc_length):
            sim = get_sim_value(lnc_dis[i],lnc_dis[j],dis_sim)
            lnc_sim[i,j] = sim
    for i in range(lnc_length):
        lnc_sim[i, i] = 1
    return lnc_sim

# --- lncRNA similarity



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
# define a class
class value_index():
    def __init__(self,num,i,j):
        self.value = num  # save value
        self.value_x = i  # save index of row
        self.value_y = j  # save index of column

# define input data
class data_input():
    def __init__(self, value, x, y):
        self.value = value # value 0 or 1
        self.index_x = x  # the row in A_association   A(x,y)
        self.index_y = y  # the column in A_association A(x,y)
        # self.d_d_association = d_d_association  # save 1 * 326  disease * diseases
        # self.m_m_association = m_m_association  # save 1 * 190  miRNA * miRNAs
        # self.d_m_association = d_m_association  # 1 * 490  disease * miRNAs
        # self.m_d_association = m_d_association  # 1 * 326 miRNA * disease
        # self.probability = 0.0
        # self.predict_value = -1
    def add_probability_predict(self,probability,predict_value):
        self.probability = probability
        self.predict_value = predict_value

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
def data_partitioning(matrix_A,k):
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length  = crossvalidation(matrix_A,k)
    all_k_Sd_associated = []  # save all k * 1 * 326   k * disease * diseases   if(A(i,j) =1)
    all_k_count_positive = []  # save all  k positive length  (every time number of positive samples)
    all_k_development = []  # save all k development sets
    all_k_test_data = []  # save all k test data
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
                if (save_all_count_A[x][i][j] == 1):  # use new associated_A get positive
                    count_positive += 1
                    # because save_all_count_A come from matrix A, so location i and j not change ,so could use i ,j
                    data = data_input(1, i, j)
                    k_Sd_associated.append(data)  # save a certain disease  with certain miRNA  label is 1
                    # print('positive number')
                    # print(count_positive)
                    # shuffle before choose negative samples
        random.shuffle(save_count_zero)
        # choose negative sample

        k_zeros = []
        k_Sd_negative = []
        count_negative = count_positive
        # get frontal count_positive number because shuffle before taking
        for i in range(count_negative):
            k_Sd_negative.append(
                data_input(0, save_count_zero[i].value_x, save_count_zero[i].value_y ))

        # del frontal count_positive number
        count = 0
        while (count < count_negative):
            del save_count_zero[0]
            count += 1
            # combine positive and negative， but divide into train sets and development sets
        # first shuffle and then divide
        random.shuffle(k_Sd_associated)
        random.shuffle(k_Sd_negative)
        # length_dividing_point of train and validation
        length_dividing_point = int(len(k_Sd_associated) / 8 * 7)

        # train sets
        k_Sd_samples.extend(k_Sd_associated[:length_dividing_point])
        k_Sd_samples.extend(k_Sd_negative[:length_dividing_point])  # save in a raw
        random.shuffle(k_Sd_samples)
        all_k_Sd_associated.append(k_Sd_samples)
        all_k_count_positive.append(count_positive)

        # development sets
        k_Sd_samples_development.extend(k_Sd_associated[length_dividing_point:])
        k_Sd_samples_development.extend(k_Sd_negative[length_dividing_point:])
        random.shuffle(k_Sd_samples_development)
        all_k_development.append(k_Sd_samples_development)

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
                data_input(save_all_count_zero[i].value, save_all_count_zero[i].value_x, save_all_count_zero[i].value_y))
        all_k_test_data.append(temp_save)

    return all_k_Sd_associated, all_k_count_positive, all_k_test_data,  num, A_length, sava_association_A, all_k_development,save_all_count_A


# data load,i--miRna  j--disease, use pt[i] and pt[:,490+j] as feature
#    240*240      240*405           240 * 495
# [[lncRna*lncRna] [A(lncRna*Disease)] [lncRna * miRna]
#  [A.T(Disease*lncRna)] [Disease*Disease] [miRna * Disease]]
#     405*240            405*405          495 * 405





def get_test_data(data,A,Rna_matrix,disease_matrix,lnc_mi,mi_dis):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        x_A = data[j].index_x
        y_A = data[j].index_y
        rna_disease_mi = np.concatenate((Rna_matrix[x_A], A[x_A],lnc_mi[x_A]), axis=0)
        disease_rna_mi = np.concatenate((disease_matrix[y_A], A[:, y_A],mi_dis[:,y_A]), axis=0)
        temp_save.append(rna_disease_mi)
        temp_save.append(disease_rna_mi)
        x.append([temp_save])
        y.append(data[j].value)

    #test_x = Variable(torch.FloatTensor(np.array(x)))
    #test_y = torch.IntTensor(np.array(y))
    test_x = Variable(torch.FloatTensor(np.array(x))).cuda()
    test_y = torch.IntTensor(np.array(y)).cuda()
    return test_x, test_y
# data load,i--miRna  j--disease,
# use       pt[i]
#       pt[:,490+j]
#     miRna matrix[i] A[i]
#     disease matrix[j]  A[:,j]
# as feature
def load_data2(data,A,Rna_matrix,disease_matrix,BATCH_SIZE,lnc_mi,mi_dis,drop = False):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        x_A = data[j].index_x
        y_A = data[j].index_y
        rna_disease_mi = np.concatenate((Rna_matrix[x_A],A[x_A],lnc_mi[x_A]),axis=0)
        disease_rna_mi = np.concatenate((disease_matrix[y_A],A[:,y_A],mi_dis[:,y_A]),axis=0)
        temp_save.append(rna_disease_mi)
        temp_save.append(disease_rna_mi)

        x.append([temp_save])
        y.append(data[j].value)
    x = torch.FloatTensor(np.array(x))
    #print(x.size())
    y = torch.LongTensor(np.array(y))
    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=False,              # random shuffle for training
            num_workers=1,              # subprocesses for loading data
            drop_last= drop
        )
    return data2_loader



class Attention2(nn.Module):
    def __init__(self,input_size,output_size):
        super(Attention2,self).__init__()

        self.node = nn.Linear(input_size, output_size,bias=True) #input_size * 2   + zs_size(hidden * 2)
        nn.init.xavier_normal_(self.node.weight)

        #self.ZS = nn.Linear(zs_size, output_size, bias=False)  # input_size * 2   + zs_size(hidden * 2)
        #nn.init.xavier_normal_(self.ZS.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,input_size))
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, h_n_states):
        # z_h = torch.cat([h_n_states, torch.cat([z for _ in range(h_n_states.size(0))], 0)],1) #torch.Size([4, 1832])
        #print(zs)
        #print(zs.size())
        temp_nodes = self.node(h_n_states) # 4 * (input_size *2) ,4 * output_size
        temp_nodes = F.tanh(temp_nodes)
        # print(temp_nodes.size())  # torch.Size([50, 1, 100])
        #temp_zs = self.ZS(z)
        #temp_zs = F.tanh(temp_zs)
        #temp = torch.cat((temp_nodes,temp_zs),0)
        nodes_score = torch.matmul(temp_nodes, self.h_n_parameters)
        # nodes_score = torch.bmm(temp_nodes,self.h_n_parameters)
        # print(nodes_score.size())  torch.Size([50, 1, 240])
        #nodes_score = torch.t(nodes_score)  # 1 * 4
        #print(nodes_score)
        #print(nodes_score[0])
        alpha = F.softmax(nodes_score,dim=2)

        #print(h_n_states.size())  #
        # print(alpha.size())  # torch.Size([50, 1, 240])
        # print("************")
        # print(alpha[0][0][1])
        # print(h_n_states[0][0][1])
        y_i = alpha * h_n_states
        # print(y_i[0][0][1])
        # print(alpha[0][0][1] * h_n_states[0][0][1])
        # print("-----------")
        # print(y_i.size())  # torch.Size([50, 1, 240])
        return y_i


class Beta_score2(nn.Module):
    def __init__(self,input_size_lnc,input_size_A,input_size_lncmi,input_size_AT,input_size_dis,input_size_midis, output_size,batch_size):
        super(Beta_score2,self).__init__()

        self.node1 = nn.Linear(input_size_lnc+input_size_A+input_size_lncmi, output_size,bias=True) #input_size * 2 + zs_size(hidden * 2)
        nn.init.xavier_normal_(self.node1.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,1))
        nn.init.xavier_normal_(self.h_n_parameters)

    def forward(self, result_ls,result_A,result_lm,result_AT,result_ds,result_dm):
        length = result_ls.size()[2] + result_A.size()[2] + result_lm.size()[2]
        # print(length)
        batch_size_ = result_ls.size()[0]
        #z_h = torch.cat([y_i, torch.cat([z for _ in range(y_i.size(0))], 0)], 1)
        ls_pad = Variable(torch.zeros(batch_size_,result_ls.size()[1], length-result_ls.size()[2])).cuda()
        ls_pad = torch.cat((result_ls,ls_pad),dim=2)  # [50, 1, 1140]

        result_A_pad = Variable(torch.zeros(batch_size_, result_A.size()[1], length - result_A.size()[2])).cuda()
        result_A_pad = torch.cat((result_A, result_A_pad), dim=2)

        result_lm_pad = Variable(torch.zeros(batch_size_, result_lm.size()[1], length - result_lm.size()[2])).cuda()
        result_lm_pad = torch.cat((result_lm, result_lm_pad), dim=2)

        result_AT_pad = Variable(torch.zeros(batch_size_, result_AT.size()[1], length - result_AT.size()[2])).cuda()
        result_AT_pad = torch.cat((result_AT, result_AT_pad), dim=2)

        result_ds_pad =Variable(torch.zeros(batch_size_, result_ds.size()[1], length - result_ds.size()[2])).cuda()
        result_ds_pad = torch.cat((result_ds, result_ds_pad), dim=2)

        result_dm_pad = Variable(torch.zeros(batch_size_, result_dm.size()[1], length - result_dm.size()[2])).cuda()
        result_dm_pad = torch.cat((result_dm, result_dm_pad), dim=2)


        reslut = torch.cat((ls_pad,result_A_pad),dim=1)
        reslut = torch.cat((reslut, result_lm_pad), dim=1)
        reslut = torch.cat((reslut, result_AT_pad), dim=1)
        reslut = torch.cat((reslut, result_ds_pad), dim=1)
        reslut = torch.cat((reslut, result_dm_pad), dim=1)
        # print(reslut.size())  # [50, 6, 1140]

        temp_nodes = self.node1(reslut) # 103519 * (input_size) ,103519 * output_size
        temp_nodes = F.tanh(temp_nodes)  # 50, 6, 100
        # print(temp_nodes.size())
        nodes_score = torch.matmul(temp_nodes,self.h_n_parameters)  # [50, 6, 100] * [100, 1] = [50, 6, 1]
        # print(nodes_score.size())

        nodes_score = nodes_score.view(-1,1,6)  # [50, 1, 6]
        # print(nodes_score.size())
        # print(nodes_score.size())
        # print((self.gamma * num).size())
        # print(nodes_score)
        # print(self.gamma * num)
        # nodes_score = nodes_score - (self.gamma * num)
        # print(nodes_score)
        # nodes_score = torch.t(nodes_score)  # 1 * 103519
        beta = F.softmax(nodes_score,dim=2)  # 50 * 1 * 6
        # beta1 = beta[:,:,0]
        # print(beta1[0])
        # print(beta1[1])
        # print(beta.size())  # [50, 1]
        # print(result_ls.size())  # [50, 1, 240]
        # print(result_ls[0][0][2])
        # print(result_ls[1][0][2])
        # result_ls = beta1 * result_ls
        # print(result_ls[0][0][2])
        # print(result_ls[1][0][2])


        # result_A
        # result_lm
        # result_AT
        # result_ds
        # result_dm

        # print(result_A.size())
        #
        #
        # print(beta.size())
        # print(beta[0])
        # print(torch.sum(beta[0]))
        z_i = torch.matmul(beta, reslut)  # 50 * 1 * 6, 50 * 6 * 1140 , 50 * 1 * 1140
        # print(beta)

        return z_i



class the_model(nn.Module):
    def __init__(self,input_size_lnc,input_size_A,input_size_lncmi,input_size_dis,input_size_midis,output_size,batch_size):
        super(the_model, self).__init__()
        self.attention_ls = Attention2(input_size_lnc,output_size)
        self.attention_A = Attention2(input_size_A, output_size)
        self.attention_lm = Attention2(input_size_lncmi, output_size)
        self.attention_AT = Attention2(input_size_lnc, output_size)
        self.attention_ds = Attention2(input_size_dis, output_size)
        self.attention_dm = Attention2(input_size_midis, output_size)

        self.attention_soures = Beta_score2(input_size_lnc,input_size_A,input_size_lncmi,input_size_lnc,input_size_dis,input_size_midis, output_size,batch_size)

        self.cnn_att = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,  kernel_size=2,  stride=1,  padding=1),
            #nn.Dropout(0.5),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            ) # may 16 * 1 * 1140
        self.cnn1 = CNN()
        # self.w_parameters = nn.Parameter(torch.FloatTensor([1,1]))
        self.fully_connected = nn.Sequential(nn.Linear(16*1*1140+32*2*1140, 2), nn.BatchNorm1d(2))
        # self.fully_connected = nn.Sequential(nn.Linear(16*1*1140+32*2*1140, 2), nn.BatchNorm1d(2), nn.Sigmoid())


    def forward(self, x):
        # print(x.size()[0]) 50
        # print(x.size()[1]) 1
        # print(x.size()[2]) 2
        # print(x.size()[3]) 1140
        # print(x.size())  # [50, 1, 2, 1140]
        #    240*240      240*405           240 * 495
        #     405*240            405*405          495 * 405
        ls = x[:,:,0,:240]  # 240
        A = x[:,:,0,240:645]  # 405
        lm = x[:,:,0,645:1140]  # 495

        AT = x[:,:,1,:240]  # 240
        ds = x[:,:,1,240:645]  # 405
        dm = x[:,:,1,645:1140]  # 495

        result_ls = self.attention_ls(ls)
        result_A = self.attention_A(A)
        result_lm = self.attention_lm(lm)

        result_AT = self.attention_AT(AT)
        result_ds = self.attention_ds(ds)
        result_dm = self.attention_dm(dm)
        # print(result_ls.size())  # [50, 1, 240]

        #  = torch.cat((all_y_i, y_i), 0)
        z_i = self.attention_soures(result_ls,result_A,result_lm,result_AT,result_ds,result_dm)  # 50 * 1 * 1140
        z_i = z_i.view(x.size()[0],1,1,1140)
        # print(z_i.size())
        f_c_att = self.cnn_att(z_i).view(x.size()[0], -1)
        f_c_m = self.cnn1(x)
        #print(self.w_parameters)
        #f_c_att = self.w_parameters[0] * f_c_att
        #f_c_m = self.w_parameters[1] * f_c_m
        # print(f_c_att.size())
        # print(f_c_m.size())
        f_c = torch.cat((f_c_att,f_c_m),dim=1)
        # print(f_c.size())
        out = self.fully_connected(f_c)
        out = F.softmax(out,dim=1)


        return out







































# CNN Framwork
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # now 1, 2, 645
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # padding
            ),  # may 16 * 3 * 646
            #nn.Dropout(0.5),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),  # may 16 * 2 * 645


        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 2, 1, 1),  # may 32 * 3 * 646
            #nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 0),  # may 32 * 2 * 645
            #nn.BatchNorm2d(32)
            #nn.Dropout(0.2)
        )

        #self.out = nn.Sequential(nn.Linear(32 * 1 * 812,2),nn.BatchNorm1d(2),nn.Dropout(0.5),nn.Sigmoid())
        #self.out = nn.Sequential(nn.Linear(32 * 2 * 1140, 2), nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Sigmoid())
        #self.out = nn.Linear(32 * 2 * 906, 2)


        #self.softmax = torch.nn.Softmax()
    def forward(self,x):
        #print(x.size())
        x = self.conv1(x)  # batch_size(50) * 16 * 4 * 492
        #print(x.size())
        #x = self.dropout(x)
        x = self.conv2(x)  # may
        # print(x.size())
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #out = self.out(x)
        #print(torch.is_tensor(x))
        #out = torch.FloatTensor(out)
        #out = self.softmax(out)
        #print(out.size())
        return x

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # now 1, 2, 511
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=(2,3),  # filter size
                stride=(1,3),  # filter movement/step
                padding=1,  # padding
            ),  # may 16 * 3 * 171
            #nn.Dropout(0.5),
            nn.BatchNorm2d(16),
            nn.ReLU(),  # activation
            nn.AvgPool2d(kernel_size=2, stride=1, padding=0),  # may 16 * 2 * 170


        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32,(1,3), 1, 0),  # may 32 * 1 * 168
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((1,2), 2, 0),  # may 32 * 1 * 84
            #nn.BatchNorm2d(32)
            #nn.Dropout(0.2)
        )

        #self.out = nn.Sequential(nn.Linear(32 * 1 * 812,2),nn.BatchNorm1d(2),nn.Dropout(0.5),nn.Sigmoid())
        #self.out = nn.Sequential(nn.Linear(32 * 2 * 511, 2), nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Sigmoid())
        #self.out = nn.Linear(32 * 2 * 906, 2)


        #self.softmax = torch.nn.Softmax()
    def forward(self,x):
        x = self.conv1(x)  # batch_size(50) * 16 * 4 * 492
        #print(x.size())
        #x = self.dropout(x)
        x = self.conv2(x)  # may
        #print(x.size())
        out = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        #out = self.out(x)

        #print(torch.is_tensor(x))
        #out = torch.FloatTensor(out)
        #out = self.softmax(out)
        #print(out.size())
        return out


class CNN0(nn.Module):
    def __init__(self):
        super(CNN0, self).__init__()
        self.cnn1 = CNN1()
        self.cnn2 = CNN2()
        self.h_n_parameters = nn.Parameter(torch.randn(1))
        self.out = nn.Linear(32 * 2 * 511+ 32 * 1 * 84, 2)
        #self.out = nn.Linear(32 * 2 * 906, 2)

        #self.softmax = torch.nn.Softmax()
    def forward(self,x):
        x1 = self.cnn1(x)

        x2 = self.cnn2(x)
        x2 = torch.mul(x2,self.h_n_parameters)

        x = torch.cat((x1,x2),1)
        out = self.out(x)

        return out





def train3(model,EPOCH,train_loader,test_loader,LR):

    optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-7)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    #exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(EPOCH):
        #scheduler.step()
        for step, (x, y) in enumerate(train_loader):
            model.train()
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            #b_x = Variable(x)
            #b_y = Variable(y)
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #exp_lr_scheduler.step()
            # if step % 500 == 0:
            #     model.eval()
            #     same_number = 0.0
            #     same_number_length = 0.0
            #     for _, (xx, yy) in enumerate(test_loader):
            #         dev_x = Variable(xx).cuda()
            #         dev_y = yy.cuda().int()
            #         # print(yy.size(0))
            #         # print(torch.typename(yy))
            #         # print(torch.typename(yy.size))
            #         # print(torch.typename(yy.size(0)))
            #         # print(dev_x.size())
            #         dev_output = model(dev_x)
            #         pred_y = torch.max(dev_output, 1)[1].data.squeeze().int()
            #         #print(torch.typename(pred_y))
            #         same_number += sum(np.array(pred_y) == np.array(dev_y))
            #         # print(dev_y.size(0))
            #         # print(torch.typename(dev_y))
            #         # print(torch.typename(dev_y.size(0)))
            #         same_number_length += float(dev_y.size(0))
            #         #if epoch == 9:
            #             #print('This is predict', pred_y)
            #             #print('This is label', dev_y)
            #     same_number_length = float(same_number_length)
            #     same_number = float(same_number)
            #     accuracy = same_number / same_number_length
            #     #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
            #
            #     same_number1 = 0.0
            #     same_number_length1 = 0.0
            #     for _, (xx1, yy1) in enumerate(train_loader):
            #         train_x = Variable(xx1).cuda()
            #         train_y = yy1.cuda().int()
            #         # print(train_x.size())
            #         train_output = model(train_x)
            #
            #         pred_train_y = torch.max(train_output, 1)[1].data.squeeze().int()
            #         #print(torch.typename(pred_y))
            #         same_number1 += sum(np.array(pred_train_y) == np.array(train_y))
            #         same_number_length1 += train_y.size(0)
            #         #if epoch == 9:
            #             #print('This is predict', pred_y)
            #             #print('This is label', dev_y)
            #     same_number1 = float(same_number1)
            #     same_number_length1 = float(same_number_length1)
            #     accuracy1 = same_number1 / same_number_length1
            #
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| train accuracy: %.4f' % accuracy1,'| test accuracy: %.4f' % accuracy)


    # save_to_file2('cnnlncRNA.txt', we)


def train4(cnn,EPOCH,train_loader,test_loader,LR):

    optimizer = torch.optim.Adam(cnn.parameters(),lr=LR,weight_decay=1e-8)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_func = nn.CrossEntropyLoss()
    #exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(EPOCH):
        #scheduler.step()
        for step, (xx, yy) in enumerate(test_loader):
            dev_x = Variable(xx).cuda()
            dev_y = Variable(yy).cuda()
            output = cnn(dev_x)
            loss = loss_func(output, dev_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            #exp_lr_scheduler.step()
            if step % 500 == 0:
                cnn.eval()
                same_number = 0
                same_number_length = 0
                for _, (xx0, yy0) in enumerate(test_loader):
                    dev_x0 = Variable(xx0).cuda()
                    dev_y0 = yy0.cuda().int()
                    dev_output = cnn(dev_x0)
                    pred_y = torch.max(dev_output, 1)[1].data.squeeze().int()
                    #print(torch.typename(pred_y))
                    same_number += sum(pred_y == dev_y0)
                    same_number_length += dev_y0.size(0)
                    #if epoch == 9:
                        #print('This is predict', pred_y)
                        #print('This is label', dev_y)
                accuracy = same_number / float(same_number_length)
                #print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

                same_number1 = 0
                same_number_length1 = 0
                for _, (xx1, yy1) in enumerate(train_loader):
                    train_x = Variable(xx1).cuda()
                    train_y = yy1.cuda().int()
                    train_output = cnn(train_x)
                    pred_train_y = torch.max(train_output, 1)[1].data.squeeze().int()
                    #print(torch.typename(pred_y))
                    same_number1 += sum(pred_train_y == train_y)
                    same_number_length1 += train_y.size(0)
                    #if epoch == 9:
                        #print('This is predict', pred_y)
                        #print('This is label', dev_y)
                accuracy1 = same_number1 / float(same_number_length1)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| train accuracy: %.4f' % accuracy1,'| test accuracy: %.4f' % accuracy)


    # save_to_file2('cnnlncRNA.txt', we)

# train
def train2(model,Epoch,train_loader,dev_loader,current_optim,w_decay,_scheduler,_LR):

    if current_optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=_LR, weight_decay=w_decay)
    elif current_optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=_LR, weight_decay=w_decay)
    elif current_optim == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=_LR, weight_decay=w_decay)
    elif current_optim == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=_LR, weight_decay=w_decay)

    if _scheduler == 'lambda':
        lambda2 = lambda epoch: 0.90 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda2)
    elif _scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif _scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        scheduler = None
        # optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    loss_func = nn.CrossEntropyLoss()
    #exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(Epoch):
        if (scheduler != None):
            scheduler.step()
        for step, (x, y) in enumerate(train_loader):
            model.train()
            # length_train_loader = len(train_loader)
            b_x = Variable(x).cuda()
            b_y = Variable(y).cuda()
            # if torch.cuda.is_available():
            #     b_x = Variable(x).cuda()
            #     b_y = Variable(y).cuda()
            # else:
            #     b_x = Variable(x)
            #     b_y = Variable(y)

            output = model(b_x)  # FloatTensor of size 50x2
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if ((step+1) % length_train_loader == 0) and (epoch == Epoch-1):
            # #if ((step + 1) % length_train_loader == 0) :
            #     model.eval()
            #     same_number = 0
            #     same_number_length = 0
            #     for step, (xx, yy) in enumerate(dev_loader):
            #         dev_x = Variable(xx).cuda()
            #         dev_y = yy.cuda().int()
            #         dev_output = model(dev_x)
            #         pred_y = torch.max(dev_output, 1)[1].data.squeeze().int()
            #
            #         same_number += sum(pred_y == dev_y)
            #         same_number_length += dev_y.size(0)
            #
            #     accuracy = same_number / float(same_number_length)
            #     context ='Epoch: ', epoch, '| train loss: %.6f' % loss.data[0], '| dev accuracy: %.4f' % accuracy
            #     save_to_file2('rwcnntest.txt', str(context))


# ROC
# draw ROC
# def draw_roc_column1(cnn,columns,column_test_data,BATCH_SIZE,A,Sd,Sm):
#
#     area = 0.0
#     count_have_positive = columns
#     Y_TPR_all_columns = []
#     X_FPR_all_columns = []
#     area_all_column = []
#     # according to column
#     for i in range(columns):
#         test_loader = load_data2(column_test_data[i],A,Sd,Sm,BATCH_SIZE)
#         #test_out = torch.FloatTensor()
#         #cnn.eval()
#         pred_probability = []
#         predict_values = []
#         for step, (x, y) in enumerate(test_loader):
#             if torch.cuda.is_available():
#                 b_x = Variable(x).cuda()
#                 #b_y = Variable(y).cuda()
#             else:
#                 b_x = Variable(x)
#                 #b_y = Variable(y)
#             out = cnn(b_x)
#
#             #print(F.softmax(out,dim=1).data.squeeze().cpu().numpy())
#             #print(torch.max(F.softmax(out,dim=1), 1)[0].data.squeeze().cpu().numpy())
#             #print(torch.max(F.softmax(out,dim=1), 1)[1].data.squeeze().int().cpu().numpy())
#             pred_one_probability = F.softmax(out,dim=1)[:,1].data.squeeze().cpu().numpy()
#             predict_value = torch.max(F.softmax(out,dim=1), 1)[1].data.squeeze().int().cpu().numpy()  # the index of the high probability, predicted value
#             pred_probability = np.concatenate((pred_probability,pred_one_probability),axis=0)
#             predict_values = np.concatenate((predict_values,predict_value),axis=0)
#         # according column to their initial column, and compute every column samples and its positive samples
#         length_test = len(column_test_data[i])
#         row = 1
#         count_positive = np.array([0,0])  # [[count_column_number,count_positive_number] [] ...]
#         for j in range(length_test):
#             column_test_data[i][j].add_probability_predict(pred_probability[j], predict_values[j])
#             count_positive[0] += 1
#             if(column_test_data[i][j].value==1):
#                 count_positive[1] += 1
#         #if the current column of test data no positive,so continue (get rid of train 0,1 )
#         if(count_positive[1]==0):
#             count_have_positive -= 1
#             continue
#         # else
#         # sort
#         temp_test_data = sorted(column_test_data[i], key=attrgetter('probability'), reverse=True)
#         """
#         # ---------test
#         print("this lung neoplasms")
#         for i in range(length_test):
#             a = miRNA_name[temp_test_data[i].index_x]
#             b = temp_test_data[i].predict_value
#             c = temp_test_data[i].value
#             print("miRNA_name,predict_value,real_value",a,b,c)
#         # ---------test
#         """
#         # ROC
#         all_number = length_test
#         count_assume_true = 0
#         count_negative = all_number - count_positive[1]
#         # compute tp,fn,fp,tn --------first using not sorted data
#         TP = 0
#         FP = 0
#         TN = 0
#         FN = 0
#         TPR = 0.0
#         FPR = 0.0
#         TPR_temp = []
#         FPR_temp = []
#         for x in range(all_number):
#             if (temp_test_data[x].value == 1 ):
#                 TP += 1
#                 FN = count_positive[1] - TP
#                 TPR = TP / count_positive[1]
#                 FPR = FP / count_negative
#                 TPR_temp.append(TPR)
#                 #print(x_TPR_temp)
#                 FPR_temp.append(FPR)
#             else:
#                 FP += 1
#                 TN = count_negative - FP
#                 TPR = TP / count_positive[1]
#                 FPR = FP / count_negative
#                 TPR_temp.append(TPR)
#                 FPR_temp.append(FPR)
#         area_column = area_ROC(TPR_temp, FPR_temp)
#         Y_TPR_all_columns.append(TPR_temp)
#         X_FPR_all_columns.append(FPR_temp)
#         area_all_column.extend([area_column])
#         #print("该次最终roc面积", area_column)
#         #plt.plot(FPR_temp, TPR_temp)
#         #plt.show()
#         print(i)
#         area += area_column
#     print("----")
#     print(len(X_FPR_all_columns))
#     print(count_have_positive)
#     average_column_area = area/count_have_positive
#
#     return Y_TPR_all_columns,X_FPR_all_columns,area_all_column,average_column_area
def draw_roc_column1(cnn,rows,test_data,Sd_associated,Sm,Sd,Lm,Md):
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
        test_x, test_y = get_test_data(test_data[i],Sd_associated,Sm,Sd,Lm,Md)
        test_out= cnn(test_x)
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
        # FN = 0
        TPR = 0.0
        FPR = 0.0
        P = 0.0
        R = 0.0
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
                P = TP / (TP+FP)
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
        area_column_PR = area_PR(P_temp,R_temp)
        #print("该次最终roc面积", area_column)
        #plt.plot(FPR_temp, TPR_temp)
        #plt.show()
        areaROC += area_column_ROC
        areaPR += area_column_PR
    areaROC = areaROC/count_have_positive
    areaPR = areaPR/count_have_positive
    return areaROC,areaPR

def draw_roc_column(cnn,columns,column_test_data,Sd_associated,Sm,Sd,Lm,Md):
    areaROC = 0.0
    areaPR = 0
    count_have_positive = columns
    Y_TPR_all_columns = []
    X_FPR_all_columns = []
    Y_P_all_columns = []
    X_R_all_columns = []
    area_all_column = []
    # according to column
    for i in range(columns):
        test_x, test_y = get_test_data(column_test_data[i],Sd_associated,Sm,Sd,Lm,Md)
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
    area = P[0] * P[0]
    for i in range(point_number-1):
        area += (R[i+1] - R[i]) * P[i]
    return area


def get_current_AUC_ROC(cnn,test_data,columns,Sd_associated,Sm,Sd,Lm,Md,time):

    #cnn.eval()
    length_test = len(test_data)
    columns = columns
    all_column_test_data = [[] for row in range(columns)]  # create a  list [[],[]...]  rows
    for i in range(length_test):
        all_column_test_data[test_data[i].index_y].append(test_data[i])  # ------pay attention append or extend
    # according to different column
    Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns, areaROC, areaPR = draw_roc_column(cnn,columns, all_column_test_data,Sd_associated,Sm,Sd,Lm,Md)
    x_all_FPR, y_all_TPR, x_all_R, y_all_P = gsp(Y_TPR_all_columns, X_FPR_all_columns, Y_P_all_columns, X_R_all_columns)
    x_FPR = np.sum(x_all_FPR, axis=0) / len(X_FPR_all_columns)
    y_TPR = np.sum(y_all_TPR, axis=0) / len(Y_TPR_all_columns)
    x_R = np.sum(x_all_R, axis=0) / len(X_R_all_columns)
    y_P = np.sum(y_all_P, axis=0) / len(Y_P_all_columns)
    #print('Time:{}, AUC:{}'.format(time, average_column_area))
    #plt.plot(x_FPR, y_TPR)
    #plt.show()
    return x_FPR,y_TPR,x_R,y_P,areaROC, areaPR


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
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()



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



# validation
def validation_test(alpha, k, max_count,batchsize,EPOCH,LR,dev_batchsize):
    all_k_Sd_associated, all_k_count_positive, all_k_test_data, A, num,\
    A_length, sava_association_A, all_k_development, k_pt_features,Sm, Sd = data_partitioning(alpha, k, max_count)
    X_all_k_FPR = []
    Y_all_k_TPR = []
    area_all_k = []
    for i in range(k):
        train_loader = load_data1(all_k_Sd_associated[i],k_pt_features[i], BATCH_SIZE=batchsize)
        # test_x, test_y = get_test_data(all_k_validation[0][:])
        dev_loader = load_data1(all_k_development[i],k_pt_features[i], BATCH_SIZE=dev_batchsize)
        train2(EPOCH, train_loader, dev_loader, LR)


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

# files operator
def save_to_file2(file_name, contents):
    with open(file_name, 'a') as f:
        f.write(contents + '\n')

# save_to_file2('a.txt', a)
def data_partitioning1(matrix_A,k):
    save_all_count_A, sava_association_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed, num, A_length  = crossvalidation(matrix_A,k)
    all_k_Sd_associated = []  # save all k * 1 * 326   k * disease * diseases   if(A(i,j) =1)
    all_k_count_positive = []  # save all  k positive length  (every time number of positive samples)
    all_k_development = []  # save all k development sets
    all_k_test_data = []  # save all k test data
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
                if (save_all_count_A[x][i][j] == 1):  # use new associated_A get positive
                    count_positive += 1
                    # because save_all_count_A come from matrix A, so location i and j not change ,so could use i ,j
                    data = data_input(1, i, j)
                    k_Sd_associated.append(data)  # save a certain disease  with certain miRNA  label is 1
                    # print('positive number')
                    # print(count_positive)
                    # shuffle before choose negative samples
        random.shuffle(save_count_zero)
        # choose negative sample

        k_zeros = []
        k_Sd_negative = []
        count_negative = count_positive
        # get frontal count_positive number because shuffle before taking
        for i in range(count_negative):
            k_Sd_negative.append(
                data_input(0, save_count_zero[i].value_x, save_count_zero[i].value_y ))

        # del frontal count_positive number
        """......."""
        # count = 0
        # while (count < count_negative):
        #     del save_count_zero[0]
        #     count += 1
            # combine positive and negative， but divide into train sets and development sets
        # first shuffle and then divide
        random.shuffle(k_Sd_associated)
        random.shuffle(k_Sd_negative)
        # length_dividing_point of train and validation
        length_dividing_point = int(len(k_Sd_associated) / 8 * 7)

        # train sets
        k_Sd_samples.extend(k_Sd_associated[:length_dividing_point])
        k_Sd_samples.extend(k_Sd_negative[:length_dividing_point])  # save in a raw
        random.shuffle(k_Sd_samples)
        all_k_Sd_associated.append(k_Sd_samples)
        all_k_count_positive.append(count_positive)

        # development sets
        k_Sd_samples_development.extend(k_Sd_associated[length_dividing_point:])
        k_Sd_samples_development.extend(k_Sd_negative[length_dividing_point:])
        random.shuffle(k_Sd_samples_development)
        all_k_development.append(k_Sd_samples_development)

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
                data_input(save_all_count_zero[i].value, save_all_count_zero[i].value_x, save_all_count_zero[i].value_y))
        all_k_test_data.append(temp_save)

    return all_k_Sd_associated, all_k_count_positive, all_k_test_data,  num, A_length, sava_association_A, all_k_development,save_all_count_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed

# test
def test(k):
    which_optim = {'Adam': 'Adam'}
    which_alpha = {'0.1': 0.1, '0.2': 0.2}
    which_maxcountrw = {'10': 10, '15': 15, '20': 20, '50': 50}
    which_Scheduler = {'None': 'None', 'lambda': 'LambdaLR', 'step': 'StepLR', 'Reduce': 'ReduceLROnPlateau'}
    which_weight_decay = {'0': 0, '0.1': 0.1, '0.5': 0.5, '0.9': 0.9, '0.01': 0.01, '0.05': 0.05, '0.001': 0.001,
                          '0.0001': 0.0001}
    which_LR = {'0.005': 0.005, '0.001': 0.001, '0.0005': 0.0005, '0.0001': 0.0001}
    which_EPOCH = {'10': 10, '20': 20, '30': 30, '40': 40, '50': 50, '60': 60}
    which_BatchSize = {'30': 30, '50': 50, '70': 70, '90': 90, '110': 110}
    which_l_r = {'10': 10, '20': 20, '30': 30}

    # which_optim = {'Adam': 'Adam', 'SGD': 'SGD', 'Adagrad': 'Adagrad','ASGD':'ASGD'}
    # which_alpha = {'0.1': 0.1,'0.2': 0.2,'0.3': 0.3,'0.4': 0.4,'0.5': 0.5,'0.6': 0.6,'0.7': 0.7,'0.8': 0.8,'0.9': 0.9}
    # which_maxcountrw = {'5': 5,'10': 10,'15': 15,'20': 20,'25': 25,'30': 30,'35': 35,'40': 40,'45': 45}
    # which_Scheduler ={'None':'None','lambda': 'LambdaLR','step': 'StepLR','Reduce':'ReduceLROnPlateau'}
    # which_weight_decay = {'0': 0,'0.1': 0.1,'0.5': 0.5,'0.9': 0.9,'0.01': 0.01,'0.05': 0.05,'0.001': 0.001,'0.006': 0.006}
    # which_LR = {'0.05': 0.05,'0.01': 0.01,'0.005': 0.005,'0.001': 0.001,'0.0005': 0.0005,'0.0001': 0.0001}
    # which_LR =np.arange(0,0.01,0.0001)
    # batchsize
    # Epoch
    time = 0
    # i = 0
    A, Sm, Sd, Lm, Md = read_data_flies()

    all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, association_A, all_k_validation, save_all_count_A = data_partitioning(
        A, k)

    input_size_lnc = len(A)  # 240
    input_size_A = len(A[0])  # 405
    input_size_lncmi = len(Lm[0])  # 495
    input_size_dis = input_size_A
    input_size_midis = input_size_lncmi

    """ 卷积过滤器的多少、dropout、等还未包括"""
    for alpha in which_alpha.values():  # 明确指出遍历字典的值
       # dev_loader = load_data1(all_k_development[i], k_pt_features[i], BATCH_SIZE=80)

        #for batchsize in range(10, 120, 10):
        for batchsize in which_BatchSize.values():

            # train_loader = load_data1(all_k_Sd_associated[i], k_pt_features[i], BATCH_SIZE=batchsize)
            #for epoch in range(10, 100, 2):
            for epoch in which_EPOCH.values():
                for _optim in which_optim.values():
                    for w_decay in which_weight_decay.values():
                        for _scheduler in which_Scheduler.values():
                            for _LR in which_LR.values():
                                output_size = 100
                                area_all = 0
                                for i in range(k):
                                    dev_loader = load_data2(all_k_validation[i], save_all_count_A[i], Sm, Sd, 80, Lm,
                                                            Md)
                                    train_loader = load_data2(all_k_Sd_associated[i], save_all_count_A[i], Sm, Sd,
                                                              batchsize, Lm, Md)
                                    mymodel = the_model(input_size_lnc, input_size_A, input_size_lncmi, input_size_dis,
                                                        input_size_midis, output_size, batchsize).cuda()

                                    mymodel.train()
                                    # X_all_k_FPR = []
                                    # Y_all_k_TPR = []
                                    # area_all_k = []

                                    # cnn.train()
                                    train2(mymodel, epoch, train_loader, dev_loader, _optim, w_decay, _scheduler, _LR)

                                    mymodel.eval()
                                    test_data = all_k_test_data[i]
                                    # x_all_FPR, y_all_TPR, current_k_area = get_current_AUC_ROC(cnn, test_data, len(A[0]),
                                    #                                                            k_pt_features[i],
                                    #                                                            i)  # all column

                                    length_test = len(test_data)
                                    rows = len(A[0])
                                    # print(rows)
                                    all_column_test_data = [[] for row in
                                                            range(rows)]  # create a  list [[],[]...]  rows
                                    for j in range(length_test):
                                        all_column_test_data[test_data[j].index_y].append(
                                            test_data[j])  # ------pay attention append or extend
                                    # # according to different column
                                    area = draw_roc_column1(mymodel, rows, all_column_test_data, save_all_count_A[i],
                                                            Sm, Sd,
                                                            Lm, Md)
                                    area_all += area
                                    del mymodel
                                a_area = area_all / k
                                d = 'Time:{}, AUC:{}, alpha:{}, BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
                                        time, a_area, alpha, batchsize, epoch, _optim, w_decay, _scheduler,
                                        _LR)
                                save_to_file2('DLACNNtest.txt', d)


def test1(k):
    which_optim = {'Adam': 'Adam'}
    # which_alpha = {'0.1': 0.1, '0.2': 0.2}
    # which_maxcountrw = {'10': 10, '15': 15, '20': 20, '50': 50}
    # which_Scheduler = {'None': 'None', 'lambda': 'LambdaLR', 'step': 'StepLR'}
    which_Scheduler = {'None': 'None', 'lambda': 'LambdaLR', 'step': 'StepLR'}
    # which_weight_decay = {'0': 0, '0.1': 0.1, '0.5': 0.5, '0.9': 0.9, '0.01': 0.01, '0.05': 0.05, '0.001': 0.001,
    #                       '0.0001': 0.0001}
    which_weight_decay = {'0': 0}
    which_LR = {'0.0005': 0.0005, '0.0001': 0.0001}
    # which_EPOCH = {'10': 10, '20': 20, '30': 30, '40': 40, '50': 50, '60': 60}
    which_EPOCH = {'40': 40, '80': 80, '160': 160}
    which_BatchSize = {'64': 64, '84': 84, '128': 128,}
    #which_l_r = {'10': 10, '20': 20, '30': 30}

    # which_optim = {'Adam': 'Adam', 'SGD': 'SGD', 'Adagrad': 'Adagrad','ASGD':'ASGD'}
    # which_alpha = {'0.1': 0.1,'0.2': 0.2,'0.3': 0.3,'0.4': 0.4,'0.5': 0.5,'0.6': 0.6,'0.7': 0.7,'0.8': 0.8,'0.9': 0.9}
    # which_maxcountrw = {'5': 5,'10': 10,'15': 15,'20': 20,'25': 25,'30': 30,'35': 35,'40': 40,'45': 45}
    # which_Scheduler ={'None':'None','lambda': 'LambdaLR','step': 'StepLR','Reduce':'ReduceLROnPlateau'}
    # which_weight_decay = {'0': 0,'0.1': 0.1,'0.5': 0.5,'0.9': 0.9,'0.01': 0.01,'0.05': 0.05,'0.001': 0.001,'0.006': 0.006}
    # which_LR = {'0.05': 0.05,'0.01': 0.01,'0.005': 0.005,'0.001': 0.001,'0.0005': 0.0005,'0.0001': 0.0001}
    # which_LR =np.arange(0,0.01,0.0001)
    # batchsize
    # Epoch
    time = 0
    # i = 0
    A, Sm, Sd, Lm, Md = read_data_flies()

    all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, association_A, all_k_validation, save_all_count_A = data_partitioning(
        A, k)

    input_size_lnc = len(A)  # 240
    input_size_A = len(A[0])  # 405
    input_size_lncmi = len(Lm[0])  # 495
    input_size_dis = input_size_A
    input_size_midis = input_size_lncmi

    """ 卷积过滤器的多少、dropout、学习率下降幅度、动量等等还未包括"""
    # dev_loader = load_data1(all_k_development[i], k_pt_features[i], BATCH_SIZE=80)

    # for batchsize in range(10, 120, 10):
    for batchsize in which_BatchSize.values():

        # train_loader = load_data1(all_k_Sd_associated[i], k_pt_features[i], BATCH_SIZE=batchsize)
        # for epoch in range(10, 100, 2):
        for epoch in which_EPOCH.values():
            for _optim in which_optim.values():
                for w_decay in which_weight_decay.values():
                    for _scheduler in which_Scheduler.values():
                        for _LR in which_LR.values():
                            output_size = 100
                            area_all = 0
                            for i in range(k):
                                dev_loader = load_data2(all_k_validation[i], save_all_count_A[i], Sm, Sd, 200, Lm,
                                                        Md)
                                train_loader = load_data2(all_k_Sd_associated[i], save_all_count_A[i], Sm, Sd,
                                                          batchsize, Lm, Md)
                                mymodel = the_model(input_size_lnc, input_size_A, input_size_lncmi, input_size_dis,
                                                    input_size_midis, output_size, batchsize).cuda()

                                mymodel.train()
                                # X_all_k_FPR = []
                                # Y_all_k_TPR = []
                                # area_all_k = []

                                # cnn.train()
                                train2(mymodel, epoch, train_loader, dev_loader, _optim, w_decay, _scheduler, _LR)

                                mymodel.eval()
                                test_data = all_k_test_data[i]
                                # x_all_FPR, y_all_TPR, current_k_area = get_current_AUC_ROC(cnn, test_data, len(A[0]),
                                #                                                            k_pt_features[i],
                                #                                                            i)  # all column

                                length_test = len(test_data)
                                rows = len(A[0])
                                # print(rows)
                                all_column_test_data = [[] for row in
                                                        range(rows)]  # create a  list [[],[]...]  rows
                                for j in range(length_test):
                                    all_column_test_data[test_data[j].index_y].append(
                                        test_data[j])  # ------pay attention append or extend
                                # # according to different column
                                area = draw_roc_column1(mymodel, rows, all_column_test_data, save_all_count_A[i],
                                                        Sm, Sd,
                                                        Lm, Md)
                                area_all += area
                                del mymodel
                            a_area = area_all / k
                            d = 'Time:{}, AUC:{},  BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
                                time, a_area, batchsize, epoch, _optim, w_decay, _scheduler,
                                _LR)
                            save_to_file2('DLACNNtest.txt', d)
                            time += 1


def test2(k):
    A, Sm, Sd, Lm, Md = read_data_flies()

    all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, association_A, all_k_validation,save_all_count_A = data_partitioning(A,k)
    epoch1 = 10
    epoch2 = 5
    _LR = 0.0001
    batchsize = 50
    # test_x, test_y = get_test_data(all_k_validation[0][:])
    area_all = 0
    for i in range(k):

        dev_loader = load_data2(all_k_validation[i],save_all_count_A[i],Sm,Sd,200,Lm,Md)



        train_loader = load_data2(all_k_Sd_associated[i], save_all_count_A[i],Sm,Sd,batchsize,Lm,Md)
        cnn = CNN().cuda()

        cnn.train()
        train3(cnn, epoch1, train_loader, dev_loader,_LR)
        #train4(cnn, epoch2, train_loader, dev_loader, _LR)

        cnn.eval()
        test_data = all_k_test_data[i]
        # current_k_area = draw_roc_column1(cnn,len(matrix_A[0]),test_data)  # all column

        # draw_roc2(test_data)  # all column
        length_test = len(test_data)



        rows = len(A[0])
        #print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].index_y].append(test_data[j])  # ------pay attention append or extend
        # # according to different column
        area = draw_roc_column1(cnn, rows,all_column_test_data,save_all_count_A[i],Sm,Sd,Lm,Md)
        area_all+=area
        print(area)

        # d = 'Time:{}, AUC:{}, BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
        # time, area, batchsize, epoch, _optim, w_decay, _scheduler, _LR)
        # save_to_file2('cnntest.txt', d)
        del cnn
        # time += 1
    a_a = area_all/k
    print(a_a)
    d = 'area:{},LR:{},epoch1:{},epoch2:{}'.format(a_a,_LR,epoch1,epoch2)
    print(d)
    # save_to_file2('cnnlncRNA.txt', d)


def test3(k):
    A, Sd, Lm, Md = read_data_flies()

    all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, association_A, all_k_validation,save_all_count_A = data_partitioning(A,k)
    epoch1 = 80
    epoch2 = 10
    _LR = 0.0001
    batchsize = 50
    # length of row
    input_size_lnc = len(A)  # 240
    input_size_A = len(A[0])  # 405
    input_size_lncmi = len(Lm[0])  # 495
    input_size_dis = input_size_A
    input_size_midis = input_size_lncmi
    output_size = 100
    # test_x, test_y = get_test_data(all_k_validation[0][:])
    area_all_ROC = 0.0
    area_all_PR = 0.0
    for i in range(k):
        Sm = get_all_sim1(save_all_count_A[i],Sd)

        dev_loader = load_data2(all_k_validation[i],save_all_count_A[i],Sm,Sd,batchsize,Lm,Md)



        train_loader = load_data2(all_k_Sd_associated[i], save_all_count_A[i],Sm,Sd,batchsize,Lm,Md)
        model = the_model(input_size_lnc,input_size_A,input_size_lncmi,input_size_dis,input_size_midis,output_size,batchsize).cuda()

        model.train()
        train3(model, epoch1, train_loader, dev_loader,_LR)
        #train3(model, epoch2, dev_loader, train_loader, _LR)
        #train4(cnn, epoch2, train_loader, dev_loader, _LR)
        model.eval()
        test_data = all_k_test_data[i]
        # current_k_area = draw_roc_column1(cnn,len(matrix_A[0]),test_data)  # all column

        # draw_roc2(test_data)  # all column
        length_test = len(test_data)



        rows = len(A[0])
        #print(rows)
        all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        for j in range(length_test):
            all_column_test_data[test_data[j].index_y].append(test_data[j])  # ------pay attention append or extend
        # # according to different column
        areaROC, areaPR = draw_roc_column1(model, rows,all_column_test_data,save_all_count_A[i],Sm,Sd,Lm,Md)
        area_all_ROC+=areaROC
        area_all_PR+=areaPR
        print(areaROC)
        print(areaPR)

        # d = 'Time:{}, AUC:{}, BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
        # time, area, batchsize, epoch, _optim, w_decay, _scheduler, _LR)
        # save_to_file2('cnntest.txt', d)
        del model
        # time += 1
    a_a_1 = area_all_ROC/k
    a_a_2 = area_all_PR / k
    # print(a_a)
    d = 'area_ROC:{},area_PR:{},LR:{},epoch1:{},epoch2:{}'.format(a_a_1,a_a_2,_LR,epoch1,epoch2)
    print(d)
    # save_to_file2('cnnlncRNA.txt', d)


def test4(k):
    A, Sd, Lm, Md = read_data_flies()

    all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, association_A, all_k_validation,save_all_count_A = data_partitioning(A,k)
    epoch1 = 80
    epoch2 = 10
    _LR = 0.0001
    batchsize = 50
    # length of row
    input_size_lnc = len(A)  # 240
    input_size_A = len(A[0])  # 405
    input_size_lncmi = len(Lm[0])  # 495
    input_size_dis = input_size_A
    input_size_midis = input_size_lncmi
    output_size = 100
    # test_x, test_y = get_test_data(all_k_validation[0][:])
    area_all_ROC = 0.0
    area_all_PR = 0.0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    for i in range(k):
        Sm = get_all_sim1(save_all_count_A[i],Sd)

        dev_loader = load_data2(all_k_validation[i],save_all_count_A[i],Sm,Sd,batchsize,Lm,Md)



        train_loader = load_data2(all_k_Sd_associated[i], save_all_count_A[i],Sm,Sd,batchsize,Lm,Md)
        model = the_model(input_size_lnc,input_size_A,input_size_lncmi,input_size_dis,input_size_midis,output_size,batchsize).cuda()

        model.train()
        train3(model, epoch1, train_loader, dev_loader,_LR)
        #train3(model, epoch2, dev_loader, train_loader, _LR)
        #train4(cnn, epoch2, train_loader, dev_loader, _LR)
        model.eval()
        test_data = all_k_test_data[i]
        # current_k_area = draw_roc_column1(cnn,len(matrix_A[0]),test_data)  # all column

        # draw_roc2(test_data)  # all column
        length_test = len(test_data)



        # rows = len(A[0])
        # #print(rows)
        # all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        # for j in range(length_test):
        #     all_column_test_data[test_data[j].index_y].append(test_data[j])  # ------pay attention append or extend
        # # # according to different column
        # areaROC0, areaPR0 = draw_roc_column1(model, rows,all_column_test_data,save_all_count_A[i],Sm,Sd,Lm,Md)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(model, test_data, len(A[0]),save_all_count_A[i],Sm,Sd,Lm,Md,i)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC+=areaROC
        area_all_PR+=areaPR
        # print("*****")
        print(areaROC)
        print(areaPR)
        # print(areaROC)
        # print(areaPR)
        # plt.plot(x_FPR, y_TPR)
        # plt.plot(x_R, y_P)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.show()
        # d = 'Time:{}, AUC:{}, BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
        # time, area, batchsize, epoch, _optim, w_decay, _scheduler, _LR)
        # save_to_file2('cnntest.txt', d)
        del model
        # time += 1

    a_a_1 = area_all_ROC/k
    a_a_2 = area_all_PR / k
    # print(a_a)
    d = 'area_ROC:{},area_PR:{},LR:{},epoch1:{},epoch2:{}'.format(a_a_1,a_a_2,_LR,epoch1,epoch2)
    print(d)
    get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    # save_to_file2('cnnlncRNA.txt', d)

def test6(k):
    A, Sd, Lm, Md = read_data_flies()

    all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, sava_association_A, \
    all_k_development, save_all_count_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed = data_partitioning1(A,k)
    epoch1 = 80
    epoch2 = 10
    _LR = 0.0001
    batchsize = 50
    # length of row
    input_size_lnc = len(A)  # 240
    input_size_A = len(A[0])  # 405
    input_size_lncmi = len(Lm[0])  # 495
    input_size_dis = input_size_A
    input_size_midis = input_size_lncmi
    output_size = 100
    # test_x, test_y = get_test_data(all_k_validation[0][:])
    area_all_ROC = 0.0
    area_all_PR = 0.0
    X_all_k_FPR = []
    Y_all_k_TPR = []
    X_all_k_R = []
    Y_all_k_P = []
    for i in range(k):
        Sm = get_all_sim1(save_all_count_A[i],Sd)

        dev_loader = load_data2(all_k_development[i],save_all_count_A[i],Sm,Sd,batchsize,Lm,Md)
        train_loader = load_data2(all_k_Sd_associated[i], save_all_count_A[i],Sm,Sd,batchsize,Lm,Md)
        model = the_model(input_size_lnc,input_size_A,input_size_lncmi,input_size_dis,input_size_midis,output_size,batchsize).cuda()
        model.train()
        train3(model, epoch1, train_loader, dev_loader,_LR)
        #train3(model, epoch2, dev_loader, train_loader, _LR)
        #train4(cnn, epoch2, train_loader, dev_loader, _LR)
        model.eval()
        test_data = all_k_test_data[i]
        # current_k_area = draw_roc_column1(cnn,len(matrix_A[0]),test_data)  # all column
        # draw_roc2(test_data)  # all column
        length_test = len(test_data)
        # rows = len(A[0])
        # #print(rows)
        # all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
        # for j in range(length_test):
        #     all_column_test_data[test_data[j].index_y].append(test_data[j])  # ------pay attention append or extend
        # # # according to different column
        # areaROC0, areaPR0 = draw_roc_column1(model, rows,all_column_test_data,save_all_count_A[i],Sm,Sd,Lm,Md)
        x_FPR, y_TPR, x_R, y_P, areaROC, areaPR = get_current_AUC_ROC(model, test_data, len(A[0]),save_all_count_A[i],Sm,Sd,Lm,Md,i)
        X_all_k_FPR.append(x_FPR)
        Y_all_k_TPR.append(y_TPR)
        X_all_k_R.append(x_R)
        Y_all_k_P.append(y_P)
        area_all_ROC+=areaROC
        area_all_PR+=areaPR
        # print("*****")
        # print(areaROC0)
        # print(areaPR0)
        # print(areaROC)
        # print(areaPR)
        # plt.plot(x_FPR, y_TPR)
        # plt.plot(x_R, y_P)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.show()
        # d = 'Time:{}, AUC:{}, BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
        # time, area, batchsize, epoch, _optim, w_decay, _scheduler, _LR)
        # save_to_file2('cnntest.txt', d)
        del model
        # time += 1
    a_a_1 = area_all_ROC/k
    a_a_2 = area_all_PR / k
    # print(a_a)
    d = 'area_ROC:{},area_PR:{},LR:{},epoch1:{},epoch2:{}'.format(a_a_1,a_a_2,_LR,epoch1,epoch2)
    print(d)
    DL_FPR, DL_TPR, DL_R, DL_P = get_all_k_Curve(X_all_k_FPR, Y_all_k_TPR, X_all_k_R, Y_all_k_P, k)
    # save_to_file2('cnnlncRNA.txt', d)

    # A, Sd, Lm, Md = read_data_flies()
    #
    # all_k_Sd_associated, all_k_count_positive, all_k_test_data, num, A_length, sava_association_A, \
    # all_k_development, save_all_count_A, save_all_count_zero_not_changed, save_all_count_zero_every_time_changed = data_partitioning(
    #     A, k)

    # DL_FPR, DL_TPR = test5(A, Sd, Lm, Md, all_k_Sd_associated, all_k_test_data, all_k_development, save_all_count_A)

    SIMC_FPR, SIMC_TPR,SIMC_R,SIMC_P,SIMC_AUC,SIMC_PR = SIMC.test2(A, Sd, save_all_count_A, sava_association_A, save_all_count_zero_every_time_changed,
                                    num, A_length)

    LDAP_FPR, LDAP_TPR,LDAP_R,LDAP_P,LDAP_AUC,LDAP_PR = LDAP.test2(A, save_all_count_A, save_all_count_zero_not_changed,
                                    save_all_count_zero_every_time_changed, num, A_length, sava_association_A)

    ANML_FPR, ANML_TPR,ANML_R,ANML_P,ANML_AUC,ANML_PR = ANML.test3(A, save_all_count_A, sava_association_A, save_all_count_zero_every_time_changed,
                                    num, A_length)

    MFLDA_FPR, MFLDA_TPR,MFLDA_R,MFLDA_P,MFLDAC_AUC,MFLDA_PR = MFLDA.test2(A, save_all_count_A, sava_association_A, save_all_count_zero_not_changed,
                                       save_all_count_zero_every_time_changed, num, A_length)

    p2, = plt.plot(SIMC_FPR, SIMC_TPR, 'g')
    p1, = plt.plot(DL_FPR, DL_TPR, 'r')
    p5, = plt.plot(LDAP_FPR, LDAP_TPR, 'b')
    p3, = plt.plot(ANML_FPR, ANML_TPR, 'y')
    p4, = plt.plot(MFLDA_FPR, MFLDA_TPR, 'magenta')

    l1 = plt.legend([p1, p2, p3, p4, p5],
                    ["DLACNNLDA(0.9473)", "SIMCLDA(0.7464)", "ANMLDA(0.8714)", "MFLDA(0.6545)", "LDAP(0.8829)"],
                    loc='lower right')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('Five fold Cross−Validation', fontsize='large', fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()

    p2, = plt.plot(SIMC_R, SIMC_P, 'g')
    p1, = plt.plot(DL_R, DL_P, 'r')
    p5, = plt.plot(LDAP_R, LDAP_P, 'b')
    p3, = plt.plot(ANML_R, ANML_P, 'y')
    p4, = plt.plot(MFLDA_R, MFLDA_P, 'magenta')

    l1 = plt.legend([p1, p2, p3, p4, p5],
                    ["DLACNNLDA", "SIMCLDA", "ANMLDA", "MFLDA", "LDAP"],
                    loc='upper right')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title('Precision−Recall Curve', fontsize='large', fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()
if __name__ == '__main__':
    # model
    # if torch.cuda.is_available():
    #     cnn = CNN().cuda()
    # else:
    #     cnn = CNN()

    # parameters
    # alpha = 0.1
    k = 5
    # max_count_rw = 10
    # batchsize = 50
    # EPOCH = 100
    # LR = 0.06

    # cross validation
    # validation_test(alpha, k, max_count_rw,batchsize,EPOCH,LR,dev_batchsize=50)
    #test(alpha, k, max_count_rw, batchsize, EPOCH, LR, 50)
    test6(k)  # for test
    # test1(k)  # for regulating




















