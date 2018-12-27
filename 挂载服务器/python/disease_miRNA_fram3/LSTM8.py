# LSTM_6 wants to use pack in lstm, but i want to test so create a new one
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import data_load as dl
import data_prepare as dp
import pt_search_threshold as pt
import ROC_AUC as RA
import numpy as np
from torch.nn import utils as nn_utils

# torch.manual_seed(1)    # reproducible


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


class Attention(nn.Module):
    def __init__(self,input_size,output_size):
        super(Attention,self).__init__()

        self.node = nn.Linear(input_size, output_size,bias=True) #input_size * 2   + zs_size(hidden * 2)
        nn.init.xavier_normal(self.node.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,1))
        nn.init.xavier_normal(self.h_n_parameters)

    def forward(self, h_n_states):
        temp_nodes = self.node(h_n_states) # 4 * (input_size *2) ,4 * output_size
        temp_nodes = F.tanh(temp_nodes)
        nodes_score = torch.mm(temp_nodes,self.h_n_parameters)  # 4 * 1
        nodes_score = torch.t(nodes_score)  # 1 * 4
        #print(nodes_score)
        alpha = F.softmax(nodes_score,dim=1)
        #print(alpha)
        y_i = torch.mm(alpha,h_n_states)  #  1 * 4  *  4 * input_size, 1 * input_size
        return y_i

class Attention2(nn.Module):
    def __init__(self,input_size,output_size,zs_size):
        super(Attention2,self).__init__()

        self.node = nn.Linear(input_size+zs_size, output_size,bias=True) #input_size * 2   + zs_size(hidden * 2)
        nn.init.xavier_normal(self.node.weight)

        #self.ZS = nn.Linear(zs_size, output_size, bias=False)  # input_size * 2   + zs_size(hidden * 2)
        #nn.init.xavier_normal(self.ZS.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,1))
        nn.init.xavier_normal(self.h_n_parameters)

    def forward(self, h_n_states,z):
        z_h = torch.cat([h_n_states, torch.cat([z for _ in range(h_n_states.size(0))], 0)],1) #torch.Size([4, 1832])
        #print(zs)
        #print(zs.size())
        temp_nodes = self.node(z_h) # 4 * (input_size *2) ,4 * output_size
        temp_nodes = F.tanh(temp_nodes)

        #temp_zs = self.ZS(z)
        #temp_zs = F.tanh(temp_zs)
        #temp = torch.cat((temp_nodes,temp_zs),0)

        nodes_score = torch.mm(temp_nodes,self.h_n_parameters)  # 4 * 1
        nodes_score = torch.t(nodes_score)  # 1 * 4
        #print(nodes_score)
        alpha = F.softmax(nodes_score,dim=1)
        #print(alpha)
        y_i = torch.mm(alpha,h_n_states)  #  1 * 4  *  4 * input_size, 1 * input_size
        return y_i
"""
class Batch_Attention(nn.Module):
    def __init__(self,input_size,output_size):
        super(Batch_Attention,self).__init__()

        self.node = nn.Linear(input_size, output_size,bias=True) #input_size * 2   + zs_size(hidden * 2)
        nn.init.xavier_normal(self.node.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,1))
        nn.init.xavier_normal(self.h_n_parameters)

    def forward(self, h_n_states):  # batch * time_step * (output_size * num_directions)
        temp_nodes = self.node(h_n_states) # 4 * (input_size *2) ,4 * output_size
        temp_nodes = F.tanh(temp_nodes)
        nodes_score = torch.mm(temp_nodes,self.h_n_parameters)  # 4 * 1
        nodes_score = torch.t(nodes_score)  # 1 * 4
        #print(nodes_score)
        alpha = F.softmax(nodes_score,dim=1)
        #print(alpha)
        y_i = torch.mm(alpha,h_n_states)  #  1 * 4  *  4 * input_size, 1 * input_size
        return y_i
"""

class Beta_score(nn.Module):
    def __init__(self,input_size,output_size,gamma):
        super(Beta_score,self).__init__()

        self.gamma = gamma

        self.node = nn.Linear(input_size , output_size,bias=True) #input_size * 2 + zs_size(hidden * 2)
        nn.init.xavier_normal(self.node.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,1))
        nn.init.xavier_normal(self.h_n_parameters)

    def forward(self, y_i,num):
        temp_nodes = self.node(y_i) # 103519 * (input_size) ,103519 * output_size
        temp_nodes = F.tanh(temp_nodes)
        nodes_score = torch.mm(temp_nodes,self.h_n_parameters)  # 103519 * 1
        # print(nodes_score.size())
        # print((self.gamma * num).size())
        nodes_score = nodes_score - (self.gamma * num)
        nodes_score = torch.t(nodes_score)  # 1 * 103519
        beta = F.softmax(nodes_score,dim=1)
        #print(beta)

        return beta

class Beta_score2(nn.Module):
    def __init__(self,input_size,output_size,gamma,zs_size):
        super(Beta_score2,self).__init__()

        self.gamma = gamma

        self.node = nn.Linear(input_size+zs_size , output_size,bias=True) #input_size * 2 + zs_size(hidden * 2)
        nn.init.xavier_normal(self.node.weight)

        self.h_n_parameters = nn.Parameter(torch.randn(output_size,1))
        nn.init.xavier_normal(self.h_n_parameters)

    def forward(self, y_i,num,z):
        z_h = torch.cat([y_i, torch.cat([z for _ in range(y_i.size(0))], 0)], 1)

        temp_nodes = self.node(z_h) # 103519 * (input_size) ,103519 * output_size
        temp_nodes = F.tanh(temp_nodes)
        nodes_score = torch.mm(temp_nodes,self.h_n_parameters)  # 103519 * 1
        nodes_score = torch.t(nodes_score)
        #print(nodes_score.size())
        #print((self.gamma * num).size())
        #print(nodes_score)
        #print(self.gamma * num)
        nodes_score = nodes_score - (self.gamma * num)
        #print(nodes_score)
        #nodes_score = torch.t(nodes_score)  # 1 * 103519
        beta = F.softmax(nodes_score,dim=1)  # 1 * 2
        #print(beta)

        return beta
"""
def load_data_train(nodes_set,A,current_A,miRNA_miRNA_matrix,disease_disease_matrix,BATCH_SIZE):
    x = []
    y = []
    goals, paths_feature = dl.add_feature(nodes_set[:10], A, save_all_count_A[0], Sm, Sd)
        x.append([temp_save])
        y.append(k_Sd_associated[j].value)
    x = torch.FloatTensor(np.array(x))
    print(x.size())
    y = torch.IntTensor(np.array(y))
    torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
    train_loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=False,              # random shuffle for training
            num_workers=1,              # subprocesses for loading data
        )
    return train_loader

"""
class Node_listm(nn.Module):
    def __init__(self,input_size,output_size):
        super(Node_listm,self).__init__()
        self.lstm_node1 = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=output_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
        )
        nn.init.xavier_normal(self.lstm_node1.all_weights[0][0], 1)
        nn.init.xavier_normal(self.lstm_node1.all_weights[0][1], 1)
        nn.init.xavier_normal(self.lstm_node1.all_weights[1][0], 1)
        nn.init.xavier_normal(self.lstm_node1.all_weights[1][1], 1)

        self.lstm_node2 = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=output_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
        )
        nn.init.xavier_normal(self.lstm_node2.all_weights[0][0])
        nn.init.xavier_normal(self.lstm_node2.all_weights[0][1])
        nn.init.xavier_normal(self.lstm_node2.all_weights[1][0])
        nn.init.xavier_normal(self.lstm_node2.all_weights[1][1])

        self.lstm_node3 = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=output_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
        )
        nn.init.xavier_normal(self.lstm_node3.all_weights[0][0])
        nn.init.xavier_normal(self.lstm_node3.all_weights[0][1])
        nn.init.xavier_normal(self.lstm_node3.all_weights[1][0])
        nn.init.xavier_normal(self.lstm_node3.all_weights[1][1])

        self.lstm_node4 = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=output_size,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True,
        )
        nn.init.xavier_normal(self.lstm_node4.all_weights[0][0])
        nn.init.xavier_normal(self.lstm_node4.all_weights[0][1])
        nn.init.xavier_normal(self.lstm_node4.all_weights[1][0])
        nn.init.xavier_normal(self.lstm_node4.all_weights[1][1])

    def forward(self, x):
        # x should be a batch
        temp_x = x[0].view(1,1,-1)
        r_out_1, (h_n_1, h_c_1) = self.lstm_node1(temp_x, None)  # None represents zero initial hidden state
        #print(self.rnn.all_weights)
        #print(r_out[:, -1, :])
        #print(h_n_1.size())#torch.Size([2, 1, 123])
        temp_x = x[1].view(1, 1, -1)
        r_out_2, (h_n_2, h_c_2) = self.lstm_node2(temp_x, (h_n_1, h_c_1))
        temp_x = x[2].view(1, 1, -1)
        r_out_3, (h_n_3, h_c_3) = self.lstm_node3(temp_x, (h_n_2, h_c_2))
        temp_x = x[3].view(1, 1, -1)
        r_out_4, (h_n_4, h_c_4) = self.lstm_node4(temp_x, (h_n_3, h_c_3))

        h_n_1 = h_n_1.view(1,1,-1)
        #print(h_n_1.size())#torch.Size([1, 1, 246])
        h_n_2 = h_n_2.view(1,1, -1)
        h_n_3 = h_n_3.view(1,1, -1)
        h_n_4 = h_n_4.view(1,1, -1)
        h_n = torch.squeeze(torch.cat((h_n_1,h_n_2,h_n_3,h_n_4),0))  # 1 represent cat with row  8 * input_size
        # print(h_n.size())  torch.Size([8, 1, 123])
        #h_n = h_n.view(4,-1)
        #print(h_n.size()) # torch.Size([4, 246])
        return  h_n


class model1(nn.Module):
    def __init__(self,input_size,output_size):
        super(model, self).__init__()
        #self.nodes_listm =Node_listm(input_size,output_size) # x ,one batch
        self.nodes_listm = nn.LSTM(input_size,output_size,num_layers=1, batch_first=True,bidirectional=True ) #816 *100
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        self.attention_nodes = Attention2((output_size * 2),output_size,(input_size*2))#
        self.attention_paths = Beta_score2((output_size * 2),output_size,gamma=0.2,zs_size = (input_size*2))
        self.fully_connected = nn.Linear( (2* input_size+2*output_size),2)

    def forward(self, x,z,length_pre_,path_length):
        #h_n = self.nodes_listm(torch.unsqueeze(x[0],dim=0))  # torch.Size([4, 246])
        temp_x = torch.unsqueeze(x[0],0)
        #print(temp_x.size())
        r_out, (h_n, h_c) = self.nodes_listm(temp_x,None) # r_out shape (batch, time_step, output_size)       use
        #print(r_out[0][:path_length[0],:])
        all_y_i = self.attention_nodes(r_out[0][:path_length[0],:],z)  # 1 * input_size   (r_out[0],time_step * (output_size * num_directions))
        all_y_i_num = []
        y_i_num = r_out[0].size(1)
        all_y_i_num.append([y_i_num])
        #all_y_i = self.attention_nodes(r_out)  # 1 * input_size   ,(r_out,batch * time_step * (output_size * num_directions))
        #all_y_i = torch.Tensor([])
        for i in range(length_pre_-1):#103519       change for loop in matrix multiplication
            #h_n = self.nodes_listm(x[i+1])
            #print(x[i+1].size())
            temp_x = torch.unsqueeze(x[i+1],0)
            r_out, (h_n, h_c) = self.nodes_listm(temp_x, None)  # r_out shape (batch, time_step, output_size)       use
            y_i = self.attention_nodes(r_out[0][:path_length[i+1],:],z) #1 * input_size
            y_i_num = r_out[0].size(0)
            all_y_i = torch.cat((all_y_i,y_i),0) #103519 * (input_size)
            all_y_i_num.append([y_i_num])
        # print(np.shape(all_y_i_num))
        all_y_i_num = Variable(torch.Tensor(all_y_i_num))
        beta = self.attention_paths(all_y_i,all_y_i_num,z)  #103519 * input_size,  1 * 103519
        g_z = torch.mm(beta , all_y_i) # 1 * attention.input_size
        #print(g_z.size())
        Z = torch.cat((g_z,z),1)  # 1 * input_size + 2 * input_size
        y = self.fully_connected(Z)

        return y


class model(nn.Module):
    def __init__(self,input_size,output_size):
        super(model, self).__init__()
        #self.nodes_listm =Node_listm(input_size,output_size) # x ,one batch
        self.nodes_listm = nn.LSTM(input_size,output_size,num_layers=1, batch_first=True,bidirectional=True ) #816 *100
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        # self.attention_nodes = Attention2((output_size * 2),output_size,(input_size*2))#
        self.attention_nodes = Attention(output_size*2 , output_size)
        self.attention_paths = Beta_score2((output_size * 2),output_size,gamma=0.9,zs_size = (input_size*2))
        self.fully_connected = nn.Linear( (2*output_size+ input_size),2)
        # could add operation like conv
    def forward(self, x,z,z_element):
        r_out, (h_n, h_c) = self.nodes_listm(x, None)
        r_outputs,r_out_length = nn_utils.rnn.pad_packed_sequence(r_out,batch_first=True)
        # print(r_outputs.size())  # torch.Size([2, 3, 200])     batch, seq_len, hidden_size * num_directions
        # print(r_out_length)
        # print(r_outputs[1][2])  # This is padding one, so this hidden_state is all 0.
        # print(h_n.size())  # torch.Size([2, 2, 100]) b * (num_layer*num_directions) * hidden_size

        paths = r_outputs.size(0)
        #print(type(r_out_length))
        r_out_length = Variable(torch.Tensor([r_out_length])).cuda()
        # print(paths)
        # y_i_all = []  # all path y_i
        # for path in range(paths):
        #     y_i = self.attention_nodes(r_outputs[path])  # 1 * input_size
        #     y_i_all.append(y_i)
        # beta = self.attention_paths(y_i_all, r_out_length, z)

        # y_i_all = []  # all path y_i
        for path in range(paths):
            if (path!=0) :
                y_i = torch.cat((y_i,self.attention_nodes(r_outputs[path])),dim=0)
            else:
                y_i = self.attention_nodes(r_outputs[path])  # 1 * input_size
        # print(y_i.size())
        beta = self.attention_paths(y_i, r_out_length, z)
        # print(beta.size())
        g_z = torch.mm(beta, y_i)  # 1 * attention.input_size
        #print(g_z.size())
        Z = torch.cat((g_z, z_element), 1)  # 1 * input_size + 2 * input_size
        y = self.fully_connected(Z)
        return y

def train(the_model,A, Sm, Sd,save_all_count_A,nodes_set,devs_set,epoch,LR):


    optimizer = torch.optim.Adam(the_model.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    nodes_length = len(nodes_set)

    for sample in range(nodes_length):
        the_model.train()
        #print(nodes_set[sample])
        goals, paths_feature,paths_length= dl.add_feature5(nodes_set[sample], A, save_all_count_A, Sm, Sd)
        #print(np.shape(paths_feature))

        x = Variable(torch.FloatTensor(paths_feature)).cuda()  # paths * nodes * feature
        y = Variable(torch.LongTensor(np.array([goals[2]]).astype(np.int64))).cuda()

        z_st, temp_feature1, temp_feature2 = dl.add_feature_goal_node(int(goals[0]), int(goals[1]), save_all_count_A, Sm,Sd)

        z_st = Variable(torch.FloatTensor([z_st])).cuda()  # [m_m,m_d,d_m,d_d] 816 * 2
        # print(z_st.size())
        z_element = Variable(torch.FloatTensor(np.array([temp_feature1]) * np.array([temp_feature2]))).cuda()

        # print(np.shape(z_element))
        # print(z_element.size())
        # z_st [torch.FloatTensor of size 1x1632]
        x_packed = nn_utils.rnn.pack_padded_sequence(x, paths_length, batch_first=True)

        # out = the_model(x_packed, z_st)
        out = the_model(x_packed,z_st,z_element)
        loss = loss_func(out, y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()

        if((sample)%3500==1):
            the_model.eval()
            dev_length = len(devs_set)
            same_number = 0
            #same_number_length = 0
            for dev_sample in range(dev_length):
                dev_goals, dev_paths_feature, dev_paths_length = dl.add_feature5(devs_set[dev_sample], A, save_all_count_A, Sm, Sd)
                # print(np.shape(paths_feature))
                dev_x = Variable(torch.FloatTensor(dev_paths_feature)).cuda()  # paths * nodes * feature
                dev_y = torch.IntTensor(np.array([dev_goals[2]]).astype(np.int64))
                #dev_y = Variable(torch.LongTensor(np.array([goals[dev_sample][2]]).astype(np.int64))).cuda()
                dev_z_st, dev_temp_feature1, dev_temp_feature2 = dl.add_feature_goal_node(int(dev_goals[0]),
                                                                              int(dev_goals[1]), save_all_count_A,
                                                                              Sm, Sd)

                dev_z_st = Variable(torch.FloatTensor([dev_z_st])).cuda()  # [m_m,m_d,d_m,d_d] 816 * 2
                # print(z_st.size())
                dev_z_element = Variable(torch.FloatTensor(np.array([dev_temp_feature1]) * np.array([dev_temp_feature2]))).cuda()

                # print(np.shape(z_element))
                # print(z_element.size())
                # z_st [torch.FloatTensor of size 1x1632]
                dev_x_packed = nn_utils.rnn.pack_padded_sequence(dev_x, dev_paths_length, batch_first=True)

                # out = the_model(x_packed, z_st)
                dev_out = the_model(dev_x_packed, dev_z_st, dev_z_element)
                pred_y = torch.max(dev_out, 1)[1].data.squeeze().cpu().int()
                #print(type(pred_y))
                # print(dev_y)
                # print(pred_y)
                same_number += sum(pred_y == dev_y)
                #same_number_length += dev_y.size(0)

            accuracy = same_number / float(dev_length)
            print('Epoch: ',epoch, '| train loss: %.4f' % loss.data[0], '| dev accuracy: %.4f' % accuracy)
            #the_model.train()

def dev_test(A, Sm, Sd,save_all_count_A,devs_set,loss):
    the_model.eval()
    dev_length = len(devs_set)
    same_number = 0
    # same_number_length = 0
    for dev_sample in range(dev_length):
        dev_goals, dev_paths_feature, dev_paths_length = dl.add_feature5(devs_set[dev_sample], A, save_all_count_A, Sm,
                                                                         Sd)
        # print(np.shape(paths_feature))
        dev_x = Variable(torch.FloatTensor(dev_paths_feature)).cuda()  # paths * nodes * feature
        dev_y = torch.IntTensor(np.array([dev_goals[2]]).astype(np.int64))
        # dev_y = Variable(torch.LongTensor(np.array([goals[dev_sample][2]]).astype(np.int64))).cuda()
        dev_z_st, dev_temp_feature1, dev_temp_feature2 = dl.add_feature_goal_node(int(dev_goals[0]),
                                                                                  int(dev_goals[1]), save_all_count_A,
                                                                                  Sm, Sd)

        dev_z_st = Variable(torch.FloatTensor([dev_z_st])).cuda()  # [m_m,m_d,d_m,d_d] 816 * 2
        # print(z_st.size())
        dev_z_element = Variable(
            torch.FloatTensor(np.array([dev_temp_feature1]) * np.array([dev_temp_feature2]))).cuda()

        # print(np.shape(z_element))
        # print(z_element.size())
        # z_st [torch.FloatTensor of size 1x1632]
        dev_x_packed = nn_utils.rnn.pack_padded_sequence(dev_x, dev_paths_length, batch_first=True)

        # out = the_model(x_packed, z_st)
        dev_out = the_model(dev_x_packed, dev_z_st, dev_z_element)
        pred_y = torch.max(dev_out, 1)[1].data.squeeze().cpu().int()
        # print(type(pred_y))
        same_number += sum(pred_y == dev_y)
        # same_number_length += dev_y.size(0)

    accuracy = same_number / float(dev_length)
    print('| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

def test():
    which_optim = {'Adam': 'Adam'}

    # which_Scheduler = {'None': 'None', 'lambda': 'LambdaLR', 'step': 'StepLR', 'Reduce': 'ReduceLROnPlateau'}
    which_Scheduler = { 'step': 'StepLR'}
    which_weight_decay = {'0.001': 1e-5}
    which_LR = {'0.001': 0.001, '0.0005': 0.0005, '0.0001': 0.0001}
    #which_EPOCH = { '20': 20, '30': 30, '40': 40, '50': 50, '60': 60}
    which_EPOCH = { '100':100}
    #which_BatchSize = {'30': 30, '50': 50, '70': 70, '90': 90, '110': 110}
    which_BatchSize = {'100': 100}
    #which_l_r = {'10': 10, '20': 20, '30': 30}







    all_k_Sd_associated, all_k_count_positive, all_k_test_data, matrix_A, num, A_length, association_A, all_k_validation = get_pre_train_test_data(
        k_fold, k_means)

    for epoch in which_EPOCH.values():
        for _scheduler in which_Scheduler.values():
            for _LR in which_LR.values():
                area_all = []
                for x in range(k_fold):

                    # test_x, test_y = get_test_data(all_k_validation[0][:])
                    dev_loader = get_test_data2(all_k_validation[x][:])
                    test_data = all_k_test_data[x]
                    # current_k_area = draw_roc_column1(cnn,len(matrix_A[0]),test_data)  # all column

                    # draw_roc2(test_data)  # all column
                    length_test = len(test_data)
                    rows = 326
                    all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
                    for i in range(length_test):
                        all_column_test_data[test_data[i].index_y].append(
                            test_data[i])  # ------pay attention append or extend

                    time = 0
                    i = 0
                    train_loader = load_data_train(all_k_Sd_associated[x], BATCH_SIZE=100)

                    cnn = CNN().cuda()

                    cnn.train()
                    train3(cnn, epoch, train_loader, dev_loader, current_optim='Adam', w_decay=1e-5,
                       _scheduler=_scheduler, _LR=_LR)

                    cnn.eval()

                    # # according to different column
                    area = draw_roc_column1(cnn, rows, all_column_test_data)
                    area_all.append(area)
                # print("最终AUC", area)

                # length_test = len(test_data)
                # rows = 326
                # all_column_test_data = [[] for row in
                #                         range(rows)]  # create a  list [[],[]...]  rows
                # for i in range(length_test):
                #     all_column_test_data[test_data[i].index_y].append(
                #         test_data[i])  # ------pay attention append or extend
                # # according to different column
                # area = draw_roc_column(rows, all_column_test_data)
                # print("最终AUC", area)

                # X_all_k_FPR = []
                # Y_all_k_TPR = []
                # area_all_k = []

                    d = 'Time:{}, AUC:{}, BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
                        time, area, 100, epoch, 'Adam', 1e-5, _scheduler, _LR)
                    save_to_file2('cnntest.txt', d)
                mean_area = 'average_AUC:{}'.format(np.mean(area_all))
                save_to_file2('cnntest.txt',mean_area )

                del cnn
                time += 1


def get_result():
    # Hyper Parameters
    # EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
    # BATCH_SIZE = 64
    # TIME_STEP = 28  # rnn time step / image height
    # INPUT_SIZE = 28  # rnn input size / image width
    LR = 0.001  # learning rate

    the_model = model(816, 100).cuda()
    print(the_model)

    # nodes_set = dl.read_npz_data1(1, 500)
    # devs_set = dl.read_npz_data1(2,500)
    # A, Sm, Sd = dp.read_data_flies()
    # D = np.load('./get_data/data_prepare/devided_data.npz')
    # # all_k_train_set = D['all_k_train_set']
    # # all_k_development_set = D['all_k_development_set']
    # # all_k_test_set = D['all_k_test_set']
    # save_all_count_A = D['save_all_count_A']
    #
    #
    #
    # train(A, Sm, Sd,save_all_count_A,nodes_set,devs_set)
    A, Sm, Sd = dp.read_data_flies()
    k = 10
    k_count = 0
    D = np.load('./get_data/data_prepare/devided_data.npz')
    all_k_train_set = D['all_k_train_set']
    all_k_development_set = D['all_k_development_set']
    all_k_test_set = D['all_k_test_set']
    save_all_count_A = D['save_all_count_A']

    # save_paths_nodes3(all_k_train_set[0],all_k_development_set[0],all_k_test_set[0],save_all_count_A[0],Sm,Sd, threshold_value=0.2)
    # path_node_train_set, path_node_dev_set, path_node_test_set, path_edge_weights_train_set, path_edge_weights_dev_set, path_edge_weights_test_set = pt.save_paths_nodes4(all_k_train_set[0], all_k_development_set[0], all_k_test_set[0], save_all_count_A[0], Sm, Sd,
    # 0,threshold_value=0.2)
    path_node_train_set = np.load('./get_data/path_search/new_all/train_nodes0.npz')['path_node_train_set']
    path_node_dev_set = np.load('./get_data/path_search/new_all/dev_nodes0.npz')['path_node_dev_set']
    for epoch in range(20):
        train(the_model, A, Sm, Sd, save_all_count_A[k_count], path_node_train_set, path_node_dev_set,epoch,LR)


    path_node_test_set = np.load('./get_data/path_search/new_all/test_nodes0.npz')['path_node_test_set']
    rows = len(A[0])
    length_test = len(path_node_test_set)
    test_data = path_node_test_set

    all_column_test_data = [[] for row in range(rows)]  # create a  list [[],[]...]  rows
    for j in range(length_test):
        all_column_test_data[test_data[j][0][-1]].append(test_data[j])  # ------pay attention append or extend
    X_FPR_all_columns, Y_TPR_all_columns, area = RA.draw_roc_column2(the_model,rows,all_column_test_data,save_all_count_A[k_count],Sm,Sd,A)
    print(area)

if __name__ == '__main__':
    # Hyper Parameters
    # EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
    # BATCH_SIZE = 64
    # TIME_STEP = 28  # rnn time step / image height
    # INPUT_SIZE = 28  # rnn input size / image width
    # LR = 0.01  # learning rate


    get_result()


    # the_model = model(816,100).cuda()
    # print(the_model)



    # nodes_set = dl.read_npz_data1(1, 500)
    # devs_set = dl.read_npz_data1(2,500)
    # A, Sm, Sd = dp.read_data_flies()
    # D = np.load('./get_data/data_prepare/devided_data.npz')
    # # all_k_train_set = D['all_k_train_set']
    # # all_k_development_set = D['all_k_development_set']
    # # all_k_test_set = D['all_k_test_set']
    # save_all_count_A = D['save_all_count_A']
    #
    #
    #
    # train(A, Sm, Sd,save_all_count_A,nodes_set,devs_set)
    # A, Sm, Sd = dp.read_data_flies()
    # k = 10
    # k_count = 0
    # D = np.load('./get_data/data_prepare/devided_data.npz')
    # all_k_train_set = D['all_k_train_set']
    # all_k_development_set = D['all_k_development_set']
    # all_k_test_set = D['all_k_test_set']
    # save_all_count_A = D['save_all_count_A']


    # save_paths_nodes3(all_k_train_set[0],all_k_development_set[0],all_k_test_set[0],save_all_count_A[0],Sm,Sd, threshold_value=0.2)
    # path_node_train_set, path_node_dev_set, path_node_test_set, path_edge_weights_train_set, path_edge_weights_dev_set, path_edge_weights_test_set = pt.save_paths_nodes4(all_k_train_set[0], all_k_development_set[0], all_k_test_set[0], save_all_count_A[0], Sm, Sd,
                      # 0,threshold_value=0.2)
    # path_node_train_set = np.load('./get_data/path_search/new_all/train_nodes0.npz')['path_node_train_set']
    # path_node_dev_set = np.load('./get_data/path_search/new_all/dev_nodes0.npz')['path_node_dev_set']
    # for epoch in range(1):
    #     train(the_model,A, Sm, Sd, save_all_count_A[k_count], path_node_train_set, path_node_dev_set)









