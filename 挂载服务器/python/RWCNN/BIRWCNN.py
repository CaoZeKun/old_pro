import BIRandomWalk as rw
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
        self.probability = 0.0
        self.predict_value = -1
    def add_probability_predict(self,probability,predict_value):
        self.probability = probability
        self.predict_value = predict_value


# data partitioning train dev test
def data_partitioning(alpha,k,l,r):
    A, Sm, Sd, save_all_count_A, sava_association_A, save_all_count_zero_not_changed, \
    save_all_count_zero_every_time_changed, num, A_length, k_pt_features = rw.get_k_pt(alpha,k,l,r)
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
        count_negative = 0
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
        length_dividing_point = int(len(k_Sd_associated) / 9 * 7)

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

    return all_k_Sd_associated, all_k_count_positive, all_k_test_data, A, num, A_length, sava_association_A, all_k_development,k_pt_features,Sm, Sd


# data load,i--miRna  j--disease, use pt[i] and pt[:,490+j] as feature
#    490*490      490*326
# [[Rna*Rna] [A(Rna*Disease)]
#  [A.T(Disease*Rna)] [Disease*Disease]]
#     326*490            326*326
def load_data1(data,pt_feature,BATCH_SIZE):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        temp_save.append(pt_feature[:,data[j].index_x])
        temp_save.append(pt_feature[:,data[j].index_y+490])
        x.append([temp_save])
        y.append(data[j].value)
    x = torch.FloatTensor(np.array(x))
    #print(x.size())
    y = torch.IntTensor(np.array(y))
    torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
    data1_loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=False,              # random shuffle for training
            num_workers=1,              # subprocesses for loading data
        )
    return data1_loader


def get_test_data(data,pt_feature):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        temp_save.append(pt_feature[:, data[j].index_x])
        temp_save.append(pt_feature[:, data[j].index_y + 490])
        x.append([temp_save])
        y.append(data[j].value)
    test_x = Variable(torch.FloatTensor(np.array(x))).cuda()
    test_y = torch.IntTensor(np.array(y)).cuda()
    return test_x, test_y
# data load,i--miRna  j--disease,
# use       pt[i]
#       pt[:,490+j]
#     miRna matrix[i] A[i]
#     disease matrix[j]  A[:,j]
# as feature
def load_data2(data,pt_feature,A,Rna_matrix,disease_matrix,BATCH_SIZE):
    x = []
    y = []
    for j in range(len(data)):
        temp_save = []  # cat features
        x_A = data[j].index_x
        y_A = data[j].index_y
        rna_disease = np.concatenate((Rna_matrix[x_A],A[x_A]),axis=1)
        disease_rna = np.concatenate((disease_matrix[y_A],A[:,y_A]),axis=1)
        temp_save.append(rna_disease)
        temp_save.append(disease_rna)
        temp_save.append(pt_feature[:,x_A])
        temp_save.append(pt_feature[:,y_A+490])
        x.append([temp_save])
        y.append(data[j].value)
    x = torch.FloatTensor(np.array(x))
    print(x.size())
    y = torch.IntTensor(np.array(y))
    torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
    data2_loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=False,              # random shuffle for training
            num_workers=1,              # subprocesses for loading data
        )
    return data2_loader
# CNN Framwork
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # now 1, 2, 816
                in_channels=1,  # input height
                out_channels=20,  # n_filters
                kernel_size=2,  # filter size
                stride=1,  # filter movement/step
                padding=1,  # padding
            ),  # may 16 * 3 * 817
            nn.Dropout(0.5),
            nn.BatchNorm2d(20),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),  # may 16 * 2 * 816


        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 32, 2, 1, 0),  # may 32 * 1 * 815
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,4), 1, 0),  # may 32 * 1 * 812
            #nn.BatchNorm2d(32)
            #nn.Dropout(0.2)
        )

        #self.out = nn.Sequential(nn.Linear(32 * 1 * 812,2),nn.BatchNorm1d(2),nn.Dropout(0.5),nn.Sigmoid())
        self.out = nn.Linear(32 * 1 * 812, 2)


        #self.softmax = torch.nn.Softmax()
    def forward(self,x):
        x = self.conv1(x)  # batch_size(50) * 16 * 4 * 492
        #print(x.size())
        #x = self.dropout(x)
        x = self.conv2(x)  # may
        #print(x.size())
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out = self.out(x)
        #print(torch.is_tensor(x))
        #out = torch.FloatTensor(out)
        #out = self.softmax(out)
        #print(out.size())
        return out

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
    if _scheduler == 'None':
        scheduler = None
    elif _scheduler == 'lambda':
        lambda2 = lambda epoch: 0.85 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda2)
    elif _scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif _scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    loss_func = nn.CrossEntropyLoss()
    #exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(Epoch):
        if (scheduler != None):
            scheduler.step()
        for step, (x, y) in enumerate(train_loader):
            model.train()
            length_train_loader = len(train_loader)
            if torch.cuda.is_available():
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()
            else:
                b_x = Variable(x)
                b_y = Variable(y)

            output = model(b_x)  # FloatTensor of size 50x2
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((step+1) % length_train_loader == 0) and (epoch == Epoch-1):
            #if ((step + 1) % length_train_loader == 0):
                model.eval()
                same_number = 0
                same_number_length = 0
                for step, (xx, yy) in enumerate(dev_loader):
                    dev_x = Variable(xx).cuda()
                    dev_y = yy.cuda().int()
                    dev_output = model(dev_x)
                    pred_y = torch.max(dev_output, 1)[1].data.squeeze().int()

                    same_number += sum(pred_y == dev_y)
                    same_number_length += dev_y.size(0)

                accuracy = same_number / float(same_number_length)
                context ='Epoch: ', epoch, '| train loss: %.6f' % loss.data[0], '| dev accuracy: %.4f' % accuracy
                #print(context)
                save_to_file2('birwcnntest.txt', str(context))


def train3(model,Epoch,train_loader,dev_loader,current_optim,w_decay,_scheduler,_LR):

    if current_optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=_LR, weight_decay=w_decay)
    elif current_optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=_LR, weight_decay=w_decay)
    elif current_optim == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=_LR, weight_decay=w_decay)
    elif current_optim == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=_LR, weight_decay=w_decay)
    if _scheduler == 'None':
        scheduler = None
    elif _scheduler == 'lambda':
        lambda2 = lambda epoch: 0.85 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda2)
    elif _scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif _scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    loss_func = nn.CrossEntropyLoss()
    #exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(Epoch):
        if (scheduler != None):
            scheduler.step()
        for step, (x, y) in enumerate(train_loader):
            model.train()
            length_train_loader = len(train_loader)
            if torch.cuda.is_available():
                b_x = Variable(x).cuda()
                b_y = Variable(y).cuda()
            else:
                b_x = Variable(x)
                b_y = Variable(y)

            output = model(b_x)  # FloatTensor of size 50x2
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if ((step+1) % length_train_loader == 0) and (epoch == Epoch-1):
            if (step  % 200 == 0):
                model.eval()
                same_number = 0
                same_number_length = 0
                for _step, (xx, yy) in enumerate(dev_loader):
                    dev_x = Variable(xx).cuda()
                    dev_y = yy.cuda().int()
                    dev_output = model(dev_x)
                    pred_y = torch.max(dev_output, 1)[1].data.squeeze().int()

                    same_number += sum(pred_y == dev_y)
                    same_number_length += dev_y.size(0)

                accuracy = same_number / float(same_number_length)
                context ='Epoch: ', epoch, '| train loss: %.6f' % loss.data[0], '| dev accuracy: %.4f' % accuracy
                print(context)
                #save_to_file2('birwcnntest.txt', str(context))
# ROC
# draw ROC
def draw_roc_column1(columns,column_test_data,pt_feature,BATCH_SIZE):

    area = 0.0
    count_have_positive = columns
    Y_TPR_all_columns = []
    X_FPR_all_columns = []
    area_all_column = []
    # according to column
    for i in range(columns):
        test_loader = load_data1(column_test_data[i],pt_feature,BATCH_SIZE)
        #test_out = torch.FloatTensor()
        #cnn.eval()
        pred_probability = []
        predict_values = []
        for step, (x, y) in enumerate(test_loader):
            if torch.cuda.is_available():
                b_x = Variable(x).cuda()
                #b_y = Variable(y).cuda()
            else:
                b_x = Variable(x)
                #b_y = Variable(y)
            out = cnn(b_x)

            #print(F.softmax(out,dim=1).data.squeeze().cpu().numpy())
            #print(torch.max(F.softmax(out,dim=1), 1)[0].data.squeeze().cpu().numpy())
            #print(torch.max(F.softmax(out,dim=1), 1)[1].data.squeeze().int().cpu().numpy())
            pred_one_probability = F.softmax(out,dim=1)[:,1].data.squeeze().cpu().numpy()
            predict_value = torch.max(F.softmax(out,dim=1), 1)[1].data.squeeze().int().cpu().numpy()  # the index of the high probability, predicted value
            pred_probability = np.concatenate((pred_probability,pred_one_probability),axis=0)
            predict_values = np.concatenate((predict_values,predict_value),axis=0)
        # according column to their initial column, and compute every column samples and its positive samples
        length_test = len(column_test_data[i])
        row = 1
        count_positive = np.array([0,0])  # [[count_column_number,count_positive_number] [] ...]
        for j in range(length_test):
            column_test_data[i][j].add_probability_predict(pred_probability[j], predict_values[j])
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
        print(i)
        area += area_column
    print("----")
    print(len(X_FPR_all_columns))
    print(count_have_positive)
    average_column_area = area/count_have_positive

    return Y_TPR_all_columns,X_FPR_all_columns,area_all_column,average_column_area


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

# get same point of TPR,FPR in columns,not contain all 0 column
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
def validation_test(cnn,alpha, k,batchsize,EPOCH,LR,dev_batchsize,l,r):
    all_k_Sd_associated, all_k_count_positive, all_k_test_data, A, num,\
    A_length, sava_association_A, all_k_development, k_pt_features,Sm, Sd = data_partitioning(alpha, k,l,r)
    X_all_k_FPR = []
    Y_all_k_TPR = []
    area_all_k = []
    for i in range(k):
        train_loader = load_data1(all_k_Sd_associated[i],k_pt_features[i], BATCH_SIZE=batchsize)
        # test_x, test_y = get_test_data(all_k_validation[0][:])
        dev_loader = load_data1(all_k_development[i],k_pt_features[i], BATCH_SIZE=dev_batchsize)
        cnn.train()
        train3(cnn,EPOCH, train_loader, dev_loader, 'Adam',0,'None',LR)


        cnn.eval()
        test_data = all_k_test_data[i]
        x_all_FPR, y_all_TPR,current_k_area = get_current_AUC_ROC(cnn,test_data,len(A[0]),k_pt_features[i],i)  # all column
        print(current_k_area)
        if(i!= k-1):
            X_all_k_FPR.append(x_all_FPR)
            Y_all_k_TPR.append(y_all_TPR)
        area_all_k.extend([current_k_area])
    #get_all_k_ROC(X_all_k_FPR,Y_all_k_TPR,x_all_FPR,y_all_TPR,k)
    print("K Avreage AUC", np.mean(area_all_k))
    pass

# files operator
def save_to_file2(file_name, contents):
    with open(file_name, 'a') as f:
        f.write(contents + '\n')

# save_to_file2('a.txt', a)


# test
def test(k):





    which_optim = {'Adam': 'Adam', 'SGD': 'SGD', 'Adagrad': 'Adagrad','ASGD':'ASGD'}
    which_alpha = {'0.1': 0.1,'0.2':0.2}
    which_maxcountrw = {'10': 10,'15': 15,'20': 20,'50': 50}
    which_Scheduler ={'None':'None','lambda': 'LambdaLR','step': 'StepLR','Reduce':'ReduceLROnPlateau'}
    which_weight_decay = {'0': 0,'0.1': 0.1,'0.5': 0.5,'0.9': 0.9,'0.01': 0.01,'0.05': 0.05,'0.001': 0.001,'0.0001': 0.0001}
    which_LR = {'0.005': 0.005,'0.001': 0.001,'0.0005': 0.0005,'0.0001': 0.0001}
    which_EPOCH = {'10': 10, '20': 20, '30': 30, '40': 40, '50': 50, '60': 60}
    which_BatchSize = {'30': 30, '50': 50, '70': 70, '90': 90, '110': 110}
    which_l_r = {'10': 10, '20': 20, '30': 30}
    # which_LR =np.arange(0,0.01,0.0001)
    # batchsize
    # Epoch
    time = 0
    i = 0
    for alpha in which_alpha.values():  # 明确指出遍历字典的值
        for l_r in which_l_r.values():
            l = r = l_r
            all_k_Sd_associated, all_k_count_positive, all_k_test_data, A, num, \
            A_length, sava_association_A, all_k_development, k_pt_features, Sm, Sd = data_partitioning(alpha, k, l,r)
            dev_loader = load_data1(all_k_development[i], k_pt_features[i], BATCH_SIZE=80)
        for batchsize in which_BatchSize.values():
            train_loader = load_data1(all_k_Sd_associated[i], k_pt_features[i], BATCH_SIZE=batchsize)
            for epoch in which_EPOCH.values():
                for _optim in which_optim.values():
                    for w_decay in which_weight_decay.values():
                        for _scheduler in which_Scheduler.values():
                            for _LR in which_LR.values():
                                cnn = CNN().cuda()
                                # X_all_k_FPR = []
                                # Y_all_k_TPR = []
                                # area_all_k = []

                                cnn.train()
                                train2(cnn, epoch, train_loader, dev_loader, _optim, w_decay, _scheduler, _LR)

                                cnn.eval()
                                test_data = all_k_test_data[i]
                                x_all_FPR, y_all_TPR, current_k_area = get_current_AUC_ROC(cnn, test_data, len(A[0]),
                                                                                           k_pt_features[i],
                                                                                           i)  # all column
                                d = 'Time:{}, AUC:{}, alpha:{}, BatchSize:{}, Epoch:{}, optim:{}, weightdecay:{}, Scheduler:{}, LR:{}'.format(
                                    time, current_k_area, alpha, batchsize, epoch, _optim, w_decay, _scheduler,
                                    _LR)
                                save_to_file2('birwcnntest.txt', d)
                                #print(d)
                                del cnn
                                time += 1






#
if __name__ == '__main__':
    # model
    if torch.cuda.is_available():
        cnn = CNN().cuda()
    else:
        cnn = CNN()

    # parameters
    alpha = 0.1
    k =10
    # max_count_rw = 10
    batchsize = 30
    EPOCH = 50
    LR = 0.0001
    #
    l=r=20

    # cross validation
    validation_test(cnn,alpha, k,batchsize,EPOCH,LR,50,l,r)
    # #test(alpha, k, max_count_rw, batchsize, EPOCH, LR, 50)
    # test(k)





















