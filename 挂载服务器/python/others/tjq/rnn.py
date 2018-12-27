import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.002           # learning rate


def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        print(x.size())#torch.Size([1, 10, 1])
        r_out, h_state = self.rnn(x, h_state)
        print(r_out.size())#torch.Size([1, 10, 32])
        print(h_state.size())#torch.Size([1, 1, 32])



        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state


class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

# rnn = RNN()
# print(rnn)
net = lstm_reg(1, 4)

optimizer   = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
criterion  = nn.MSELoss()

h_state = None      # for initial hidden state


name = ['quote000002', 'quote000006', 'quote000007', 'quote000011', 'quote000014',  'quote000031', 'quote600533', 'quote600641', 'quote900911']
colour = ['blue','green','cyan','magenta','yellow','red','black','maroon','salmon','C1','C0']
# data1 = pd.read_csv('./house/'+name[i]+'/lrb.csv',encoding='gbk',header=None)


for step in range(len(name)):
    data1 = pd.read_csv('./house/' + name[step] + '/lrb.csv', encoding='gbk', header=None)

    data1 = np.array(data1.iloc[1,1:].astype(int))

    data_X,data_Y = create_dataset(data1,1)
    train_size = int(len(data_X) * 0.9)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    train_X = train_X.reshape(-1, 1, 1)
    train_Y = train_Y.reshape(-1, 1, 1)
    test_X = test_X.reshape(-1, 1, 1)

    train_x = torch.FloatTensor(train_X)
    train_y = torch.FloatTensor(train_Y)
    test_x = torch.FloatTensor(test_X)

    for e in range(1000):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = net(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if (e + 1) % 100 == 0:  # 每 100 次输出结果
            # print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))
    net = net.eval()  # 转换成测试模式
    Var_test_x = Variable(test_x)
    out = net(Var_test_x)
    print(out)
    print(test_Y)


